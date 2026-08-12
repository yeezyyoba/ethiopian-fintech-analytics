"""
credit_risk_llm_layer.py  (Google AI Studio / Gemini version)

An LLM-based "second opinion" layer for the Ethiopian Fintech Analytics
Platform credit risk model (Notebooks 01/03/04/05 -- XGBoost, AUC-ROC
0.8842, trained on Give Me Some Credit).

The XGBoost model + SHAP stay the primary, numeric decision-makers.
This layer takes an applicant's raw feature values PLUS the model's
predicted default probability and its SHAP-derived top contributing
features, and asks Gemini to independently reason about the case in
plain language for a loan officer. Returns clean structured JSON:

    {
        "is_suspicious": bool,   # here: "flag for manual review"
        "reason": str,           # loan-officer-readable explanation
        "risk_score": float      # 0-100, Gemini's own independent read
    }

Uses Gemini's structured outputs (response_mime_type="application/json"
+ response_schema) -- a single request, no multi-turn tool loop.

Setup:
    pip install google-genai
    export GEMINI_API_KEY=your_key_here   # from Google AI Studio
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from google import genai
from google.genai import types


MODEL = "gemini-3.6-flash"

# Matches the final feature set from Notebook 03 (after dropping raw
# delinquency columns and NumberOfDependents).
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "delinquency_score",
    "debt_to_income",
    "is_young_borrower",
    "is_senior_borrower",
    "high_utilization",
    "total_past_due",
    "has_delinquency",
    "has_dependents",
]

# Gemini's response_schema uses JSON Schema-like types but with
# capitalized type names (STRING/NUMBER/BOOLEAN/OBJECT).
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_suspicious": {
            "type": "BOOLEAN",
            "description": "Whether this applicant should be flagged for manual underwriter review.",
        },
        "reason": {
            "type": "STRING",
            "description": (
                "Plain-language explanation a loan officer can read in a "
                "few seconds: what stands out in the applicant's profile, "
                "whether it agrees or disagrees with the model's default "
                "probability, and why."
            ),
        },
        "risk_score": {
            "type": "NUMBER",
            "description": (
                "0-100. This is Gemini's own independent risk read of the "
                "applicant -- NOT a copy of the model's predicted "
                "probability."
            ),
        },
    },
    "required": ["is_suspicious", "reason", "risk_score"],
}


@dataclass
class CaseInput:
    """Everything the LLM layer needs about one applicant."""

    case_id: str
    raw_data: dict[str, Any]           # applicant's raw feature values
    model_score: float                 # predict_proba()[:, 1] -- probability of default, 0-1
    top_features: list[dict[str, Any]] = field(default_factory=list)
    # e.g. [{"feature": "has_delinquency", "value": 1, "shap_value": 0.41}, ...]
    model_score_scale: str = "0-1 (probability of default)"

    @classmethod
    def from_shap_row(
        cls,
        case_id: str,
        row: pd.Series,
        proba: float,
        shap_row: np.ndarray,
        features: list[str] = FEATURES,
        top_n: int = 5,
    ) -> "CaseInput":
        """Build a CaseInput directly from Notebook 05's SHAP output.

        Usage inside Notebook 05, after computing `shap_values`:

            case = CaseInput.from_shap_row(
                case_id=str(X_sample.index[high_risk_i]),
                row=X_sample.iloc[high_risk_i],
                proba=proba_sample[high_risk_i],
                shap_row=shap_values[high_risk_i],
                features=FEATURES,
            )
        """
        contributions = sorted(
            zip(features, shap_row),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )[:top_n]
        top_features = [
            {
                "feature": feat,
                "value": row[feat].item() if hasattr(row[feat], "item") else row[feat],
                "shap_value": round(float(val), 4),
            }
            for feat, val in contributions
        ]
        return cls(
            case_id=case_id,
            raw_data={f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in features},
            model_score=float(proba),
            top_features=top_features,
        )


@dataclass
class LLMOpinion:
    case_id: str
    is_suspicious: bool
    reason: str
    risk_score: float
    raw_response: Optional[dict] = None


def _build_prompt(case: CaseInput) -> str:
    features_block = (
        json.dumps(case.top_features, indent=2)
        if case.top_features
        else "(none provided)"
    )
    return f"""You are assisting a loan officer as a SECOND, independent reviewer of
a credit default risk assessment. An XGBoost model (AUC-ROC 0.88) has
already scored this applicant. Your job is not to defer to that score,
but to reason about the applicant's profile yourself and say whether
you agree, disagree, or want to add nuance -- in plain language a loan
officer (not a data scientist) can use.

APPLICANT ID: {case.case_id}

APPLICANT FEATURES:
{json.dumps(case.raw_data, indent=2)}

MODEL OUTPUT:
- Predicted probability of default: {case.model_score:.2%} (scale: {case.model_score_scale})
- Top SHAP-contributing features (feature, value, shap_value -- positive
  shap_value pushes toward default, negative pushes away):
{features_block}

Feature glossary:
- has_delinquency / delinquency_score / total_past_due: history of missed payments (most decisive factor per prior SHAP analysis)
- high_utilization / RevolvingUtilizationOfUnsecuredLines: how much of available credit is currently used
- debt_to_income / DebtRatio: debt burden relative to income
- is_young_borrower (<30) / is_senior_borrower (>=60): age-based risk flags
- has_dependents, NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines: structural/background factors, generally minor
- MonthlyIncome: no currency is specified in this dataset -- treat it as a unitless figure and do NOT invent or assume a currency (e.g. do not say "USD" or "ETB" or any other currency).

Instructions:
- Do not state or imply facts not present in the data above (e.g.
  currency, country, institution type). If something is unspecified,
  do not fill it in with an assumption.
- Look for patterns or context the model's score alone wouldn't
  surface (e.g. a plausible explanation for a flagged feature, or a
  red flag the listed features don't fully capture).
- If you agree with the model, say so briefly and explain why.
- If you disagree, explain specifically what makes you read it
  differently.
- Keep "reason" concise: 2-4 sentences, loan-officer-readable, no
  jargon dump.
- "risk_score" is YOUR independent 0-100 read, not a copy of the
  model's predicted probability.
"""


def get_llm_opinion(
    case: CaseInput,
    client: Optional[genai.Client] = None,
) -> LLMOpinion:
    """Call Gemini with structured outputs and return a parsed LLMOpinion."""
    client = client or genai.Client()  # reads GEMINI_API_KEY from env

    response = client.models.generate_content(
        model=MODEL,
        contents=_build_prompt(case),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )

    # response.text is guaranteed valid JSON matching RESPONSE_SCHEMA.
    parsed = json.loads(response.text)

    return LLMOpinion(
        case_id=case.case_id,
        is_suspicious=parsed["is_suspicious"],
        reason=parsed["reason"],
        risk_score=parsed["risk_score"],
        raw_response=parsed,
    )


def review_batch(
    cases: list[CaseInput],
    client: Optional[genai.Client] = None,
) -> list[LLMOpinion]:
    """Run get_llm_opinion over a batch of cases. Sequential; parallelize
    with concurrent.futures if you need throughput."""
    client = client or genai.Client()
    return [get_llm_opinion(case, client=client) for case in cases]


if __name__ == "__main__":
    # Example using the same shape of high-risk customer as Notebook 05's
    # waterfall plot example.
    example_case = CaseInput(
        case_id="applicant_00123",
        raw_data={
            "RevolvingUtilizationOfUnsecuredLines": 0.98,
            "age": 27,
            "DebtRatio": 0.61,
            "MonthlyIncome": 2100.0,
            "NumberOfOpenCreditLinesAndLoans": 4,
            "NumberRealEstateLoansOrLines": 0,
            "delinquency_score": 6,
            "debt_to_income": 1281.0,
            "is_young_borrower": 1,
            "is_senior_borrower": 0,
            "high_utilization": 1,
            "total_past_due": 3,
            "has_delinquency": 1,
            "has_dependents": 0,
        },
        model_score=0.83,
        top_features=[
            {"feature": "has_delinquency", "value": 1, "shap_value": 0.41},
            {"feature": "high_utilization", "value": 1, "shap_value": 0.27},
            {"feature": "delinquency_score", "value": 6, "shap_value": 0.19},
            {"feature": "is_young_borrower", "value": 1, "shap_value": 0.08},
            {"feature": "MonthlyIncome", "value": 2100.0, "shap_value": 0.05},
        ],
    )

    opinion = get_llm_opinion(example_case)
    print(json.dumps(opinion.__dict__, indent=2, default=str))
