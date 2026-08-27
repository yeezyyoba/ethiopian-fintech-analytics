"""
app.py

Streamlit UI for the Ethiopian Fintech Analytics Platform credit risk
model. Talks to predict.py only — no feature-engineering or model
logic lives in this file.

Run from the project root:
    streamlit run src/app.py

Gemini "second opinion" layer is intentionally NOT wired in yet —
see the TODO near the bottom for where it plugs in later.
"""

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from predict import predict

st.set_page_config(
    page_title="Ethiopian Fintech — Credit Risk",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Risk Assessment")
st.caption(
    "XGBoost model (AUC-ROC 0.88) trained on the Give Me Some Credit "
    "dataset, explained with SHAP."
)

with st.sidebar:
    st.header("Applicant Information")

    revolving_util = st.slider(
        "Revolving Utilization of Unsecured Lines",
        min_value=0.0, max_value=2.0, value=0.3, step=0.01,
        help="Total balance on credit cards / credit limits.",
    )
    age = st.number_input("Age", min_value=18, max_value=96, value=35)
    dpd_30_59 = st.number_input(
        "Times 30–59 days past due (last 2 yrs)", min_value=0, value=0
    )
    debt_ratio = st.slider(
        "Debt Ratio", min_value=0.0, max_value=2.0, value=0.3, step=0.01,
        help="Monthly debt payments / monthly gross income.",
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0.0, value=4000.0, step=100.0
    )
    open_credit_lines = st.number_input(
        "Open Credit Lines & Loans", min_value=0, value=6
    )
    dpd_90 = st.number_input(
        "Times 90+ days late", min_value=0, value=0
    )
    real_estate_loans = st.number_input(
        "Real Estate Loans / Lines", min_value=0, value=1
    )
    dpd_60_89 = st.number_input(
        "Times 60–89 days past due (last 2 yrs)", min_value=0, value=0
    )
    dependents = st.number_input("Number of Dependents", min_value=0, value=0)

    submitted = st.button("Assess Risk", type="primary", use_container_width=True)

if not submitted:
    st.info("Fill in the applicant's details on the left, then click **Assess Risk**.")
    st.stop()

raw_input = {
    "RevolvingUtilizationOfUnsecuredLines": revolving_util,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": dpd_30_59,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
    "NumberOfTimes90DaysLate": dpd_90,
    "NumberRealEstateLoansOrLines": real_estate_loans,
    "NumberOfTime60-89DaysPastDueNotWorse": dpd_60_89,
    "NumberOfDependents": dependents,
}

with st.spinner("Scoring applicant..."):
    try:
        result = predict(raw_input)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

proba = result["probability_of_default"]
top_features = result["shap_top_features"]

# ── Headline score ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Probability of Default", f"{proba:.1%}")
    if proba >= 0.5:
        st.error("⚠️ High Risk — recommend manual review")
    elif proba >= 0.2:
        st.warning("🟡 Moderate Risk")
    else:
        st.success("✅ Low Risk")

with col2:
    st.subheader("Top factors driving this score")
    shap_df = pd.DataFrame(top_features)
    shap_df["direction"] = shap_df["shap_value"].apply(
        lambda v: "Increases risk" if v > 0 else "Decreases risk"
    )
    st.dataframe(
        shap_df[["feature", "value", "shap_value", "direction"]],
        hide_index=True,
        use_container_width=True,
    )

# ── SHAP waterfall-style bar chart ───────────────────────────────────
st.subheader("Why this score? (SHAP contributions)")

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in shap_df["shap_value"]]
ax.barh(shap_df["feature"], shap_df["shap_value"], color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP value (pushes toward default →)")
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig)

with st.expander("View all engineered features"):
    st.json(result["raw_engineered_features"])

# ── TODO: Gemini second opinion ──────────────────────────────────────
# Once ready to wire this in:
#   from credit_risk_llm_layer_gemini import CaseInput, get_llm_opinion
#   if st.button("Get AI second opinion"):
#       case = CaseInput(
#           case_id="ui-applicant",
#           raw_data=result["raw_engineered_features"],
#           model_score=proba,
#           top_features=top_features,
#       )
#       opinion = get_llm_opinion(case)
#       st.write(opinion.reason)
# Requires GEMINI_API_KEY set in the environment / Streamlit secrets.