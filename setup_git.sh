#!/bin/bash
# ─────────────────────────────────────────────────────────────
# setup_git.sh
# Run this ONCE to initialize the project as a GitHub repository
# ─────────────────────────────────────────────────────────────

echo "Setting up Git repository..."

# Initialize git
git init

# Create .gitkeep files so empty data folders are tracked
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep

# Create __init__.py files so src/ works as a Python package
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/visualization/__init__.py
touch tests/__init__.py

# Stage everything
git add .

# First commit
git commit -m "feat: initial project structure and starter code

- Add full folder structure (data, notebooks, src, dashboard, reports)
- Add README with project overview and weekly progress tracker
- Add requirements.txt with all dependencies
- Add .gitignore for Python/data science project
- Add src/data/loader.py — data loading utilities
- Add src/data/cleaner.py — cleaning pipelines for credit and fraud data
- Add src/features/engineer.py — feature engineering functions
- Add src/models/trainer.py — model training and evaluation utilities
- Add src/visualization/plots.py — reusable EDA and model plots
- Add notebooks/01_EDA_credit_risk.ipynb — Week 1 starter notebook
- Add docs/data_download.md — Kaggle dataset download instructions"

echo ""
echo "✓ Git initialized with first commit."
echo ""
echo "Next steps:"
echo "  1. Create a new repo on GitHub: github.com/new"
echo "     Name it: ethiopian-fintech-analytics"
echo "  2. Run these commands:"
echo "     git remote add origin https://github.com/YOUR_USERNAME/ethiopian-fintech-analytics.git"
echo "     git branch -M main"
echo "     git push -u origin main"
echo ""
echo "Your project is live on GitHub!"
