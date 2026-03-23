"""Setup configuration for the Credit Spread Analysis & Prediction Platform."""

from setuptools import setup, find_packages

setup(
    name="credit-spread-analyzer",
    version="0.1.0",
    description="Credit Spread Analysis & Prediction Platform",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "plotly>=5.15",
        "seaborn>=0.12",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "shap>=0.42",
        "statsmodels>=0.14",
        "hmmlearn>=0.3",
        "fredapi>=0.5",
        "yfinance>=0.2",
        "torch>=2.0",
        "streamlit>=1.28",
    ],
)
