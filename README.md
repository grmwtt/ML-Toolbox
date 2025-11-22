# Machine Learning & Deep Learning Benchmarks

## Overview
A comprehensive portfolio of machine learning implementations, benchmarking studies, and deep learning architectures. This repository bridges the gap between statistical theory and production-grade engineering, featuring rigorous comparisons between classical algorithms (XGBoost, LightGBM) and modern deep learning frameworks (PyTorch).

**Author:** Griffin Witt <br>
**Academic Focus:** Computational Modeling & Data Analytics | M.S. Data Science Candidate at Virginia Tech
<br>
## Repository Architecture <br>
This project follows a modular production structure rather than a flat notebook list.

```text
ml-benchmarks/
├── benchmarks/            # Comparative studies (e.g., XGBoost vs PyTorch MLP)
├── src/                   # Reusable Python modules (OOP style)
│   ├── models/            # Custom PyTorch architectures
│   ├── pipelines/         # Sklearn & PyTorch DataLoaders
│   └── visualization/     # Custom plotting for decision boundaries/loss landscapes
├── notebooks/             # Exploratory analysis & prototyping
├── tests/                 # Unit tests for custom implementations
└── environment.yml        # Reproducible Conda environment
