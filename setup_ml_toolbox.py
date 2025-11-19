#!/usr/bin/env python3
"""
ML Toolbox Setup Script
========================
This script helps set up your ML Toolbox environment.
"""

import os
import sys
import subprocess

def setup_environment():
    """Set up the ML Toolbox environment."""
    
    print("="*60)
    print("ML TOOLBOX SETUP")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("WARNING: Python 3.7+ is recommended")
    
    # Install requirements
    print("\nInstalling required packages...")
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "plotly>=5.3.0",
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
        "tqdm>=4.62.0"
    ]
    
    for package in requirements:
        print(f"  - {package}")
    
    response = input("\nDo you want to install these packages? (y/n): ")
    if response.lower() == 'y':
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + requirements, check=True)
            print("\n✅ Packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error installing packages: {e}")
            print("You can manually install with: pip install -r requirements.txt")
    
    # Directory structure
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE")
    print("="*60)
    
    structure = """
    ml-toolbox/
    ├── README.md                          # Main documentation
    ├── requirements.txt                   # Python dependencies
    ├── supervised/
    │   ├── classification/
    │   │   └── logistic_regression.ipynb  # Complete template
    │   └── regression/
    ├── unsupervised/
    │   ├── clustering/
    │   └── dimensionality_reduction/
    └── utils/
        ├── evaluation_metrics.py          # Custom metrics
        └── visualization.py               # Plotting utilities
    """
    
    print(structure)
    
    print("\n" + "="*60)
    print("GETTING STARTED")
    print("="*60)
    print("""
    1. Navigate to the ml-toolbox directory:
       cd ml-toolbox
       
    2. Start Jupyter Notebook:
       jupyter notebook
       
    3. Open logistic_regression.ipynb to see the template structure
    
    4. For each new algorithm:
       - Copy the template structure
       - Implement the algorithm from scratch
       - Add diagnostics and visualizations
       - Compare with sklearn
    
    5. Use the utility modules for consistent metrics and plots:
       from utils.evaluation_metrics import ClassificationMetrics
       from utils.visualization import MLVisualizer
    """)
    
    print("\n" + "="*60)
    print("IMPLEMENTATION CHECKLIST")
    print("="*60)
    print("""
    ✅ Completed:
    - Logistic Regression (template)
    - Evaluation Metrics Module
    - Visualization Module
    
    📝 To Implement:
    
    Classification:
    [ ] Naive Bayes (Gaussian, Multinomial, Bernoulli)
    [ ] Decision Trees (ID3, C4.5, CART)
    [ ] Random Forest
    [ ] Support Vector Machines
    [ ] k-Nearest Neighbors
    [ ] Neural Network (Basic MLP)
    
    Regression:
    [ ] Linear Regression (OLS, Gradient Descent)
    [ ] Polynomial Regression
    [ ] Ridge Regression (L2)
    [ ] Lasso Regression (L1)
    [ ] Elastic Net
    [ ] Gradient Boosting
    
    Clustering:
    [ ] K-Means
    [ ] Hierarchical Clustering
    [ ] DBSCAN
    [ ] Gaussian Mixture Models
    
    Dimensionality Reduction:
    [ ] PCA
    [ ] LDA
    [ ] t-SNE
    [ ] Autoencoders
    """)
    
    print("\n✨ Setup complete! Happy coding!")
    print("="*60)

if __name__ == "__main__":
    setup_environment()
