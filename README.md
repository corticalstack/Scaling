# 📊 ML Scaling Techniques Demo

A demonstration of various scaling techniques for machine learning models using the KDD Cup 1999 Intrusion Detection System dataset.

## 🔍 Description

This repository contains code that demonstrates the impact of different scaling techniques on machine learning model performance, particularly for datasets with imbalanced classes and skewed features. The project uses the KDD Cup 1999 IDS dataset, which is a network intrusion detection dataset with normal and attack traffic patterns.

The code implements and visualizes the effects of various scaling methods on feature distributions and evaluates their impact on classification performance using different machine learning algorithms.

## ⚙️ Prerequisites

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## ✨ Features

- Implementation of multiple scaling techniques:
  - Standard Scaling
  - Normalization
  - Min-Max Scaling
  - Binarization
  - Robust Scaling
  - Power Transformation
  - Quantile Transformation (normal and uniform distributions)
- Visualization of feature distributions before and after scaling
- Performance evaluation using multiple classifiers:
  - Random Forest
  - XGBoost
- Comprehensive metrics for imbalanced classification:
  - Precision
  - Recall
  - F1 Score
- Confusion matrix visualization for multi-class classification
- Outlier detection and removal

## 🚀 Setup Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/scaling-techniques-demo.git
   cd scaling-techniques-demo
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

3. Create a plots directory for saving visualizations:
   ```bash
   mkdir -p Scaling/plots
   ```

## 📝 Usage

Run the main script to see the effects of different scaling techniques:

```bash
cd Scaling
python main.py
```

This will:
1. Load and preprocess the KDD Cup dataset
2. Apply various scaling techniques to the features
3. Generate visualizations of feature distributions before and after scaling
4. Train classifiers on the scaled data
5. Evaluate and compare performance metrics
6. Save visualization plots to the `plots` directory

## 🧩 Dataset

The repository uses the KDD Cup 1999 10% dataset, which contains network connection records with 41 features. Each record is labeled as either normal or as a specific type of attack. The attacks fall into four main categories:
- DOS (Denial of Service)
- U2R (User to Root)
- R2L (Remote to Local)
- Probe

The code preprocesses this dataset by:
- Removing duplicates
- Cleaning label strings
- Creating binary and multi-class target variables
- Removing extreme outliers

## 📊 Visualization

The code generates two types of visualizations:
1. **KDE Plots**: Shows the distribution of selected features before and after applying each scaling technique
2. **Confusion Matrices**: Displays the classification performance for each combination of scaling technique and classifier

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
