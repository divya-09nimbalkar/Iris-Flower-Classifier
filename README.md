# Iris-Flower-Classifier
Classifies Iris flowers into 3 species using 5 ML algorithms with cross-validation comparison and feature importance analysis.

## Tech Stack
`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

## Models Compared
| Model               | Test Accuracy | CV Score |
|---------------------|--------------|----------|
| K-Nearest Neighbors | ~100%        | ~96.7%   |
| Decision Tree       | ~96.7%       | ~95.3%   |
| Random Forest       | ~100%        | ~96.7%   |
| SVM (RBF Kernel)    | ~100%        | ~98.0%   |
| Logistic Regression | ~100%        | ~97.3%   |

## Features
- 5-model comparison with 5-fold cross-validation
- Scatter plot matrix for all feature pairs
- Confusion matrix visualization
- Random Forest feature importance
- Live prediction for new samples

## How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python iris_classifier.py
```

## Output Files
- `iris_scatter.png` — feature distribution by species
- `confusion_matrix.png` — prediction accuracy
- `feature_importance.png` — which features matter most

---
**Author:** Divya Nimbalkar | [GitHub](https://github.com/divya-09nimbalkar) | [LinkedIn](https://www.linkedin.com/in/divya-nimbalkar09/)
