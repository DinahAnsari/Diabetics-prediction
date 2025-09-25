## Diabetes Prediction App (PIMA Dataset)

This Streamlit app predicts the onset of diabetes using the PIMA Indian Diabetes dataset. It provides a clean UI to explore data, train models (SVM, KNN, AdaBoost), compare metrics, visualize results, and perform single-patient predictions from manual inputs.


## How to run

1) Install dependencies:
```bash
python -m pip install -r requirements.txt
```

2) Launch the app:
```bash
python -m streamlit run project.py
```

3) Open the app in your browser at `http://localhost:8501`.


## What the app does

- Load the provided `data.csv` or upload a CSV with the standard PIMA columns
  (`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`).
- Clean the data by treating medically-impossible zeros as missing for key fields and imputing with the median.
- Split into train/test sets and optionally standardize features.
- Train and evaluate a selected model (SVM, KNN, AdaBoost) with tunable hyperparameters.
- Display metrics (Accuracy, Sensitivity, Specificity), a Confusion Matrix, ROC Curve (when probabilities are available), and AdaBoost Feature Importance.
- Allow manual entry of patient measurements to get an instant prediction with probability.
- Compare SVM, KNN, and AdaBoost side-by-side in a grouped bar chart of Accuracy, Sensitivity, and Specificity.


## Sidebar controls (left panel)

- **Dataset source**: Choose the included `data.csv` or upload your own CSV with the same headers.
- **Model**:
  - **SVM**: Set `C` (regularization strength) and `kernel` (`rbf`, `linear`, `poly`, `sigmoid`).
  - **KNN**: Set number of neighbors `k` and `weights` (`uniform` or `distance`).
  - **AdaBoost**: Set number of `estimators` and `learning rate`.
- **Test size (%)**: Percentage of data held out for testing.
- **Random state**: Seed for reproducible splits.
- **Standardize features**: Recommended for SVM and KNN.
- **Train & Evaluate**: Runs training, testing, and displays all results.


## Main page sections

### Data Preview
- Shows the first rows of the dataset, the dataset shape, class balance for `Outcome` (0 = Non-diabetic, 1 = Diabetic), and summary statistics.

### Feature Correlations (heatmap)
- Displays pairwise correlations among numeric features (excluding the target `Outcome`).
- Darker red/blue indicates stronger positive/negative correlations.
- Helps identify multicollinearity and feature relationships.

### Performance Metrics
- **Accuracy**: Overall fraction of correct predictions on the test set.
- **Sensitivity (True Positive Rate, Recall for class 1)**: Of actual diabetic cases, how many were correctly identified. High sensitivity reduces missed positive cases.
- **Specificity (True Negative Rate)**: Of actual non-diabetic cases, how many were correctly identified. High specificity reduces false alarms.

### Confusion Matrix
- 2×2 grid summarizing predictions vs. actuals.
  - Top-left: True Negatives (0 predicted as 0)
  - Top-right: False Positives (0 predicted as 1)
  - Bottom-left: False Negatives (1 predicted as 0)
  - Bottom-right: True Positives (1 predicted as 1)
- Visualizes the trade-off between capturing positives and avoiding false alarms.

### ROC Curve (with AUC)
- Plots True Positive Rate vs. False Positive Rate at various thresholds.
- **AUC (Area Under Curve)** summarizes performance across thresholds (higher is better; 0.5 ≈ random, 1.0 = perfect).
- Shown when the model provides probabilities (SVM configured with `probability=True`, KNN and AdaBoost support `predict_proba`).

### AdaBoost Feature Importance
- Bar chart of feature importances learned by AdaBoost (with decision stump base learners).
- Higher bars indicate features that contributed more to the model’s decisions.

### Model Comparison (SVM vs KNN vs AdaBoost)
- After training, the app shows a grouped bar chart comparing the three models on the same test split.
- Bars display: **Accuracy**, **Sensitivity**, **Specificity** per model.
- Uses the current preprocessing and sidebar hyperparameters. Adjust settings and re-train to update the chart.

### Manual Prediction (single patient)
- After training a model, a form appears to input:
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
    `BMI`, `DiabetesPedigreeFunction`, `Age`.
- Defaults and ranges are based on the dataset’s median and min–max to guide reasonable entries.
- Click **Predict Outcome** to get:
  - Predicted class: `0` (Non-diabetic) or `1` (Diabetic)
  - Probability of diabetes (when the model supports it)
- The app automatically applies the same preprocessing (e.g., scaling) used during training.


## Model selection guidance (client-friendly)

- **SVM**
  - Strengths: Often strong accuracy; robust with proper scaling; works well on medium-size feature spaces.
  - Considerations: Kernel selection matters; training can be slower on very large datasets.

- **KNN**
  - Strengths: Simple, intuitive; no explicit training phase.
  - Considerations: Sensitive to scaling and choice of `k`; slower predictions on large datasets.

- **AdaBoost**
  - Strengths: Ensemble that improves predictive balance; often higher sensitivity without sacrificing specificity.
  - Considerations: Can be sensitive to noisy data; tuning `n_estimators` and `learning_rate` helps.


## Interpreting results (client-friendly)

- Aim for a balance of **Sensitivity** and **Specificity** depending on clinical priorities.
  - For screening, higher sensitivity may be preferred to reduce missed diabetic cases.
  - For confirmatory tests, higher specificity reduces false alarms.
- Use the **ROC Curve** and **AUC** to compare models across thresholds.
- Consult **Feature Importance** (AdaBoost) to see which features influence decisions most (e.g., `Glucose`, `BMI`).


## Custom datasets

To use your own data, upload a CSV with these columns:
```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```
`Outcome` must be 0/1. The app will apply the same preprocessing and evaluation flow.


## Limitations and disclaimers

- This app is a decision support tool, not a medical device. Predictions should not replace clinical judgment.
- Performance depends on data quality and representativeness.
- For deployment, consider model validation on larger, diverse cohorts and align with compliance requirements.


