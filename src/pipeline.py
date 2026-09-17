import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from preprocess import engineer_longitudinal_features

# ── Step 1: Load & Clean ──────────────────────────────────────────────────────
df = pd.read_csv('../data/oasis_longitudinal.csv')

df = df.drop(columns=['Hand', 'MRI ID'])
df['MR Delay (Years)'] = df['MR Delay'] / 365.25
df = df.drop(columns=['MR Delay'])

# ── Step 2: Longitudinal Feature Engineering ──────────────────────────────────
df = engineer_longitudinal_features(df)
df['baseline_MMSE'] = df.groupby('Subject ID')['MMSE'].transform('first')
df = df.drop(columns=['Subject ID'])

# ── Step 3: Encode Target ─────────────────────────────────────────────────────
df['Group'] = df['Group'].map({'Nondemented': 0, 'Converted': 1, 'Demented': 2})

X = df.drop(columns=['Group'])
y = df['Group']

# ── Step 4: Define Column Types ───────────────────────────────────────────────
categorical_cols = ['M/F']

numerical_cols = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR',
                  'eTIV', 'nWBV', 'ASF', 'MR Delay (Years)',
                  'MMSE_change', 'nWBV_change', 'CDR_change',
                  'baseline_MMSE', 'baseline_CDR',
                  'visit_count', 'Visit']

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline steps:
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
    ('num', KNNImputer(n_neighbors=5), numerical_cols)
])

base_models = [
    ('xgb', XGBClassifier(n_estimators=200, max_depth=4,
                           learning_rate=0.05, random_state=42,
                           eval_metric='mlogloss', verbosity=0,
                           base_score=0.5)),
    ('rf',  RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=True
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler',       StandardScaler()),
    ('smote',        SMOTE(random_state=42)),
    ('model',        stacking_model)
])

print("Training pipeline…")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred,
                             target_names=['Nondemented', 'Converted', 'Demented']))



                             # Extra code for SHAP analysis and model saving below 
                             

# ── Step 9: SHAP Analysis ─────────────────────────────────────────────────────
os.makedirs('../outputs', exist_ok=True)

# Transform training data through preprocessor + scaler only (skip SMOTE)
X_train_prep = pipeline.named_steps['scaler'].transform(
    pipeline.named_steps['preprocessor'].transform(X_train)
)

# Feature names after ColumnTransformer
ohe_names = pipeline.named_steps['preprocessor'] \
    .named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
feature_names = ohe_names + numerical_cols

# RandomForest base model from stacking (used for SHAP — XGBoost 3.x breaks SHAP multiclass)
rf_model = pipeline.named_steps['model'].estimators_[1]

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_prep)  # (n_samples, n_features, n_classes)

# Global summary — Demented class (most clinically relevant)
shap.summary_plot(shap_values[:, :, 2], X_train_prep, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('../outputs/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/shap_summary.png")

# Single patient waterfall — Demented class for first patient
shap_exp = explainer(X_train_prep)
shap.plots.waterfall(shap_exp[0, :, 2], show=False)
plt.tight_layout()
plt.savefig('../outputs/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/shap_waterfall.png")

# ── Step 10: Save Model ───────────────────────────────────────────────────────
import joblib
joblib.dump(pipeline, '../outputs/model.pkl')
print("Saved: outputs/model.pkl")


