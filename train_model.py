import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Relative directory resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils.preprocess import load_data, preprocess_data

def train_and_evaluate():
    data_path = os.path.join(CURRENT_DIR, 'data', 'machine_data.csv')
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Checking fallback...")
        # fallback to generate_data or alternative dataset
        fallback_path = os.path.join(CURRENT_DIR, '..', 'predictive_a_2', 'data', 'ai4i2020.csv')
        if os.path.exists(fallback_path):
            data_path = fallback_path
        else:
            print("Please run generate_data.py or provide a CSV in data/")
            return

    print(f"Loading data from: {data_path}")
    raw_df = load_data(data_path)
    df = preprocess_data(raw_df)

    # Input features: drop non-numeric IDs and raw target
    features_to_drop = [c for c in ['machine_id', 'machine_type', 'failure'] if c in df.columns]
    X = df.drop(columns=features_to_drop)
    y = df['failure']

    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Failure class distribution: {dict(y.value_counts())}")

    # Stratified 80/20 train/test split to preserve failure proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optimized Random Forest with moderate class weight to boost failure recall
    rf_model = RandomForestClassifier(
        n_estimators=160,
        max_depth=14,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: 2.2},
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluation with tuned 0.40 probability threshold for early warning
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.40).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 55)
    print("      EVALUATION RESULTS (PREDICTIVE ANALYSIS)")
    print("=" * 55)
    print(f"Model Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Failure Recall (C1):    {rec:.4f} ({rec*100:.2f}%)  <-- Target metric")
    print(f"Failure Precision (C1): {prec:.4f} ({prec*100:.2f}%)")
    print(f"Macro F1 Score:         {f1_m:.4f}")
    print(f"ROC-AUC Score:          {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["No Failure (0)", "Failure (1)"]))

    # Feature Importance
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("Top 8 Influential Features:")
    for idx, r in importances.head(8).iterrows():
        print(f"  - {r['Feature']:<24} {r['Importance']:.4f}")

    # Save artifacts
    models_dir = os.path.join(CURRENT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, 'random_forest_model.pkl')
    cols_file = os.path.join(models_dir, 'model_columns.pkl')

    joblib.dump(rf_model, model_file)
    joblib.dump(list(X.columns), cols_file)
    print(f"\nModel artifacts successfully saved to:")
    print(f"  -> {model_file}")
    print(f"  -> {cols_file}")

if __name__ == "__main__":
    train_and_evaluate()
