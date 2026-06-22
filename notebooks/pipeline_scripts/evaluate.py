
import pandas as pd
import numpy as np
import json
import os
import tarfile
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

print("Step 3: Evaluating model...")

# Load model
model_path = "/opt/ml/processing/model/model.tar.gz"
with tarfile.open(model_path) as tar:
    tar.extractall("/opt/ml/processing/model/")

booster = xgb.Booster()
booster.load_model("/opt/ml/processing/model/xgboost-model")
print("Model loaded")

# Load test data
df_test = pd.read_csv("/opt/ml/processing/input/test/test.csv", header=None)
y_test  = df_test.iloc[:, 0].values
X_test  = df_test.iloc[:, 1:].values

# Predict
dtest       = xgb.DMatrix(X_test)
y_pred_prob = booster.predict(dtest)
y_pred      = (y_pred_prob >= 0.5).astype(int)

# Compute metrics
acc = float(accuracy_score(y_test, y_pred))
f1  = float(f1_score(y_test, y_pred, average="macro"))
auc = float(roc_auc_score(y_test, y_pred_prob))

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"AUC-ROC:  {auc:.4f}")

# Write evaluation report — SageMaker Pipelines reads this
report = {
    "binary_classification_metrics": {
        "accuracy": {"value": acc},
        "f1_macro": {"value": f1},
        "auc_roc":  {"value": auc}
    }
}

os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump(report, f, indent=2)

print("Evaluation report written")
