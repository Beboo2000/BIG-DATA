
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from pyspark.sql.functions import when, lower, trim, col

# ====================================================
# 1) CLASS DISTRIBUTION
# ====================================================

classes = ['Attack (1)', 'Normal (0)']
counts = [331372, 94704]

plt.figure(figsize=(8,5))
plt.bar(classes, counts)

plt.title("Normal vs Attack Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.savefig("results/class_distribution.png")
plt.close()

# ====================================================
# 2) PERFORMANCE METRICS
# ====================================================

metrics = {
    'Accuracy': 0.9932,
    'Precision': 0.9933,
    'Recall': 0.9932,
    'F1-Score': 0.9933
}

plt.figure(figsize=(8,5))
plt.bar(metrics.keys(), metrics.values())

plt.ylim(0.98, 1.0)

plt.title("Model Performance Metrics")
plt.ylabel("Score")

plt.savefig("results/performance_metrics.png")
plt.close()

# ====================================================
# 3) CONFUSION MATRIX
# ====================================================

cm = np.array([
    [18498, 267],
    [306, 65790]
])

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='g',
    cmap='Blues',
    xticklabels=['Normal', 'Attack'],
    yticklabels=['Normal', 'Attack']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results/confusion_matrix.png")
plt.close()

# ====================================================
# 4) PIE CHART
# ====================================================

plt.figure(figsize=(6,6))

plt.pie(
    counts,
    labels=classes,
    autopct='%1.1f%%'
)

plt.title("Dataset Distribution")

plt.savefig("results/pie_distribution.png")
plt.close()

# ====================================================
# 5) ROC CURVE
# ====================================================

pred_pd = predictions.select(
    "label",
    "probability"
).toPandas()

y_true = pred_pd["label"]

y_scores = pred_pd["probability"].apply(
    lambda x: float(x[1])
)

fpr, tpr, _ = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.savefig("results/roc_curve.png")

plt.close()

print(f"AUC-ROC Score: {roc_auc:.4f}")

print("All Advanced Charts Saved Successfully")