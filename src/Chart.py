
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
# =========================
# REAL ROC CURVE
# =========================

# =========================
# IMPORTS
# =========================

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import when, lower, trim, col

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import pandas as pd
import os

# =========================
# SPARK SESSION
# =========================

spark = SparkSession.builder \
    .appName("DDoS ROC Curve") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print("Spark Started Successfully")

# =========================
# LOAD DATASET
# =========================

path = r"C:\Spark\RealTime_DDoS_Detection\data\cicddos2019"

files = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.endswith(".parquet")
]

dfs = []

for file in files:

    df_temp = spark.read.parquet(file)

    dfs.append(df_temp)

# دمج الملفات

df = dfs[0]

for d in dfs[1:]:

    df = df.unionByName(d)

print("Dataset Loaded Successfully")

# =========================
# CLEANING
# =========================

df = df.na.drop()

df = df.dropDuplicates()

# =========================
# LABEL ENCODING
# =========================

df = df.withColumn(
    "Label",
    lower(trim(col("Label")))
)

df = df.withColumn(
    "label",
    when(col("Label") == "benign", 0).otherwise(1)
)

# =========================
# FEATURE SELECTION
# =========================

exclude_columns = ["Label", "label"]

feature_columns = [

    c for c in df.columns

    if c not in exclude_columns
]

# =========================
# VECTOR ASSEMBLER
# =========================

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="unscaled_features"
)

df = assembler.transform(df)

# =========================
# STANDARD SCALER
# =========================

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features"
)

scaler_model = scaler.fit(df)

df = scaler_model.transform(df)

# =========================
# SPLIT DATA
# =========================

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Train/Test Split Completed")

# =========================
# RANDOM FOREST MODEL
# =========================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=20
)

# =========================
# TRAIN MODEL
# =========================

model = rf.fit(train_df)

print("Model Trained Successfully")

# =========================
# PREDICTIONS
# =========================

predictions = model.transform(test_df)

print("Predictions Generated Successfully")

# =========================
# ROC CURVE
# =========================

# =========================
# IMPORTS
# =========================

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import when, lower, trim, col

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import pandas as pd
import os

# =========================
# SPARK SESSION
# =========================

spark = SparkSession.builder \
    .appName("DDoS ROC Curve") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print("Spark Started Successfully")

# =========================
# LOAD DATASET
# =========================

path = r"C:\Spark\RealTime_DDoS_Detection\data\cicddos2019"

files = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.endswith(".parquet")
]

dfs = []

for file in files:

    df_temp = spark.read.parquet(file)

    dfs.append(df_temp)

# دمج الملفات

df = dfs[0]

for d in dfs[1:]:

    df = df.unionByName(d)

print("Dataset Loaded Successfully")

# =========================
# CLEANING
# =========================

df = df.na.drop()

df = df.dropDuplicates()

# =========================
# LABEL ENCODING
# =========================

df = df.withColumn(
    "Label",
    lower(trim(col("Label")))
)

df = df.withColumn(
    "label",
    when(col("Label") == "benign", 0).otherwise(1)
)

# =========================
# FEATURE SELECTION
# =========================

exclude_columns = ["Label", "label"]

feature_columns = [

    c for c in df.columns

    if c not in exclude_columns
]

# =========================
# VECTOR ASSEMBLER
# =========================

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="unscaled_features"
)

df = assembler.transform(df)

# =========================
# STANDARD SCALER
# =========================

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features"
)

scaler_model = scaler.fit(df)

df = scaler_model.transform(df)

# =========================
# SPLIT DATA
# =========================

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Train/Test Split Completed")

# =========================
# RANDOM FOREST MODEL
# =========================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=20
)

# =========================
# TRAIN MODEL
# =========================

model = rf.fit(train_df)

print("Model Trained Successfully")

# =========================
# PREDICTIONS
# =========================

predictions = model.transform(test_df)

print("Predictions Generated Successfully")

# =========================
# ROC CURVE
# =========================

pred_pd = predictions.select(
    "label",
    "probability"
).toPandas()

# القيم الحقيقية

y_true = pred_pd["label"]

# احتمالية الهجوم

y_scores = pred_pd["probability"].apply(
    lambda x: float(x[1])
)

# حساب ROC

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_scores
)

# حساب AUC

roc_auc = auc(fpr, tpr)

print(f"AUC Score = {roc_auc:.4f}")

# =========================
# PLOT ROC CURVE
# =========================

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

# إنشاء مجلد النتائج إذا غير موجود

os.makedirs("results", exist_ok=True)

# حفظ الصورة

plt.savefig("results/roc_curve_real.png")

plt.show()

print("ROC Curve Saved Successfully")


# =========================
# EXECUTION TIME PLOT
# =========================
dataset_sizes = [25, 50, 75, 100]

execution_times = [30.08, 30.44, 28.11, 29.79]

speedups = [1.00, 0.99, 1.07, 1.01]

efficiencies = [0.25, 0.25, 0.27, 0.25]

sizes_percent = [
    int(s * 100)
    for s in dataset_sizes
]

plt.figure(figsize=(7,5))

plt.plot(
    sizes_percent,
    execution_times,
    marker='o'
)

plt.xlabel("Dataset Size (%)")

plt.ylabel("Training Time (sec)")

plt.title("Execution Time vs Dataset Size")

plt.savefig("results/execution_time.png")

plt.close()


# =========================
# SPEEDUP PLOT
# =========================

plt.figure(figsize=(7,5))

plt.plot(
    sizes_percent,
    speedups,
    marker='o'
)

plt.xlabel("Dataset Size (%)")

plt.ylabel("Speedup")

plt.title("Speedup Analysis")

plt.savefig("results/speedup.png")

plt.close()



# =========================
# EFFICIENCY PLOT
# =========================

plt.figure(figsize=(7,5))

plt.plot(
    sizes_percent,
    efficiencies,
    marker='o'
)

plt.xlabel("Dataset Size (%)")

plt.ylabel("Efficiency")

plt.title("Efficiency Analysis")

plt.savefig("results/efficiency.png")

plt.close()

print("Speedup Analysis Completed Successfully")



print("All Advanced Charts Saved Successfully")