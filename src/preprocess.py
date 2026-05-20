
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import when, lower, trim, col
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("DDoS Detection") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
print("Spark Connected Successfully")

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("DDoS").getOrCreate()

spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")

path = r"C:\Spark\RealTime_DDoS_Detection\data\cicddos2019"

files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]

dfs = []

for file in files:
    df_temp = spark.read.parquet(file)

    # توحيد الأنواع المهمة
    df_temp = df_temp.withColumn(
        "Fwd Header Length",
        col("Fwd Header Length").cast("long")
    )

    df_temp = df_temp.withColumn(
        "Fwd Act Data Packets",
        col("Fwd Act Data Packets").cast("int")
    )

    dfs.append(df_temp)

# دمج كل الملفات
df = dfs[0]

for d in dfs[1:]:
    df = df.unionByName(d)

print("Dataset Loaded Successfully")
df.printSchema()
print(df.count())
# =========================
# Data Cleaning
# =========================

print("\nOriginal Rows Count:", df.count())

# Remove null values
df = df.na.drop()

# Remove duplicate rows
df = df.dropDuplicates()

print("Cleaned Rows Count:", df.count())

# =========================
# Label Encoding
# BENIGN = 0
# ATTACK = 1
# =========================

df = df.withColumn(
    "Label",
    lower(trim(col("Label")))
)

# تحويل التصنيفات
df = df.withColumn(
    "label",
    when(col("Label") == "benign", 0).otherwise(1)
)

# =========================
# Feature Selection
# =========================

exclude_columns = ["Label", "label"]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
]

# =========================
# Feature Vectorization
# =========================

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="unscaled_features"
)

df = assembler.transform(df)

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features"
)

scaler_model = scaler.fit(df)

df = scaler_model.transform(df)

# =========================
# Split Dataset
# =========================

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("\nTraining Rows:", train_df.count())
print("Testing Rows:", test_df.count())

# =========================
# Random Forest Model
# =========================

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
        numTrees=5,
    maxDepth=5
)
train_df = train_df.limit(15000)
test_df = test_df.limit(5000)
df = df.sample(0.3, seed=42)
# =========================
# Training Time Measurement
# =========================

start_time = time.time()

model = rf.fit(train_df)

end_time = time.time()

execution_time = end_time - start_time

print(f"\nTraining Time: {execution_time:.2f} seconds")
# =========================
# SPEEDUP & EFFICIENCY TEST
# =========================

dataset_sizes = [0.25, 0.50, 0.75, 1.0]

execution_times = []

for size in dataset_sizes:

    sample_df = df.sample(size, seed=42)

    train_sample, test_sample = sample_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    start = time.time()

    temp_model = rf.fit(train_sample)

    end = time.time()

    exec_time = end - start

    execution_times.append(exec_time)

    print(
        f"Dataset Size: {int(size*100)}% "
        f"| Training Time: {exec_time:.2f} sec"
    )

# =========================
# SPEEDUP CALCULATION
# =========================

baseline_time = execution_times[0]

speedups = [
    baseline_time / t
    for t in execution_times
]

# =========================
# EFFICIENCY CALCULATION
# =========================

num_cores = 4

efficiencies = [
    s / num_cores
    for s in speedups
]

print("\n========== SPEEDUP RESULTS ==========")

for i in range(len(dataset_sizes)):

    print(
        f"Size={int(dataset_sizes[i]*100)}% | "
        f"Time={execution_times[i]:.2f}s | "
        f"Speedup={speedups[i]:.2f} | "
        f"Efficiency={efficiencies[i]:.2f}"
    )
# =========================
# Predictions
# =========================

predictions = model.transform(test_df)

# =========================
# Evaluation
# =========================

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
recall = recall_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

# =========================
# Results
# =========================

print("\n========== MODEL RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1_score:.4f}")

print("===================================")

# =========================
# Show Sample Predictions
# =========================

predictions.select(
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)

# =========================
# Stop Spark Session
# =========================

# =========================
# Feature Importance
# =========================

importances = model.featureImportances.toArray()

feature_importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance_df.head(15))

# =========================
# Feature Importance Plot
# =========================

top_features = feature_importance_df.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")

plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.savefig("results/feature_importance.png")

plt.close()
# تحويل النتائج إلى RDD
predictionAndLabels = predictions.select("prediction", "label") \
    .rdd.map(lambda x: (float(x[0]), float(x[1])))

# إنشاء Confusion Matrix
metrics = MulticlassMetrics(predictionAndLabels)

# عرض المصفوفة
confusion_matrix = metrics.confusionMatrix().toArray()

print("\nConfusion Matrix:")
print(confusion_matrix)

df.select("Label").distinct().show(50, truncate=False)
df.groupBy("label").count().show()
#model.write().overwrite().save("saved_model/ddos_rf_model")
print("Model Saved Successfully")
predictions = predictions.withColumn(
    "probability_str",
    col("probability").cast("string")
)

predictions.select(
    "label",
  "prediction",
    "probability_str"
).write.mode("overwrite").csv(
    "results/predictions.csv",
   header=True
)
print("Predictions Saved Successfully")


label_counts = df.groupBy("label").count()#.toPandas()

plt.figure(figsize=(6,4))
label_counts = label_counts.toPandas()

plt.bar(label_counts["label"].astype(str),
        label_counts["count"])
plt.title("Normal vs Attack Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()
metrics_names = ["Accuracy", "Precision", "Recall", "f1_score"]
metrics_values = [accuracy, precision, recall, f1_score]

plt.figure(figsize=(7,4))
plt.bar(metrics_names, metrics_values)

plt.ylim(0.9, 1.0)

plt.title("Model Performance Metrics")

plt.show()


#evaluatorROC = BinaryClassificationEvaluator(
  #  labelCol="label",
  #  rawPredictionCol="rawPrediction",
   # metricName="areaUnderROC"
#)

#auc = evaluatorROC.evaluate(predictions)

#print(f"AUC-ROC Score: {auc:.4f}")

# =========================
# ROC Curve REAL
# =========================

from sklearn.metrics import roc_curve, auc

sample_pred = predictions.select(
    "label",
    "probability"
).limit(5000).toPandas()

# القيم الحقيقية
y_true = sample_pred["label"]

# احتمالية الهجوم
y_scores = sample_pred["probability"].apply(
    lambda x: float(x[1])
)

# حساب ROC
fpr, tpr, _ = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

# الرسم
plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.savefig("results/roc_curve_real.png")

plt.close()

print("ROC Curve Saved Successfully")
pred_labels = predictions.select(
   "label",
   "prediction"
).toPandas()

print(
    classification_report(
        pred_labels["label"],
        pred_labels["prediction"]
    )
)


spark.stop()
