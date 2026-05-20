from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Spark Session
# =========================

spark = SparkSession.builder \
    .appName("Compare Models") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark Started")

# =========================
# Load Dataset
# =========================

df = spark.read.parquet("../dataset/Syn-training.parquet")

print("Dataset Loaded")

# =========================
# Clean Data
# =========================

df = df.dropna()

# Binary Label
df = df.withColumn(
    "label",
    when(col("Label") == "Syn", 1).otherwise(0)
)

# =========================
# Feature Selection
# =========================

feature_columns = [
    c for c in df.columns
    if c not in ["Label", "label"]
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

data = assembler.transform(df).select("features", "label")

# =========================
# Split Dataset
# =========================

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print("Train:", train_data.count())
print("Test :", test_data.count())

# =========================
# Models
# =========================

models = {
    "Logistic Regression":
        LogisticRegression(featuresCol="features", labelCol="label"),

    "Decision Tree":
        DecisionTreeClassifier(featuresCol="features", labelCol="label"),

    "Random Forest":
        RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=20
        ),

    "GBT":
        GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=20
        )
}

# =========================
# Evaluation
# =========================

results = []

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

for name, model in models.items():

    print(f"\nTraining {name}...")

    trained_model = model.fit(train_data)

    predictions = trained_model.transform(test_data)

    accuracy = evaluator_acc.evaluate(predictions)

    f1 = evaluator_f1.evaluate(predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1 Score": f1
    })

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")

# =========================
# Results DataFrame
# =========================

results_df = pd.DataFrame(results)

print("\n========== FINAL RESULTS ==========")
print(results_df)

# =========================
# Visualization
# =========================

results_df.plot(
    x="Model",
    y=["Accuracy", "F1 Score"],
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0.9, 1.0)

plt.tight_layout()

plt.savefig("../results/model_comparison.png")

print("Chart Saved")

spark.stop()