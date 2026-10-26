import os
import warnings
import joblib
import matplotlib

import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/churn.csv"

MODEL_PATH = "models/churn_model.pkl"

PREPROCESSOR_PATH = "models/preprocessor.pkl"

OUTPUT_DIR = "outputs"


os.makedirs("models", exist_ok=True)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\nFirst 5 rows:")

print(
    df.head()
)

print("\nColumns:")

print(
    df.columns.tolist()
)

print("\nData types:")

print(
    df.dtypes
)


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print(
    f"\nDuplicate rows: {duplicate_count}"
)

df = df.drop_duplicates()


# ============================================================
# 4. FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "Churn",
    "churn",
    "Exited",
    "exited",
    "Attrition",
    "attrition"
]

target_column = None

for column in target_candidates:

    if column in df.columns:

        target_column = column

        break


if target_column is None:

    raise ValueError(
        "Target column not found. "
        "Please name the target column 'Churn'."
    )


print(
    f"\nTarget column: {target_column}"
)


# ============================================================
# 5. CLEAN TARGET
# ============================================================

if df[target_column].dtype == "object":

    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_mapping = {

        "yes": 1,

        "no": 0,

        "true": 1,

        "false": 0,

        "1": 1,

        "0": 0,

        "churn": 1,

        "stay": 0,

        "left": 1,

        "exited": 1,

        "active": 0
    }

    df[target_column] = (
        df[target_column]
        .map(target_mapping)
    )


df[target_column] = pd.to_numeric(
    df[target_column],
    errors="coerce"
)


df = df.dropna(
    subset=[target_column]
)


df[target_column] = (
    df[target_column]
    .astype(int)
)


# ============================================================
# 6. CLEAN COMMON NUMERIC COLUMNS
# ============================================================

if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


# ============================================================
# 7. REMOVE ID COLUMNS
# ============================================================

id_columns = [
    "Unnamed: 0",
    "customerID",
    "CustomerID",
    "customer_id"
]

for column in id_columns:

    if column in df.columns:

        print(
            f"Removing ID column: {column}"
        )

        df = df.drop(
            columns=[column]
        )


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

print(
    "\nCreating engineered features..."
)


# Tenure group

if "tenure" in df.columns:

    df["tenure_group"] = pd.cut(

        df["tenure"],

        bins=[
            -1,
            12,
            24,
            48,
            72,
            np.inf
        ],

        labels=[
            "0-1 Year",
            "1-2 Years",
            "2-4 Years",
            "4-6 Years",
            "6+ Years"
        ]
    )


# Estimated lifetime value

if (
    "MonthlyCharges" in df.columns
    and
    "tenure" in df.columns
):

    df[
        "estimated_lifetime_value"
    ] = (
        df["MonthlyCharges"]
        *
        df["tenure"]
    )


# ============================================================
# 9. EDA
# ============================================================

print(
    "\nCreating EDA visualization..."
)


plt.figure(
    figsize=(8, 6)
)

sns.countplot(
    data=df,
    x=target_column
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel(
    "Churn"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "eda.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. SPLIT FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# ============================================================
# 11. REMOVE HIGH CARDINALITY FEATURES
# ============================================================

high_cardinality_columns = []

for column in X.columns:

    if (
        X[column].dtype == "object"
        and
        X[column].nunique() > 100
    ):

        high_cardinality_columns.append(
            column
        )


if high_cardinality_columns:

    print(
        "\nRemoving high-cardinality columns:"
    )

    print(
        high_cardinality_columns
    )

    X = X.drop(
        columns=high_cardinality_columns
    )


# ============================================================
# 12. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = (
    X
    .select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32"
        ]
    )
    .columns
    .tolist()
)


categorical_features = (
    X
    .select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    )
    .columns
    .tolist()
)


print(
    "\nNumeric features:"
)

print(
    numeric_features
)


print(
    "\nCategorical features:"
)

print(
    categorical_features
)


# ============================================================
# 13. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"\nTraining rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)


# ============================================================
# 14. NUMERIC PIPELINE
# ============================================================

numeric_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )
    ]
)


# ============================================================
# 15. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",

            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 16. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            numeric_pipeline,

            numeric_features
        ),

        (
            "categorical",

            categorical_pipeline,

            categorical_features
        )
    ]
)


# ============================================================
# 17. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 18. FULL PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "model",

            model
        )
    ]
)


# ============================================================
# 19. TRAIN
# ============================================================

print(
    "\nTraining Random Forest..."
)


pipeline.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# 20. PREDICTIONS
# ============================================================

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 21. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print(
    "\n" + "=" * 60
)

print(
    "MODEL PERFORMANCE"
)

print(
    "=" * 60
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "No Churn",
        "Churn"
    ],

    yticklabels=[
        "No Churn",
        "Churn"
    ]
)

plt.title(
    "Customer Churn Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

try:

    trained_model = (
        pipeline
        .named_steps["model"]
    )

    trained_preprocessor = (
        pipeline
        .named_steps["preprocessor"]
    )

    feature_names = (
        trained_preprocessor
        .get_feature_names_out()
    )

    importances = (
        trained_model
        .feature_importances_
    )

    feature_importance = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importances
    })


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(20)
    )


    plt.figure(
        figsize=(10, 8)
    )

    sns.barplot(

        data=feature_importance,

        x="Importance",

        y="Feature"
    )

    plt.title(
        "Top 20 Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        dpi=300
    )

    plt.close()


except Exception as e:

    print(
        "Feature importance error:",
        e
    )


# ============================================================
# 24. SAVE COMPLETE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)


print(
    f"\nModel saved to: {MODEL_PATH}"
)


# ============================================================
# 25. SAVE PREPROCESSOR SEPARATELY
# ============================================================

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


print(
    f"Preprocessor saved to: "
    f"{PREPROCESSOR_PATH}"
)


# ============================================================
# 26. SAVE PREDICTIONS
# ============================================================

results = X_test.copy()

results["Actual_Churn"] = (
    y_test.values
)

results["Predicted_Churn"] = (
    y_pred
)

results["Churn_Probability"] = (
    y_probability
)


results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),
    index=False
)


# ============================================================
# 27. SAVE MODEL METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})


metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_metrics.csv"
    ),
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)