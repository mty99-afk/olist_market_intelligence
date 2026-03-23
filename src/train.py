from pathlib import Path
import json
import duckdb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

def load_data():
    conn = duckdb.connect(str(DB_PATH))

    query = """
    SELECT
        price,
        freight_value,
        product_name_lenght,
        product_description_lenght,
        product_photos_qty,
        product_weight_g,
        product_length_cm,
        product_height_cm,
        product_width_cm,
        estimated_delivery_days,
        purchase_month,
        purchase_dayofweek,
        purchase_hour,
        freight_ratio,
        product_volume_cm3,
        seller_total_items,
        seller_avg_review,
        seller_avg_price,
        category_total_items,
        category_avg_price,
        category_avg_review,
        price_vs_category_avg_ratio,
        customer_state,
        product_category_name_english,
        bad_review
    FROM ml_order_reviews
    WHERE bad_review IS NOT NULL
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*60}")
    print(f"Modelo: {model_name}")
    print(f"{'='*60}")
    print("ROC-AUC:", round(roc_auc, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    metrics = {
        "model_name": model_name,
        "roc_auc": float(roc_auc),
        "precision_bad_review": float(precision_score(y_test, y_pred, pos_label=1)),
        "recall_bad_review": float(recall_score(y_test, y_pred, pos_label=1)),
        "f1_bad_review": float(f1_score(y_test, y_pred, pos_label=1)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return metrics, model

def train_and_compare():
    df = load_data()

    X = df.drop(columns=["bad_review"])
    y = df["bad_review"]

    numeric_features = [
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "estimated_delivery_days",
        "purchase_month",
        "purchase_dayofweek",
        "purchase_hour",
        "freight_ratio",
        "product_volume_cm3",
        "seller_total_items",
        "seller_avg_review",
        "seller_avg_price",
        "category_total_items",
        "category_avg_price",
        "category_avg_review",
        "price_vs_category_avg_ratio",
    ]

    categorical_features = [
        "customer_state",
        "product_category_name_english",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor_lr = build_preprocessor(numeric_features, categorical_features)
    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor_lr),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    preprocessor_rf = build_preprocessor(numeric_features, categorical_features)
    random_forest_model = Pipeline(steps=[
        ("preprocessor", preprocessor_rf),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    lr_metrics, fitted_lr = evaluate_model(
        "logistic_regression_pre_delivery", logistic_model, X_train, X_test, y_train, y_test
    )

    rf_metrics, fitted_rf = evaluate_model(
        "random_forest_pre_delivery", random_forest_model, X_train, X_test, y_train, y_test
    )

    all_metrics = {
        "logistic_regression_pre_delivery": lr_metrics,
        "random_forest_pre_delivery": rf_metrics
    }

    metrics_path = REPORTS_DIR / "model_metrics_pre_delivery.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        best_model_name = "random_forest_pre_delivery"
        best_model = fitted_rf
        best_metrics = rf_metrics
    else:
        best_model_name = "logistic_regression_pre_delivery"
        best_model = fitted_lr
        best_metrics = lr_metrics

    output_path = MODEL_DIR / f"best_model_{best_model_name}.joblib"
    joblib.dump(best_model, output_path)

    print(f"\nMejor modelo: {best_model_name}")
    print(f"Mejor ROC-AUC: {round(best_metrics['roc_auc'], 4)}")
    print(f"Modelo guardado en: {output_path}")
    print(f"Métricas guardadas en: {metrics_path}")

if __name__ == "__main__":
    train_and_compare()