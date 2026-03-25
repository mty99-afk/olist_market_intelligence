from pathlib import Path

import duckdb
import joblib
import pandas as pd
import shap


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"
MODEL_PATH = BASE_DIR / "models" / "best_model_random_forest_pre_delivery.joblib"

FEATURE_COLUMNS = [
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
    "customer_state",
    "product_category_name_english",
]


def load_model():
    model = joblib.load(MODEL_PATH)
    classifier = model.named_steps.get("classifier")

    if classifier is not None and hasattr(classifier, "n_jobs"):
        classifier.n_jobs = 1

    return model


def load_scored_data(limit=500):
    conn = duckdb.connect(str(DB_PATH))
    query = f"""
    SELECT
        order_id,
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
    LIMIT {limit}
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    return df


def get_transformed_data(model, X):
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    if not isinstance(X_transformed, pd.DataFrame):
        try:
            X_transformed = X_transformed.toarray()
        except Exception:
            pass

    return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)


def get_positive_class_expected_value(explainer):
    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        return expected_value[1]

    if hasattr(expected_value, "ndim") and expected_value.ndim > 0:
        return expected_value[1]

    return expected_value


def get_positive_class_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]

    if getattr(shap_values, "ndim", 0) == 3:
        return shap_values[:, :, 1]

    return shap_values


def get_shap_for_order(order_id, limit=500):
    df = load_scored_data(limit=limit)
    row = df[df["order_id"] == order_id].copy()

    if row.empty:
        raise ValueError(f"No se encontro el order_id: {order_id}")

    model = load_model()
    X = row[FEATURE_COLUMNS].copy()

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    X_transformed_df = get_transformed_data(model, X)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed_df)
    shap_values_class1 = get_positive_class_shap_values(shap_values)[0]
    base_value = get_positive_class_expected_value(explainer)

    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values_class1,
        }
    )

    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False)

    pred_proba = model.predict_proba(X)[0, 1]
    pred_class = model.predict(X)[0]
    actual_bad_review = int(row["bad_review"].iloc[0])

    order_summary = row[
        [
            "order_id",
            "bad_review",
            "customer_state",
            "product_category_name_english",
            "price",
            "freight_value",
            "seller_avg_review",
            "estimated_delivery_days",
        ]
    ].copy()

    return {
        "order_summary": order_summary,
        "pred_proba": float(pred_proba),
        "pred_class": int(pred_class),
        "actual_bad_review": actual_bad_review,
        "base_value": float(base_value),
        "shap_df": shap_df,
        "X_transformed": X_transformed_df,
        "feature_names": feature_names,
    }
