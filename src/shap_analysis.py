from pathlib import Path

import joblib
import duckdb
import matplotlib
import pandas as pd
import shap

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"
MODEL_PATH = BASE_DIR / "models" / "best_model_random_forest_pre_delivery.joblib"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_data(limit=5000):
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


def get_transformed_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = list(numeric_features)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(categorical_features).tolist())
    return feature_names


def get_positive_class_expected_value(explainer):
    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        return expected_value[1]

    if hasattr(expected_value, "ndim") and expected_value.ndim > 0:
        return expected_value[1]

    return expected_value


def main():
    model = joblib.load(MODEL_PATH)
    df = load_data(limit=5000)

    order_ids = df["order_id"].copy()
    X = df.drop(columns=["order_id", "bad_review"])

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

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)
    feature_names = get_transformed_feature_names(
        preprocessor, numeric_features, categorical_features
    )

    X_shap = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=feature_names,
    )

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_shap)
    expected_value_positive = get_positive_class_expected_value(explainer)

    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]
    else:
        shap_values_positive = shap_values
        if getattr(shap_values_positive, "ndim", 0) == 3:
            shap_values_positive = shap_values_positive[:, :, 1]

    summary_plot_path = REPORTS_DIR / "shap_summary.png"

    plt.figure()
    shap.summary_plot(shap_values_positive, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    case_index = 0
    case_order_id = order_ids.iloc[case_index]
    force_plot_path = REPORTS_DIR / f"shap_force_plot_order_{case_order_id}.png"

    shap.force_plot(
        expected_value_positive,
        shap_values_positive[case_index],
        X_shap.iloc[case_index],
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(force_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap_importance = pd.DataFrame(
        {
            "feature": X_shap.columns,
            "importance": abs(shap_values_positive).mean(axis=0),
        }
    ).sort_values(by="importance", ascending=False)

    importance_path = REPORTS_DIR / "shap_feature_importance.csv"
    shap_importance.to_csv(importance_path, index=False)

    print("\nTop features segun SHAP:\n")
    print(shap_importance.head(20))
    print(f"\nPedido analizado en force plot: {case_order_id}")
    print(f"\nGrafico guardado en: {summary_plot_path}")
    print(f"Force plot guardado en: {force_plot_path}")
    print(f"Importancias guardadas en: {importance_path}")


if __name__ == "__main__":
    main()
