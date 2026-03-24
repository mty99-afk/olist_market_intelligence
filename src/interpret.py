from pathlib import Path
import joblib
import duckdb
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"
MODEL_PATH = BASE_DIR / "models" / "best_model_random_forest_pre_delivery.joblib"

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

def get_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = []

    # numéricas
    feature_names.extend(numeric_features)

    # categóricas (one hot)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = ohe.get_feature_names_out(categorical_features)

    feature_names.extend(cat_features)

    return feature_names

def main():
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

    # cargar modelo
    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    importances = classifier.feature_importances_

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    df_importance = df_importance.sort_values(by="importance", ascending=False)

    print("\nTOP 20 FEATURES MÁS IMPORTANTES:\n")
    print(df_importance.head(20))

    # guardar
    output_path = BASE_DIR / "reports" / "feature_importance.csv"
    df_importance.to_csv(output_path, index=False)

    print(f"\nFeature importance guardado en: {output_path}")

if __name__ == "__main__":
    main()