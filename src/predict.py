from pathlib import Path
import joblib
import duckdb

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"
MODEL_PATH = BASE_DIR / "models" / "best_model_random_forest_pre_delivery.joblib"

def load_sample_data(limit=10):
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

def predict_sample():
    model = joblib.load(MODEL_PATH)
    df = load_sample_data(limit=10)

    X = df.drop(columns=["order_id", "bad_review"])
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    output = df[["order_id", "bad_review"]].copy()
    output["pred_bad_review"] = predictions
    output["pred_proba_bad_review"] = probabilities

    print(output)

if __name__ == "__main__":
    predict_sample()