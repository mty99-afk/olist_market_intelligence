from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"

def create_ml_features():
    conn = duckdb.connect(str(DB_PATH))

    query = """
    CREATE OR REPLACE TABLE ml_order_reviews AS
    WITH seller_stats AS (
        SELECT
            seller_id,
            COUNT(*) AS seller_total_items,
            AVG(review_score) AS seller_avg_review,
            AVG(price) AS seller_avg_price
        FROM mart_order_items
        WHERE review_score IS NOT NULL
        GROUP BY seller_id
    ),
    category_stats AS (
        SELECT
            COALESCE(product_category_name_english, product_category_name) AS category_name,
            COUNT(*) AS category_total_items,
            AVG(price) AS category_avg_price,
            AVG(review_score) AS category_avg_review
        FROM mart_order_items
        WHERE review_score IS NOT NULL
        GROUP BY 1
    )
    SELECT
        moi.order_id,
        moi.order_item_id,
        moi.product_id,
        moi.seller_id,
        moi.customer_id,
        moi.customer_unique_id,
        moi.customer_city,
        moi.customer_state,
        moi.order_status,
        moi.order_purchase_timestamp,
        moi.order_approved_at,
        moi.order_estimated_delivery_date,

        moi.product_category_name,
        COALESCE(moi.product_category_name_english, moi.product_category_name) AS product_category_name_english,

        moi.price,
        moi.freight_value,
        moi.product_name_lenght,
        moi.product_description_lenght,
        moi.product_photos_qty,
        moi.product_weight_g,
        moi.product_length_cm,
        moi.product_height_cm,
        moi.product_width_cm,
        moi.review_score,

        CASE
            WHEN moi.review_score <= 2 THEN 1
            WHEN moi.review_score >= 4 THEN 0
            ELSE NULL
        END AS bad_review,

        EXTRACT('month' FROM moi.order_purchase_timestamp) AS purchase_month,
        EXTRACT('dayofweek' FROM moi.order_purchase_timestamp) AS purchase_dayofweek,
        EXTRACT('hour' FROM moi.order_purchase_timestamp) AS purchase_hour,

        CASE
            WHEN moi.price > 0 THEN moi.freight_value / moi.price
            ELSE NULL
        END AS freight_ratio,

        (moi.product_length_cm * moi.product_height_cm * moi.product_width_cm) AS product_volume_cm3,

        datediff('day', moi.order_purchase_timestamp, moi.order_estimated_delivery_date) AS estimated_delivery_days,

        ss.seller_total_items,
        ss.seller_avg_review,
        ss.seller_avg_price,

        cs.category_total_items,
        cs.category_avg_price,
        cs.category_avg_review,

        CASE
            WHEN cs.category_avg_price > 0 THEN moi.price / cs.category_avg_price
            ELSE NULL
        END AS price_vs_category_avg_ratio

    FROM mart_order_items moi
    LEFT JOIN seller_stats ss
        ON moi.seller_id = ss.seller_id
    LEFT JOIN category_stats cs
        ON COALESCE(moi.product_category_name_english, moi.product_category_name) = cs.category_name
    WHERE moi.review_score IS NOT NULL
    """

    conn.execute(query)
    conn.close()

    print("Tabla ml_order_reviews creada correctamente para escenario pre-entrega.")

if __name__ == "__main__":
    create_ml_features()