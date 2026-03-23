from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"

conn = duckdb.connect(str(DB_PATH))

query = """
SELECT
    COUNT(*) AS total_rows,
    ROUND(AVG(estimated_delivery_days), 2) AS avg_estimated_delivery_days,
    ROUND(AVG(freight_ratio), 4) AS avg_freight_ratio,
    ROUND(AVG(product_volume_cm3), 2) AS avg_product_volume_cm3,
    ROUND(AVG(seller_avg_review), 4) AS avg_seller_avg_review
FROM ml_order_reviews
WHERE bad_review IS NOT NULL
"""

result = conn.execute(query).fetchdf()
print(result)

conn.close()