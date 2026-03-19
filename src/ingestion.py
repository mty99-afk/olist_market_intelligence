from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"

files = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

def load_csvs_to_duckdb():
    conn = duckdb.connect(str(DB_PATH))

    for table_name, file_name in files.items():
        file_path = RAW_DIR / file_name
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{file_path.as_posix()}');
        """)
        print(f"Tabla cargada: {table_name}")

    conn.close()
    print(f"\nBase creada en: {DB_PATH}")

if __name__ == "__main__":
    load_csvs_to_duckdb()