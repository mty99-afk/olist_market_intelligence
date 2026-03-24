from pathlib import Path
import json
import duckdb
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "mart" / "olist.duckdb"
MODEL_PATH = BASE_DIR / "models" / "best_model_random_forest_pre_delivery.joblib"
METRICS_PATH = BASE_DIR / "reports" / "model_metrics_pre_delivery.json"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "reports" / "feature_importance.csv"

st.set_page_config(page_title="Olist Risk Dashboard", layout="wide")

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_feature_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)

@st.cache_data
def load_prediction_data(limit=500):
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

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def score_data(df):
    model = load_model()
    X = df.drop(columns=["order_id", "bad_review"])
    df_result = df[["order_id", "bad_review", "customer_state", "product_category_name_english"]].copy()
    df_result["pred_bad_review"] = model.predict(X)
    df_result["pred_proba_bad_review"] = model.predict_proba(X)[:, 1]
    return df_result

st.title("🛒 Olist Market Intelligence")
st.subheader("Predicción de malas reviews en escenario pre-entrega")

metrics = load_metrics()
fi = load_feature_importance()
raw_df = load_prediction_data()
pred_df = score_data(raw_df)

rf_metrics = metrics["random_forest_pre_delivery"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", f"{rf_metrics['roc_auc']:.4f}")
col2.metric("Precision bad_review", f"{rf_metrics['precision_bad_review']:.4f}")
col3.metric("Recall bad_review", f"{rf_metrics['recall_bad_review']:.4f}")
col4.metric("F1 bad_review", f"{rf_metrics['f1_bad_review']:.4f}")

st.markdown("---")

st.subheader("Top 10 variables más importantes")
st.bar_chart(fi.head(10).set_index("feature"))

st.markdown("---")

st.subheader("Predicciones de riesgo")

state_options = ["Todos"] + sorted(pred_df["customer_state"].dropna().unique().tolist())
selected_state = st.selectbox("Filtrar por estado", state_options)

if selected_state != "Todos":
    pred_df = pred_df[pred_df["customer_state"] == selected_state]

threshold = st.slider("Threshold de riesgo", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

filtered_df = pred_df.copy()
filtered_df["pred_bad_review_threshold"] = (filtered_df["pred_proba_bad_review"] >= threshold).astype(int)

only_high_risk = st.checkbox("Mostrar solo alto riesgo", value=False)
if only_high_risk:
    filtered_df = filtered_df[filtered_df["pred_bad_review_threshold"] == 1]

st.dataframe(
    filtered_df.sort_values("pred_proba_bad_review", ascending=False),
    use_container_width=True
)

st.markdown("---")

st.subheader("Hallazgos de negocio")
st.markdown("""
- La reputación histórica del seller es el driver más fuerte.
- El costo de envío y el tiempo estimado de entrega capturan riesgo importante.
- El precio relativo dentro de la categoría ayuda a explicar expectativa vs satisfacción.
- El modelo pre-entrega sacrifica performance respecto al post-entrega, pero gana realismo operacional.
""")