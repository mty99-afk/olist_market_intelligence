from shap_utils import load_scored_data, get_shap_for_order

df = load_scored_data(limit=50)
sample_order_id = df["order_id"].iloc[0]

result = get_shap_for_order(sample_order_id, limit=50)

print("ORDER SUMMARY:")
print(result["order_summary"])
print("\nPRED_PROBA:", result["pred_proba"])
print("\nTOP SHAP:")
print(result["shap_df"].head(10))