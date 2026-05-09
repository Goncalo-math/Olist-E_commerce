import pandas as pd

def build_master(dfs):
    orders    = dfs["orders"].copy()
    reviews   = dfs["reviews"].copy()
    items     = dfs["order_items"].copy()
    payments  = dfs["payments"].copy()
    customers = dfs["customers"].copy()
    products  = dfs["products"].copy()
    trans     = dfs["translations"].copy()

    # --- Parse timestamps ---
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])

    # --- Delivery features ---
    orders["days_to_deliver"] = (
        orders["order_delivered_customer_date"] -
        orders["order_purchase_timestamp"]
    ).dt.days

    orders["delivery_delay"] = (
        orders["order_delivered_customer_date"] -
        orders["order_estimated_delivery_date"]
    ).dt.days  # positive = late, negative = early

    # --- Payment aggregates per order ---
    pay_agg = payments.groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        n_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda x: x.mode()[0])
    ).reset_index()

    # --- Items aggregates per order ---
    item_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        total_freight=("freight_value", "sum"),
        avg_price=("price", "mean"),
    ).reset_index()

    # --- Translate product categories ---
    products = products.merge(trans, on="product_category_name", how="left")

    # --- Merge everything ---
    df = (
        orders
        .merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    )
  

    # --- Target: binary flag for negative reviews ---
    # review_score 1 or 2 = negative (1), 3-5 = not negative (0)
    df["is_negative"] = (df["review_score"] <= 2).astype(int)

    # Drop rows without a review (can't train on them)
    df = df.dropna(subset=["review_score"])

    return df