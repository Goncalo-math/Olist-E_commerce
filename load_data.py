
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")

def load_all():
    customers    = pd.read_csv(DATA_PATH / "olist_customers_dataset.csv")
    geolocation  = pd.read_csv(DATA_PATH / "olist_geolocation_dataset.csv")
    order_items  = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")
    payments     = pd.read_csv(DATA_PATH / "olist_order_payments_dataset.csv")
    reviews      = pd.read_csv(DATA_PATH / "olist_order_reviews_dataset.csv")
    orders       = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")
    products     = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")
    sellers      = pd.read_csv(DATA_PATH / "olist_sellers_dataset.csv")
    translations = pd.read_csv(DATA_PATH / "product_category_name_translation.csv")

    return {
        "customers":    customers,
        "geolocation":  geolocation,
        "order_items":  order_items,
        "payments":     payments,
        "reviews":      reviews,
        "orders":       orders,
        "products":     products,
        "sellers":      sellers,
        "translations": translations,
    }

if __name__ == "__main__":
    dfs = load_all()
    for name, df in dfs.items():
        print(f"{name}: {df.shape[0]} rows × {df.shape[1]} cols")