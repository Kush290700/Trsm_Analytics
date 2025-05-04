import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Desired dtypes for each table
CSV_DTYPES = {
    "orders": {
        "OrderId": "category",
        "CustomerId": "category",
        "SalesRepId": "category",
        "ShippingMethodRequested": "category",
        "OrderStatus": "category",
    },
    "order_lines": {
        "OrderLineId": "category",
        "OrderId": "category",
        "ProductId": "category",
        "ShipperId": "category",
        "QuantityShipped": "float32",
        "Price": "float32",
        "CostPrice": "float32",
    },
    "products": {
        "ProductId": "category",
        "SupplierId": "category",
        "UnitOfBillingId": "category",
        "ProductName": "category",
        "ProductListPrice": "float32",
        "CostPrice": "float32",
        "IsProduction": "category",
        "SKU": "category",
    },
    "customers": {
        "CustomerId": "category",
        "RegionId": "category",
        "CustomerName": "category",
    },
    "regions": {
        "RegionId": "category",
        "RegionName": "category",
    },
    "shippers": {
        "ShipperId": "category",
        "Carrier": "category",
    },
    "suppliers": {
        "SupplierId": "category",
        "SupplierName": "category",
    },
    "shipping_methods": {
        "ShippingMethodId": "category",
        "ShippingMethodName": "category",
    },
    "packs": {
        "PickedForOrderLine": "category",
        "WeightLb": "float32",
        "ItemCount": "float32",
    },
}

# Columns to parse as dates if they exist
DATE_COLS = {
    "orders": ["CreatedAt", "DateOrdered", "DateExpected", "DateShipped"],
    "order_lines": ["CreatedAt", "DateShipped"],
    "packs": ["ShippedAt"],
}


def load_csv_tables(csv_dir="data") -> dict[str, pd.DataFrame]:
    table_names = [
        "orders", "order_lines", "products", "customers",
        "regions", "shippers", "suppliers", "shipping_methods", "packs"
    ]
    raw: dict[str, pd.DataFrame] = {}

    for name in table_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.exists(path):
            logger.warning(f"⚠️ Missing table: {name}.csv")
            raw[name] = pd.DataFrame()
            continue

        df = pd.read_csv(path, low_memory=False)

        # Cast dtypes
        for col, dtype in CSV_DTYPES.get(name, {}).items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    logger.warning(f"Failed to cast {name}.{col} to {dtype}")

        # Parse dates
        for dtcol in DATE_COLS.get(name, []):
            if dtcol in df.columns:
                df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce")

        raw[name] = df

    return raw


def prepare_full_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Validate
    if raw.get("orders", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'.")
    if raw.get("order_lines", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'.")

    # Base merge
    orders = raw["orders"].rename(columns={"CreatedAt": "CreatedAt_order"})
    lines  = raw["order_lines"]
    df = lines.merge(orders, on="OrderId", how="inner", suffixes=("", "_ord"))
    logger.info(f"After orders↔lines join: {len(df):,} rows")

    # Lookup definitions
    lookups = {
        "customers": {
            "key": "CustomerId",
            "cols": ["CustomerId", "RegionId", "CustomerName"],
            "df": raw.get("customers", pd.DataFrame()),
        },
        "products": {
            "key": "ProductId",
            "cols": ["ProductId", "SupplierId", "UnitOfBillingId", "ProductName", "SKU", "IsProduction", "ProductListPrice", "CostPrice"],
            "df": raw.get("products", pd.DataFrame()),
        },
        "regions": {
            "key": "RegionId",
            "cols": ["RegionId", "RegionName"],
            "df": raw.get("regions", pd.DataFrame()),
        },
        "shippers": {
            "key": "ShipperId",
            "cols": ["ShipperId", "Carrier"],
            "df": raw.get("shippers", pd.DataFrame()),
        },
        "suppliers": {
            "key": "SupplierId",
            "cols": ["SupplierId", "SupplierName"],
            "df": raw.get("suppliers", pd.DataFrame()),
        },
        "smethods": {
            "key": "ShippingMethodRequested",
            "cols": ["ShippingMethodRequested", "ShippingMethodName"],
            "df": raw.get("shipping_methods", pd.DataFrame()),
            "rename": {"ShippingMethodId": "ShippingMethodRequested"},
        },
    }

    # Perform merges, skipping missing columns
    for name, info in lookups.items():
        lkdf = info["df"]
        if lkdf is None or lkdf.empty:
            logger.warning(f"Skipping empty lookup table: {name}")
            continue

        # Rename if needed
        for old, new in info.get("rename", {}).items():
            if old in lkdf.columns:
                lkdf = lkdf.rename(columns={old: new})

        key = info["key"]
        if key not in lkdf.columns or key not in df.columns:
            logger.warning(f"Skipping merge for '{name}'—key '{key}' not in both tables.")
            continue

        # Only select the cols that actually exist
        valid_cols = [c for c in info["cols"] if c in lkdf.columns]
        sub = lkdf[valid_cols]

        df = df.merge(sub, on=key, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # Packs aggregation
    packs = raw.get("packs", pd.DataFrame())
    if not packs.empty and "PickedForOrderLine" in packs.columns:
        packs = packs.rename(columns={"PickedForOrderLine": "OrderLineId"})
        psum = (
            packs.groupby("OrderLineId", as_index=False)
                 .agg(
                     WeightLb=("WeightLb", "sum"),
                     ItemCount=("ItemCount", "sum"),
                     DeliveryDate=("ShippedAt", "max")
                 )
        )
        psum["DeliveryDate"] = pd.to_datetime(psum["DeliveryDate"], errors="coerce")
        df = df.merge(psum, on="OrderLineId", how="left")
    else:
        df["WeightLb"] = 0.0
        df["ItemCount"] = 0.0
        df["DeliveryDate"] = pd.NaT

    # Numeric conversions & business logic
    for col in ["QuantityShipped", "Price", "CostPrice", "WeightLb", "ItemCount"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

    df["ShippedWeightLb"] = np.where(
        df.get("UnitOfBillingId") == "3",
        df["WeightLb"],
        df["ItemCount"]
    )
    df["Revenue"] = np.where(
        df.get("IsProduction") != "1",
        df["ShippedWeightLb"] * df.get("Price", 0.0),
        0.0
    )
    df["Cost"] = np.where(
        df.get("IsProduction") != "1",
        df["ShippedWeightLb"] * df.get("CostPrice", 0.0),
        0.0
    )
    df["Profit"] = df["Revenue"] - df["Cost"]

    # Final date columns
    df["Date"] = pd.to_datetime(df["CreatedAt_order"], errors="coerce").dt.normalize()
    df["ShipDate"] = pd.to_datetime(df.get("DateShipped"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")
    df["TransitDays"] = (
        df["DeliveryDate"] - df["ShipDate"]
    ).dt.days.clip(lower=0).fillna(0).astype("int32")
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"],
        "On Time",
        "Late"
    ).astype("category")

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
