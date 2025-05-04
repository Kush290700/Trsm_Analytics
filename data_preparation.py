import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

def load_csv_tables(csv_dir="data") -> dict:
    table_names = [
        "orders", "order_lines", "products", "customers",
        "regions", "shippers", "suppliers", "shipping_methods", "packs"
    ]
    raw = {}
    for name in table_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        if os.path.exists(path):
            raw[name] = pd.read_csv(path, low_memory=False)
        else:
            raw[name] = pd.DataFrame()
            logger.warning(f"⚠️ Missing table: {name}.csv")
    return raw

def prepare_full_data(raw: dict) -> pd.DataFrame:
    # 1) required tables
    if raw.get("orders", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'.")
    if raw.get("order_lines", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'.")

    orders_df = raw["orders"].copy()
    lines_df  = raw["order_lines"].copy()

    # 2) cast key columns to string
    def cast(df, col):
        if col not in df.columns:
            df[col] = ""
            logger.warning(f"⚠️ Column '{col}' missing—filling with empty strings.")
        df[col] = df[col].astype(str)

    for df_, cols in [
        (orders_df,     ["OrderId","CustomerId","SalesRepId","ShippingMethodRequested"]),
        (lines_df,      ["OrderLineId","OrderId","ProductId","ShipperId"])
    ]:
        for c in cols:
            cast(df_, c)

    # 3) join orders + lines
    df = lines_df.merge(
        orders_df,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 4) lookup tables
    lookups = {
        "customers":        ("CustomerId", ["CustomerId","RegionId","CustomerName"], raw.get("customers")),
        "products":         ("ProductId",  ["ProductId","SupplierId","UnitOfBillingId","ProductName","ProductListPrice","CostPrice","IsProduction","SKU"], raw.get("products")),
        "regions":          ("RegionId",   ["RegionId","RegionName"], raw.get("regions")),
        "shippers":         ("ShipperId",  ["ShipperId","Carrier"], raw.get("shippers")),
        "suppliers":        ("SupplierId", ["SupplierId","SupplierName"], raw.get("suppliers")),
        # shipping_methods may arrive with SMId or ShippingMethodId
        "shipping_methods": (
            "ShippingMethodRequested",
            ["ShippingMethodName"],
            raw.get("shipping_methods"),
            {"SMId":"ShippingMethodRequested","ShippingMethodId":"ShippingMethodRequested"}
        ),
    }

    for name, info in lookups.items():
        keycol, cols, lookup_df, *maybe_map = info
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Skipping merge '{name}'—table empty.")
            continue

        # apply any rename_map
        if maybe_map:
            for old, new in maybe_map[0].items():
                if old in lookup_df.columns:
                    lookup_df = lookup_df.rename(columns={old:new})

        # ensure key and required cols exist
        valid_cols = [c for c in cols if c in lookup_df.columns]
        if keycol not in lookup_df.columns or keycol not in df.columns:
            logger.warning(f"Skipping merge '{name}'—key '{keycol}' missing.")
            continue

        # cast them to str
        for c in [keycol] + valid_cols:
            cast(lookup_df, c)

        df = df.merge(
            lookup_df[[keycol] + valid_cols],
            on=keycol,
            how="left"
        )
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 5) make sure shipping columns always exist
    if "ShippingMethodRequested" not in df.columns:
        df["ShippingMethodRequested"] = ""
    if "ShippingMethodName" not in df.columns:
        df["ShippingMethodName"] = ""

    # 6) packs aggregation
    packs = raw.get("packs")
    if packs is not None and not packs.empty:
        packs = packs.copy()
        cast(packs, "PickedForOrderLine")
        packs["OrderLineId"] = packs["PickedForOrderLine"]
        psum = packs.groupby("OrderLineId", as_index=False).agg(
            WeightLb     = ("WeightLb", "sum"),
            ItemCount    = ("ItemCount", "sum"),
            DeliveryDate = ("DeliveryDate", "max")
        )
        cast(psum, "OrderLineId")
        df = df.merge(psum, on="OrderLineId", how="left")
        df[["WeightLb","ItemCount"]] = df[["WeightLb","ItemCount"]].fillna(0)
        logger.info(f"After merging 'packs': {len(df):,} rows")
    else:
        df["WeightLb"]     = 0.0
        df["ItemCount"]    = 0.0
        df["DeliveryDate"] = pd.NaT

    # 7) numeric conversions, shipped weight, revenue/cost/profit, drop production
    num_cols = ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["UnitOfBillingId"] = df.get("UnitOfBillingId","").astype(str)

    # weight-vs-count logic
    df["ShippedWeightLb"] = np.where(
        df["UnitOfBillingId"] == "3",
        df["WeightLb"],
        df["ItemCount"]
    )
    df["Revenue"] = df["ShippedWeightLb"] * df["SalePrice"]
    df["Cost"]    = df["ShippedWeightLb"] * df["UnitCost"]
    df["Profit"]  = df["Revenue"] - df["Cost"]

    if "IsProduction" in df.columns:
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        prod_mask = df["IsProduction"] == 1
        df.loc[prod_mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {prod_mask.sum():,} production rows from finance")

    # 8) Final dates and delivery metrics
    df["Date"] = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()

    # ShipDate fallback logic
    if "ShipDate" in df.columns:
        df["ShipDate"] = pd.to_datetime(df["ShipDate"], errors="coerce")
    elif "DateShipped" in df.columns:
        df["ShipDate"] = pd.to_datetime(df["DateShipped"], errors="coerce")
    else:
        df["ShipDate"] = pd.NaT

    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")

    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0).fillna(0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"],
        "On Time", "Late"
    )

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
