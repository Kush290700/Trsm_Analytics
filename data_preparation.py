# File: data_preparation.py

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def prepare_full_data(raw: dict) -> pd.DataFrame:
    """
    Merge raw tables, compute Revenue, Cost, Profit, and delivery metrics.
    Expects at least 'orders' and 'order_lines' in raw.
    """

    # 1) ENSURE presence of the two core tables
    if "orders" not in raw or raw["orders"] is None:
        raise RuntimeError("Missing 'orders' in raw data. Cannot continue.")
    if "order_lines" not in raw or raw["order_lines"] is None:
        raise RuntimeError("Missing 'order_lines' in raw data. Cannot continue.")

    orders_df = raw["orders"]
    lines_df  = raw["order_lines"]

    # 2) FAIL FAST if no order‐lines
    if lines_df.empty:
        raise RuntimeError(
            "🚨 'order_lines' is empty—no rows were fetched. "
            "Check your date filters or database permissions."
        )

    # 3) CAST join keys to string, with presence checks
    def cast(df, col):
        if col not in df.columns:
            raise RuntimeError(f"Expected '{col}' in DataFrame but got {df.columns.tolist()}")
        df[col] = df[col].astype(str)

    for df, cols in [
        (orders_df, ["OrderId", "CustomerId", "SalesRepId", "ShippingMethodRequested"]),
        (lines_df,  ["OrderLineId", "OrderId", "ProductId", "ShipperId"])
    ]:
        for c in cols:
            cast(df, c)

    # 4) INNER JOIN orders ↔ order_lines
    df = lines_df.merge(
        orders_df,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 5) LOOKUP merges (only if present & non‐empty)
    lookups = {
        "customers":    ("CustomerId",         ["CustomerId","RegionId","CustomerName"], raw.get("customers")),
        "products":     ("ProductId",          ["ProductId","SupplierId","ProductName","ProductListPrice","CostPrice"], raw.get("products")),
        "regions":      ("RegionId",           ["RegionId","RegionName"], raw.get("regions")),
        "shippers":     ("ShipperId",          ["ShipperId","Carrier"], raw.get("shippers")),
        "suppliers":    ("SupplierId",         ["SupplierId","SupplierName"], raw.get("suppliers")),
        "smethods":     ("ShippingMethodRequested", ["ShippingMethodRequested","ShippingMethodName"], raw.get("shipping_methods")),
    }

    for name, (keycol, required_cols, lookup_df) in lookups.items():
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Lookup table '{name}' is missing or empty—skipping merge.")
            continue

        # prepare shipping methods rename
        if name == "smethods" and "SMId" in lookup_df.columns:
            lookup_df = lookup_df.rename(columns={"SMId": "ShippingMethodRequested"})

        for col in required_cols:
            cast(lookup_df, col)

        df = df.merge(lookup_df, on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 6) PACKS aggregation (optional)
    packs = raw.get("packs")
    if packs is not None and not packs.empty:
        packs = packs.copy()
        cast(packs, "PickedForOrderLine")
        packs["OrderLineId"] = packs["PickedForOrderLine"]
        psum = (
            packs.groupby("OrderLineId", as_index=False)
            .agg(
                WeightLb     = ("WeightLb", "sum"),
                ItemCount    = ("ItemCount","sum"),
                DeliveryDate = ("DeliveryDate","max")
            )
        )
        cast(psum, "OrderLineId")
        df = df.merge(psum, on="OrderLineId", how="left")
        df[["WeightLb","ItemCount"]] = df[["WeightLb","ItemCount"]].fillna(0)
        logger.info(f"After merging 'packs': {len(df):,} rows")
    else:
        # ensure columns exist downstream
        df["WeightLb"]    = 0.0
        df["ItemCount"]   = 0.0
        df["DeliveryDate"]= pd.NaT

    # 7) NUMERIC columns — coerce & fill
    numeric_cols = ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 8) SHIPPED WEIGHT logic
    per_item = df["WeightLb"] / df["ItemCount"].replace(0, np.nan)
    is_wt = (df["UnitOfBillingId"] == 3) & (df["WeightLb"] > 0)
    df["ShippedWeightLb"] = np.where(
        is_wt,
        df["WeightLb"],
        df["ItemCount"] * per_item.fillna(0)
    )

    # 9) REVENUE, COST, PROFIT
    df["Revenue"] = np.where(
        is_wt,
        df["WeightLb"] * df["SalePrice"],
        df["ItemCount"] * df["SalePrice"]
    )
    df["Cost"]    = np.where(
        is_wt,
        df["WeightLb"] * df["UnitCost"],
        df["ItemCount"] * df["UnitCost"]
    )
    df["Profit"]  = df["Revenue"] - df["Cost"]

    # 10) DATE & DELIVERY METRICS
    df["Date"]         = pd.to_datetime(df["CreatedAt_order"], errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df["ShipDate"], errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    df["DateExpected"]= pd.to_datetime(df.get("DateExpected"), errors="coerce")

    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"],
        "On Time", "Late"
    )

    logger.info(f"Prepared full data: {len(df):,} rows")
    return df