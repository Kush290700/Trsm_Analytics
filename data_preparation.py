import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

def load_csv_tables(csv_dir="data") -> dict:
    """
    Load each of the nine CSVs into a DataFrame (or empty DataFrame if missing).
    """
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
            logger.warning(f"⚠️ Missing table: {name}.csv — using empty DataFrame")
    return raw

def prepare_full_data(raw: dict) -> pd.DataFrame:
    """
    Merge and prepare the full analytic table from the raw CSV dict.
    """
    # 1) Basic existence checks
    if raw.get("orders", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'. Cannot continue.")
    if raw.get("order_lines", pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'. Cannot continue.")

    orders_df = raw["orders"].copy()
    lines_df  = raw["order_lines"].copy()

    # 2) Ensure key columns exist (cast to str if present, else blank)
    def safe_cast(df, col):
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = ""
            logger.warning(f"⚠️ Column '{col}' not found; filling with empty strings")

    for df_, cols in [
        (orders_df, ["OrderId", "CustomerId", "SalesRepId", "ShippingMethodRequested"]),
        (lines_df,  ["OrderLineId", "OrderId", "ProductId",  "ShipperId"])
    ]:
        for c in cols:
            safe_cast(df_, c)

    # 3) Join order_lines → orders
    df = lines_df.merge(
        orders_df,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 4) Lookup merges: (table_key, required cols, optional rename map)
    lookups = {
        "customers":        ("CustomerId", ["RegionId","CustomerName"], {}),
        "products":         ("ProductId",  ["SupplierId","UnitOfBillingId","ProductName","ProductListPrice","CostPrice","IsProduction"], {}),
        "regions":          ("RegionId",   ["RegionName"], {}),
        "shippers":         ("ShipperId",  ["Carrier"], {}),
        "suppliers":        ("SupplierId", ["SupplierName"], {}),
        "shipping_methods": ("ShippingMethodRequested", ["ShippingMethodName"], {"ShippingMethodId":"ShippingMethodRequested"})
    }

    for name, (keycol, req_cols, rename_map) in lookups.items():
        lookup_df = raw.get(name, pd.DataFrame()).copy()
        if lookup_df.empty:
            logger.warning(f"Skipping merge '{name}'—table empty.")
            continue

        # rename any columns (e.g. ShippingMethodId → ShippingMethodRequested)
        for old, new in rename_map.items():
            if old in lookup_df.columns:
                lookup_df = lookup_df.rename(columns={old:new})

        # must have the key
        if keycol not in lookup_df.columns or keycol not in df.columns:
            logger.warning(f"Skipping merge '{name}'—key '{keycol}' missing.")
            continue

        # only keep required cols that actually exist, *excluding* the key itself
        valid_cols = [c for c in req_cols if c in lookup_df.columns and c != keycol]
        cols_to_take = [keycol] + valid_cols

        # cast each to string
        for c in cols_to_take:
            safe_cast(lookup_df, c)

        df = df.merge(
            lookup_df[cols_to_take],
            on=keycol,
            how="left"
        )
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 5) Packs aggregation (optional)
    packs = raw.get("packs", pd.DataFrame()).copy()
    if not packs.empty:
        safe_cast(packs, "PickedForOrderLine")
        packs = packs.rename(columns={"PickedForOrderLine": "OrderLineId"})

        # only aggregate columns that exist
        agg_dict = {}
        if "WeightLb" in packs.columns:      agg_dict["WeightLb"] = ("WeightLb", "sum")
        if "ItemCount" in packs.columns:     agg_dict["ItemCount"] = ("ItemCount", "sum")
        if "DeliveryDate" in packs.columns:  agg_dict["DeliveryDate"] = ("DeliveryDate", "max")

        if not agg_dict:
            logger.warning("No aggregatable columns in 'packs'; skipping.")
        else:
            psum = (
                packs
                .groupby("OrderLineId", as_index=False)
                .agg(**agg_dict)
            )
            safe_cast(psum, "OrderLineId")
            df = df.merge(psum, on="OrderLineId", how="left")
            for col in ["WeightLb","ItemCount"]:
                if col in df:
                    df[col] = df[col].fillna(0.0)
            logger.info(f"After merging 'packs': {len(df):,} rows")

    else:
        df["WeightLb"]    = 0.0
        df["ItemCount"]   = 0.0
        df["DeliveryDate"]= pd.NaT

    # 6) Numeric conversions
    num_cols = ["QuantityShipped","SalePrice","CostPrice","UnitCost","WeightLb","ItemCount"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # unify cost columns if necessary
    if "CostPrice" in df and "UnitCost" not in df:
        df["UnitCost"] = df["CostPrice"]

    # 7) Compute shipped weight
    if "UnitOfBillingId" in df.columns:
        df["UnitOfBillingId"] = df["UnitOfBillingId"].astype(str)
    # weight‐based if UnitOfBillingId == "3"
    df["ShippedWeightLb"] = np.where(
        df.get("UnitOfBillingId","") == "3",
        df.get("WeightLb",0.0),
        df.get("ItemCount",0.0)
    )

    # 8) Revenue / cost / profit, excluding production if flagged
    df["Revenue"] = df["ShippedWeightLb"] * df.get("SalePrice",0.0)
    df["Cost"]    = df["ShippedWeightLb"] * df.get("UnitCost",0.0)
    df["Profit"]  = df["Revenue"] - df["Cost"]

    if "IsProduction" in df.columns:
        # treat non‐"1" as 0
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        prod_mask = df["IsProduction"] == 1
        df.loc[prod_mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {prod_mask.sum():,} production rows from margin")

    # 9) Dates & delivery metrics
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order", df.get("CreatedAt")), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate"), errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")

    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"],
        "On Time", "Late"
    )

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
