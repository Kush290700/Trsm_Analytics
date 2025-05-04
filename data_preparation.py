import logging
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

def load_csv_tables(csv_dir="data") -> dict:
    """
    Load each CSV into a DataFrame (or an empty one if missing).
    """
    files = {
        "orders":            "orders.csv",
        "order_lines":       "order_lines.csv",
        "products":          "products.csv",
        "customers":         "customers.csv",
        "regions":           "regions.csv",
        "shippers":          "shippers.csv",
        "suppliers":         "suppliers.csv",
        "shipping_methods":  "shipping_methods.csv",
        "packs":             "packs.csv",
    }
    raw = {}
    for table, fname in files.items():
        path = os.path.join(csv_dir, fname)
        if os.path.exists(path):
            raw[table] = pd.read_csv(path, low_memory=False)
        else:
            raw[table] = pd.DataFrame()
            logger.warning(f"⚠️ Missing {fname}: loaded empty DataFrame")
    return raw

def prepare_full_data(raw: dict) -> pd.DataFrame:
    """
    Merge and prepare the full analytics table.
    """
    # 1) Validate existence
    orders = raw.get("orders", pd.DataFrame())
    lines  = raw.get("order_lines", pd.DataFrame())
    if orders.empty:
        raise RuntimeError("Missing or empty orders.csv")
    if lines.empty:
        raise RuntimeError("Missing or empty order_lines.csv")

    # 2) Cast all key columns to str (or fill blanks if missing)
    def cast_str(df, col):
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = ""
            logger.warning(f"⚠️ Column '{col}' not found; filling with empty strings")

    for df, cols in [
        (orders,   ["OrderId","CustomerId","SalesRepId","ShippingMethodRequested"]),
        (lines,    ["OrderLineId","OrderId","ProductId","ShipperId"])
    ]:
        for c in cols:
            cast_str(df, c)

    # 3) Merge lines → orders
    df = lines.merge(
        orders,
        on="OrderId",
        how="inner",
        suffixes=("", "_ord")
    )
    logger.info(f"After orders+lines join: {len(df):,} rows")

    # 4) Lookup tables
    # — customers
    if "CustomerId" in df.columns and not raw["customers"].empty:
        cust = raw["customers"].copy()
        for c in ["CustomerId","CustomerName","RegionId","IsRetail"]:
            cast_str(cust, c)
        df = df.merge(
            cust[["CustomerId","CustomerName","RegionId","IsRetail"]],
            on="CustomerId", how="left"
        )
        logger.info(f"After customers merge: {len(df):,} rows")

    # — products
    if "ProductId" in df.columns and not raw["products"].empty:
        prod = raw["products"].copy()
        for c in ["ProductId","SKU","ProductName","UnitOfBillingId","SupplierId","ProductListPrice","CostPrice"]:
            cast_str(prod, c)
        # supply a default IsProduction if missing
        cast_str(prod, "IsProduction")
        df = df.merge(
            prod[[
                "ProductId","SKU","ProductName",
                "UnitOfBillingId","SupplierId",
                "ProductListPrice","CostPrice","IsProduction"
            ]],
            on="ProductId", how="left"
        )
        logger.info(f"After products merge: {len(df):,} rows")

    # — regions
    if "RegionId" in df.columns and not raw["regions"].empty:
        reg = raw["regions"].copy()
        for c in ["RegionId","RegionName"]:
            cast_str(reg, c)
        df = df.merge(reg[["RegionId","RegionName"]], on="RegionId", how="left")
        logger.info(f"After regions merge: {len(df):,} rows")

    # — shippers
    if "ShipperId" in df.columns and not raw["shippers"].empty:
        shp = raw["shippers"].copy()
        for c in ["ShipperId","Carrier"]:
            cast_str(shp, c)
        df = df.merge(shp[["ShipperId","Carrier"]], on="ShipperId", how="left")
        logger.info(f"After shippers merge: {len(df):,} rows")

    # — suppliers
    if "SupplierId" in df.columns and not raw["suppliers"].empty:
        sup = raw["suppliers"].copy()
        for c in ["SupplierId","SupplierName"]:
            cast_str(sup, c)
        df = df.merge(sup[["SupplierId","SupplierName"]], on="SupplierId", how="left")
        logger.info(f"After suppliers merge: {len(df):,} rows")

    # — shipping methods
    if "ShippingMethodRequested" in df.columns and not raw["shipping_methods"].empty:
        sm = raw["shipping_methods"].copy()
        # rename SMId → ShippingMethodRequested
        if "SMId" in sm.columns:
            sm = sm.rename(columns={"SMId":"ShippingMethodRequested"})
        for c in ["ShippingMethodRequested","ShippingMethodName"]:
            cast_str(sm, c)
        # only merge if the key is actually present
        df = df.merge(
            sm[["ShippingMethodRequested","ShippingMethodName"]],
            on="ShippingMethodRequested", how="left"
        )
        logger.info(f"After shipping_methods merge: {len(df):,} rows")

    # 5) Packs aggregation
    packs = raw["packs"].copy()
    if not packs.empty:
        # rename and cast
        cast_str(packs, "PickedForOrderLine")
        packs = packs.rename(columns={"PickedForOrderLine":"OrderLineId"})
        # build up any aggregates that exist
        agg = {}
        if "WeightLb"    in packs.columns: agg["WeightLb"]    = ("WeightLb","sum")
        if "ItemCount"   in packs.columns: agg["ItemCount"]   = ("ItemCount","sum")
        if "DeliveryDate" in packs.columns: agg["DeliveryDate"] = ("DeliveryDate","max")

        if agg:
            psum = packs.groupby("OrderLineId", as_index=False).agg(**agg)
            cast_str(psum, "OrderLineId")
            df = df.merge(psum, on="OrderLineId", how="left")
            # fill zeros
            for c in ["WeightLb","ItemCount"]:
                if c in df: 
                    df[c] = df[c].fillna(0.0)
            logger.info(f"After packs merge: {len(df):,} rows")
        else:
            logger.warning("No aggregatable columns in packs; skipped.")
    else:
        # ensure columns exist
        df["WeightLb"]    = 0.0
        df["ItemCount"]   = 0.0
        df["DeliveryDate"]= pd.NaT

    # 6) Numeric conversions
    for c in ["QuantityShipped","SalePrice","UnitCost","ProductListPrice","CostPrice","WeightLb","ItemCount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # unify cost column if needed
    if "UnitCost" not in df.columns and "CostPrice" in df.columns:
        df["UnitCost"] = df["CostPrice"]

    # 7) Shipped weight
    df["UnitOfBillingId"] = df.get("UnitOfBillingId","").astype(str)
    df["ShippedWeightLb"] = np.where(
        df["UnitOfBillingId"] == "3",
        df["WeightLb"],
        df["ItemCount"]
    )

    # 8) Revenue / Cost / Profit
    df["Revenue"] = df["ShippedWeightLb"] * df["SalePrice"]
    df["Cost"]    = df["ShippedWeightLb"] * df["UnitCost"]
    df["Profit"]  = df["Revenue"] - df["Cost"]

    # zero out production if flagged
    if "IsProduction" in df.columns:
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        mask = df["IsProduction"] == 1
        df.loc[mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {mask.sum():,} production rows")

    # 9) Final dates & delivery metrics
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate"),      errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"),  errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"),  errors="coerce")

    df["TransitDays"] = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(
        df["DeliveryDate"] <= df["DateExpected"],
        "On Time",
        "Late"
    )

    logger.info(f"✅ Prepared full data: {len(df):,} rows")
    return df
