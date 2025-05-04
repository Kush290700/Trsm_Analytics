import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Desired dtypes for each table
CSV_DTYPES = {
    "orders": {"OrderId":"category","CustomerId":"category","SalesRepId":"category","ShippingMethodRequested":"category","OrderStatus":"category"},
    "order_lines": {"OrderLineId":"category","OrderId":"category","ProductId":"category","ShipperId":"category","QuantityShipped":"float32","Price":"float32","CostPrice":"float32"},
    "products": {"ProductId":"category","SupplierId":"category","UnitOfBillingId":"category","ProductName":"category","ProductListPrice":"float32","CostPrice":"float32","IsProduction":"category","SKU":"category"},
    "customers": {"CustomerId":"category","RegionId":"category","CustomerName":"category"},
    "regions": {"RegionId":"category","RegionName":"category"},
    "shippers": {"ShipperId":"category","Carrier":"category"},
    "suppliers": {"SupplierId":"category","SupplierName":"category"},
    "shipping_methods": {"ShippingMethodId":"category","ShippingMethodName":"category"},
    "packs": {"PickedForOrderLine":"category","WeightLb":"float32","ItemCount":"float32"},
}

# Columns to parse as dates if they exist
DATE_COLS = {
    "orders": ["CreatedAt","DateOrdered","DateExpected","DateShipped"],
    "order_lines": ["CreatedAt","DateShipped"],
    "packs": ["ShippedAt"],
}


def load_csv_tables(csv_dir="data") -> dict[str, pd.DataFrame]:
    table_names = ["orders","order_lines","products","customers","regions","shippers","suppliers","shipping_methods","packs"]
    raw = {}
    for name in table_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.exists(path):
            logger.warning(f"Missing {name}.csv")
            raw[name] = pd.DataFrame()
            continue
        df = pd.read_csv(path, low_memory=False)
        # Cast existing columns
        for col,dtype in CSV_DTYPES.get(name,{}).items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception:
                    logger.warning(f"Failed to cast {name}.{col} to {dtype}")
        # Parse date cols
        for dtcol in DATE_COLS.get(name,[]):
            if dtcol in df.columns:
                df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce")
        raw[name] = df
    return raw


def prepare_full_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Ensure required tables
    if raw.get("orders",pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'orders'.")
    if raw.get("order_lines",pd.DataFrame()).empty:
        raise RuntimeError("Missing or empty 'order_lines'.")

    orders = raw["orders"].rename(columns={"CreatedAt":"CreatedAt_order"})
    lines  = raw["order_lines"]
    df = lines.merge(orders, on="OrderId", how="inner", suffixes=("","_ord"))
    logger.info(f"After join: {len(df)} rows")

    # Lookup merges
    lookups = {
        "customers": ("CustomerId", raw.get("customers")),
        "products":  ("ProductId", raw.get("products")),
        "regions":   ("RegionId", raw.get("regions")),
        "shippers":  ("ShipperId", raw.get("shippers")),
        "suppliers": ("SupplierId", raw.get("suppliers")),
        "smethods":  ("ShippingMethodRequested", raw.get("shipping_methods")),
    }
    for name,(key,lkdf) in lookups.items():
        if lkdf is None or lkdf.empty:
            logger.warning(f"Skipping empty lookup: {name}")
            continue
        if name=="smethods":
            lkdf = lkdf.rename(columns={"ShippingMethodId":"ShippingMethodRequested"})
        # pick only existing cols plus key
        cols = [c for c in lkdf.columns if c != key]
        sub = lkdf[[key]+cols]
        df = df.merge(sub, on=key, how="left")
        logger.info(f"After merge {name}: {len(df)} rows")

    # Packs
    packs = raw.get("packs",pd.DataFrame())
    if not packs.empty and "PickedForOrderLine" in packs.columns:
        packs = packs.rename(columns={"PickedForOrderLine":"OrderLineId"})
        agg = packs.groupby("OrderLineId",as_index=False).agg(
            WeightLb=("WeightLb","sum"),
            ItemCount=("ItemCount","sum"),
            DeliveryDate=("ShippedAt","max")
        )
        agg["DeliveryDate"] = pd.to_datetime(agg["DeliveryDate"],errors="coerce")
        df = df.merge(agg,on="OrderLineId",how="left")
    else:
        df["WeightLb"] = 0.0; df["ItemCount"] = 0.0; df["DeliveryDate"] = pd.NaT

    # Numeric downcast & business logic
    for col in ["QuantityShipped","Price","CostPrice","WeightLb","ItemCount"]:
        if col in df:
            df[col] = pd.to_numeric(df[col],errors="coerce").fillna(0).astype("float32")
    df["ShippedWeightLb"] = np.where(df.get("UnitOfBillingId")=="3", df.WeightLb, df.ItemCount)
    df["Revenue"] = np.where(df.get("IsProduction")!="1", df.ShippedWeightLb*df.get("Price",0), 0.0)
    df["Cost"]    = np.where(df.get("IsProduction")!="1", df.ShippedWeightLb*df.get("CostPrice",0),0.0)
    df["Profit"]  = df.Revenue - df.Cost

    # Dates
    df["Date"]         = pd.to_datetime(df["CreatedAt_order"],errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("DateShipped"),errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"),errors="coerce")
    df["TransitDays"]  = (df.DeliveryDate - df.ShipDate).dt.days.clip(lower=0).fillna(0).astype("int32")
    df["DeliveryStatus"] = np.where(df.DeliveryDate<=df.DateExpected,"On Time","Late").astype("category")

    logger.info(f"Prepared full data: {len(df)} rows")
    return df
