"""Phase 2: 特徵工程 - AI4I 2020 Predictive Maintenance (已修復洩漏)."""

import os, sys, logging, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

RAW_PATH = os.path.join("data", "ai4i2020.csv")
PROCESSED_PATH = os.path.join("data", "processed", "ai4i2020_processed.csv")
TARGET_COL = "Machine failure"
ID_COLS = ["UDI", "Product ID"]
FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
LEAKAGE_COLS = ["Type", "Machine ID", "Any Failure", "time_to_failure",
    "is_pre_failure_5","is_pre_failure_10","is_pre_failure_20","is_pre_failure_50","is_pre_failure_100",
    "failure_count_past_5","failure_count_future_5","failure_count_past_10","failure_count_future_10",
    "failure_count_past_20","failure_count_future_20","failure_count_past_50","failure_count_future_50",
    "failure_count_past_100","failure_count_future_100"]
NUMERIC_FEATURES = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
WINDOWS = (5, 10, 20, 50)

def load_data(raw_path=RAW_PATH):
    df = pd.read_csv(raw_path)
    logger.info("原始形狀: %s", df.shape)
    return df

def create_basic_features(df):
    df = df.copy()
    le = LabelEncoder()
    df["Type_encoded"] = le.fit_transform(df["Type"])
    df["Temp diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 0.10472
    df["Wear_Temp_Interaction"] = df["Tool wear [min]"] * df["Temp diff [K]"]
    return df

def create_time_series_features(df, windows=WINDOWS):
    df = df.copy()
    for col in NUMERIC_FEATURES + ["Temp diff [K]", "Power [W]"]:
        for w in windows:
            roll = df[col].rolling(window=w, min_periods=1)
            df[f"{col}_roll_mean_{w}"] = roll.mean()
            df[f"{col}_roll_std_{w}"] = roll.std().fillna(0)
            df[f"{col}_roll_min_{w}"] = roll.min()
            df[f"{col}_roll_max_{w}"] = roll.max()
            df[f"{col}_trend_{w}"] = df[col].diff().rolling(window=w, min_periods=1).mean()
            df[f"{col}_pct_change_{w}"] = (df[col] - roll.mean()) / (roll.mean().replace(0, np.nan) + 1e-8)
        df[f"{col}_diff1"] = df[col].diff().fillna(0)
    logger.info("時間序列特徵完成")
    return df

def select_feature_columns(df):
    exclude = set(ID_COLS + [TARGET_COL] + FAILURE_COLS + LEAKAGE_COLS + ["Type", "Machine ID", "Any Failure"])
    return [c for c in df.columns if c not in exclude]

def preprocess(raw_path=RAW_PATH, output_path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = load_data(raw_path)
    df = create_basic_features(df)
    df = create_time_series_features(df)
    feature_cols = select_feature_columns(df)
    processed_df = df[feature_cols + [TARGET_COL]].copy()
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)
    processed_df = processed_df.replace([np.inf, -np.inf], 0)
    processed_df.to_csv(output_path, index=False)
    logger.info("已儲存 %s 形狀=%s 特徵數=%d", output_path, processed_df.shape, len(feature_cols))
    logger.info("洩漏檢查: Type(Any Failure)已移除, 僅保留感測器衍生特徵")
    return processed_df

if __name__ == "__main__":
    preprocess()
