import pandas as pd

def min_max_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Chuẩn hóa min-max cho các cột được chỉ định."""
    df_norm = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.0  # Nếu giá trị không đổi
    return df_norm
def z_score_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Chuẩn hóa Z-score cho các cột được chỉ định."""
    df_norm = df.copy()
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val != 0:
            df_norm[col] = (df[col] - mean_val) / std_val
        else:
            df_norm[col] = 0.0  # Nếu độ lệch chuẩn là 0
    return df_norm