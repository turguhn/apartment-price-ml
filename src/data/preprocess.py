import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 10. Data Cleaning
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.drop_duplicates()
    print(f"   [Очистка]: Удалено {len(df) - len(df_cleaned)} дубликатов.")
    return df_cleaned


def fix_wrong_values(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    df_cleaned.loc[df_cleaned["area"] <= 0, "area"] = np.nan
    df_cleaned.loc[df_cleaned["rooms"] <= 0, "rooms"] = np.nan
    df_cleaned.loc[df_cleaned["year"] > 2026, "year"] = np.nan
    return df_cleaned


def handle_nans(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    for col in ["area", "rooms", "year"]:
        if df_cleaned[col].isna().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    return df_cleaned


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    # Жестко отсекаем миллиарды по цене и опечатки в площади
    q1_p, q3_p = df_cleaned["price"].quantile(0.25), df_cleaned["price"].quantile(0.75)
    df_cleaned = df_cleaned[df_cleaned["price"] <= (q3_p + 3 * (q3_p - q1_p))]

    q1_a, q3_a = df_cleaned["area"].quantile(0.25), df_cleaned["area"].quantile(0.75)
    df_cleaned = df_cleaned[df_cleaned["area"] <= (q3_a + 3 * (q3_a - q1_a))]
    return df_cleaned


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    int_cols = ["id", "rooms", "floor", "total_floors", "year"]
    df_cleaned[int_cols] = df_cleaned[int_cols].astype(int)
    return df_cleaned


# 11. Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерирует новые признаки из сырых данных."""
    df_feat = df.copy()

    # Относительный этаж (избегаем деления на 0)
    df_feat["floor_ratio"] = df_feat["floor"] / df_feat["total_floors"].replace(
        0, 1
    )

    # Возраст здания
    df_feat["building_age"] = 2026 - df_feat["year"]

    # Средний размер комнаты
    df_feat["avg_room_size"] = df_feat["area"] / df_feat["rooms"].replace(0, 1)

    print(
        f"   [Фичи]: Сгенерированы признаки. Итого колонок: {len(df_feat.columns)}"
    )
    return df_feat


# --- 12. Деление данных (Train / Val / Test) ---
def split_data(df: pd.DataFrame, target_col: str = "price"):
    """Разделяет датасет на Train (70%), Val (15%), Test (15%)."""
    # Удаляем технический ID и целевую переменную из признаков X
    feature_cols = [col for col in df.columns if col not in ["id", target_col]]

    X = df[feature_cols]
    y = df[target_col]

    # Сначала бьем на train и временную выборку (70 / 30)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    # Временную выборку бьем пополам на валидацию и тест (15 / 15)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"   [Разделение]: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# --- Единая точка входа для очистки и подготовки ---
def preprocess_pipeline(df: pd.DataFrame):
    print("\n--- Запуск пайплайна подготовки данных ---")
    df_clean = (
        df.pipe(remove_duplicates)
        .pipe(fix_wrong_values)
        .pipe(handle_nans)
        .pipe(remove_outliers)
        .pipe(cast_types)
        .pipe(engineer_features)
    )
    return split_data(df_clean)
