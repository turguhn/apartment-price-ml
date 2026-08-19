import os
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def get_connection_engine():
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "1712")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "apartment")

    engine = create_engine(
        f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

    return engine

def load_apartments_data() -> pd.DataFrame:
    """
    Загружает данные из таблицы и возвращает pandas DataFrame
    """
    query = "SELECT * FROM apartments;"
    engine = get_connection_engine()

    with engine.connect() as connection:
        df = pd.read_sql_query(query, connection)

    return df


def introduce_problems(df: pd.DataFrame) -> pd.DataFrame:
    """Намеренно портит датасет, добавляя пропуски, дубликаты,

    выбросы и аномальные значения.
    """
    df_corrupted = df.copy()
    np.random.seed(42)  # Фиксируем seed для воспроизводимости результатов

    # 1. Добавляем пропуски (NaN) — около 5% в площади и годе постройки
    mask_area_nan = np.random.rand(len(df_corrupted)) < 0.05
    df_corrupted.loc[mask_area_nan, "area"] = np.nan

    mask_year_nan = np.random.rand(len(df_corrupted)) < 0.03
    df_corrupted.loc[mask_year_nan, "year"] = np.nan

    # 2. Добавляем полные дубликаты (duplicates) — скопируем 50 случайных строк
    duplicate_rows = df_corrupted.sample(n=50, random_state=42)
    df_corrupted = pd.concat([df_corrupted, duplicate_rows], ignore_index=True)

    # 3. Вносим неверные/аномальные значения (wrong values)
    # Отрицательная площадь (у 10 квартир)
    bad_area_idx = df_corrupted.sample(n=10, random_state=1).index
    df_corrupted.loc[bad_area_idx, "area"] = -10.0

    # 0 комнат (у 15 квартир)
    bad_rooms_idx = df_corrupted.sample(n=15, random_state=2).index
    df_corrupted.loc[bad_rooms_idx, "rooms"] = 0

    # Год из далекого будущего (у 5 квартир)
    bad_year_idx = df_corrupted.sample(n=5, random_state=3).index
    df_corrupted.loc[bad_year_idx, "year"] = 3500

    # 4. Добавляем жесткие выбросы (outliers) в цену и площадь
    # Сверхдорогая квартира (мультимиллиардер)
    rich_idx = df_corrupted.sample(n=3, random_state=4).index
    df_corrupted.loc[rich_idx, "price"] = 9_999_999_999.0

    # Огромная площадь при маленькой цене (опечатка в данных)
    huge_area_idx = df_corrupted.sample(n=2, random_state=5).index
    df_corrupted.loc[huge_area_idx, "area"] = 1500.0

    return df_corrupted

if __name__ == "__main__":
    # Блок для проверки работы скрипта
    print("Запуск загрузки данных из PostgreSQL...")
    try:
        apartments_df = load_apartments_data()
        print(f"Успешно загружено строк: {len(apartments_df)}")
        print("\nПервые 5 строк DataFrame:")
        print(apartments_df.head())

        print("Искажение данных...")
        dirty_df = introduce_problems(apartments_df)
        print(f"Новый размер с дубликатами: {len(dirty_df)}")
        print(f"Количество NaN в 'area': {dirty_df['area'].isna().sum()}")
        print(f"Квартир с площадью -10: {len(dirty_df[dirty_df['area'] == -10])}")
        print(f"Максимальная цена: {dirty_df['price'].max() :_}")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")