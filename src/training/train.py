import os
import sys
import pickle

# Добавляем корневую папку в пути, чтобы Python видел модуль src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.data.load import load_apartments_data, introduce_problems
from src.data.preprocess import preprocess_pipeline


def run_tuning_and_final_train():
    # 1. Загрузка и подготовка данных
    print("Шаг 1: Загрузка данных из БД...")
    raw_df = load_apartments_data()
    dirty_df = introduce_problems(raw_df)

    print("Шаг 2: Предобработка и генерация признаков...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(dirty_df)

    # 2. Пункт 15: Ручной подбор гиперпараметров (Экспериментируем с конфигурациями)
    print("\nШаг 3: Эксперименты с гиперпараметрами Gradient Boosting...")
    print("-" * 65)
    print(f"{'Конфигурация':<35} | {'Валидационный MAE':<20}")
    print("-" * 65)

    configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},  # Дефолт
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1},  # Больше деревьев
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},  # Глубже деревья
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05}, # Баланс: глубже, но шаг меньше
    ]

    best_mae = float("inf")
    best_config = None

    for config in configs:
        model = GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        
        config_str = f"trees={config['n_estimators']}, depth={config['max_depth']}, lr={config['learning_rate']}"
        print(f"{config_str:<35} | {mae_val:,.0f} руб.")
        
        if mae_val < best_mae:
            best_mae = mae_val
            best_config = config

    print("-" * 65)
    print(f"Лучшая конфигурация: {best_config}")

    # 3. Пункт 16: Финальное обучение (Train + Validation)
    print("\nШаг 4: Финальное обучение модели на объединенных данных (Train + Val)...")
    
    # Объединяем тренировочную и валидационную выборки для финального обучения
    X_final_train = pd.concat([X_train, X_val])
    y_final_train = pd.concat([y_train, y_val])

    final_model = GradientBoostingRegressor(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        learning_rate=best_config["learning_rate"],
        random_state=42
    )
    final_model.fit(X_final_train, y_final_train)

    # 4. Финальный честный тест (на отложенной выборке Test, которую мы не подсматривали)
    print("\nШаг 5: Финальное тестирование на скрытых данных (Test)...")
    y_pred_test = final_model.predict(X_test)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print("-" * 50)
    print(f"Финальные метрики на тест-выборке:")
    print(f"MAE  = {mae_test:,.0f} рублей")
    print(f"RMSE = {rmse_test:,.0f} рублей")
    print(f"R²   = {r2_test:.2f}")
    print("-" * 50)

    # 5. Сохранение весов модели в файл model.pkl согласно архитектуре проекта
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Модель успешно сохранена в файл: {model_path}")


if __name__ == "__main__":
    # Импортируем pandas здесь, чтобы избежать конфликтов внутри скрипта
    import pandas as pd
    run_tuning_and_final_train()
