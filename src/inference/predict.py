import os
import sys
import pickle
import pandas as pd

# Добавляем корневую папку в пути, чтобы Python видел модули src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


MODEL_PATH = "models/model.pkl"

def load_trained_model():
    """Загружает сохраненную модель из файла"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Файл модели не найден по пути {MODEL_PATH}')

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_apartment_price(input_data: dict) -> float:
    """
    Принимает сырые данные об одной квартире,
    трансформирует их в фичи и возвращает
    предсказанную стоимость
    """

    df = pd.DataFrame([input_data])

    df["floor_ratio"] = df["floor"] / df["total_floors"].replace(0, 1)
    df["building_age"] = 2026 - df["year"]
    df["avg_room_size"] = df["area"] / df["rooms"].replace(0, 1)

    feature_order = ["area", "rooms", "floor", "total_floors", "year", "floor_ratio", "building_age", "avg_room_size"]
    X = df[feature_order]

    model = load_trained_model()
    prediction = model.predict(X)

    return float(prediction[0])

if __name__ == "__main__":
    test_apartment = {
        "area": 60.5,
        "rooms": 2,
        "floor": 5,
        "total_floors": 9,
        "year": 2015
    }

    print('Тестирование модуля инференса...')
    try:
        price = predict_apartment_price(test_apartment)
        print(f'Входные данные: {test_apartment}')
        print(f'Предсказанная стоимость: {price:,.2f} рублей')
    except Exception as e:
        print(f'Ошибка инференса: {e}')