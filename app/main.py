from fastapi import FastAPI, HTTPException
from app.schemas import ApartmentInput, PredictionOutput
from src.inference.predict import predict_apartment_price

app = FastAPI(
    title='Apartment Price Prediction API',
    description='Сервис для оценки стоимости квартир на основе ML-модели',
    version='1.0.0'
)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "API для оценки стоимости недвижимости запущено"}

@app.post("/predict", response_model=PredictionOutput)
def predict_price(apartment: ApartmentInput):
    try:
        input_data = apartment.model_dump()
        
        # Получаем число с плавающей точкой из инференса
        price = predict_apartment_price(input_data)
        
        # Передаем это число в выходную Pydantic схему
        return PredictionOutput(predicted_price=price)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при инференсе: {str(e)}")