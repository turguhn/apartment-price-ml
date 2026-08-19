from pydantic import BaseModel, Field

class ApartmentInput(BaseModel):
    area: float = Field(..., gt=0, description="Площадь квартиры в кв.м.")
    rooms: int = Field(..., gt=0, description="Количество комнат")
    floor: int = Field(..., gt=0, description="Этаж квартиры")
    total_floors: int = Field(..., gt=0, description="Всего этажей в здании")
    year: int = Field(..., gt=1800, le=2026, description="Год постройки")

class PredictionOutput(BaseModel):
    predicted_price: float