from typing import Optional
import numpy as np
from fastapi import FastAPI , Response, status, Request
from pydantic import BaseModel
import employee_attrition_model
import uvicorn
import pickle

app = FastAPI()
model = pickle.load(open('Pickle_LR_Model.pkl', 'rb'))


class Item(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: float
    average_montly_hours: float
    time_spend_company: int
    Work_accident: int
    left: int
    promotion_last_5years: int
    Department_IT: int
    Department_RandD: int
    Department_accounting: int
    Department_hr: int
    Department_management: int
    Department_marketing: int
    Department_product_mng: int
    Department_sales: int
    Department_support: int
    Department_technical: int
    salary_high: int
    salary_low: int
    salary_medium: int


@app.get('/')
def read_root():
    return {"Hello": "World"}

@app.post('/hr_attrition_predictor')
def predict_api(item:Item):
    """rest api post request with image"""
    print("Sending the data to predictor")
    print(item.average_montly_hours)
    print(np.array(item).reshape(1,-1))
    answer = model.predict(np.array(item.value()).reshape(1,-1))
    return answer

@app.get('/hr_attrition_predictor/accuracy')
def get_accuracy():
    """rest api get request"""
    return employee_attrition_model.accuracy_score



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")