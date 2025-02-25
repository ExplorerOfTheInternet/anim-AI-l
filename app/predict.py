# app/predict.py
from fastapi import APIRouter, UploadFile, File
from .utils import load_model, prepare_image

router = APIRouter()

# Modell und Klassennamen laden
model, class_names = load_model()

@router.post("/predict/")
async def predict_animal(file: UploadFile = File(...)):
    # Bild vorbereiten
    image = await prepare_image(file)

    # Vorhersage
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    return {"predicted_animal": class_name}
