from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = "animal_model.h5"
CLASS_NAMES = ['cat', 'dog', 'lion', 'monkey']

model = load_model(MODEL_PATH)

def predict(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"class": predicted_class, "confidence": round(confidence, 3)}
