from pyexpat import model
import tensorflow as tf
import numpy as np
import cv2

def predict(image_path, model):
    #1. das bild wird geladen
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) 
    img = img.astype("float32") / 255.0  
    img = np.expand_dims(img, axis=0)  

    #2. da wird die vorhersage gemacht
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)  #-> hier wird die Klasse mit der höchsten wahrscheinlichkeit genommen

    return predicted_class, predictions

#kurzes bsp zum schaun obs geht
image_path = "../test.jpg" 
predicted_class, predictions = predict(image_path, model)
print(f"Vorhergesagte Klasse: {predicted_class}")
