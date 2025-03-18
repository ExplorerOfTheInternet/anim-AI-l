import tensorflow as tf
import json
import keras
from keras import layers

#modell laden
model_path = "../model/model.json"
weights_path = "../model/weights.bin"

with open(model_path, "r") as f:
    model_config = json.load(f)

model = tf.keras.models.model_from_json(json.dumps(model_config))
model.load_weights(weights_path)

print("Das Modell wurde erfolgreich geladen.")
