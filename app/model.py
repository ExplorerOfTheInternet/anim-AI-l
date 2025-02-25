# app/model.py
import os
import zipfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dataset herunterladen und extrahieren
def download_and_extract_data():
    import kagglehub
    path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
    dataset_path = os.path.join(path, "animal-image-dataset.zip")
    extract_path = "./data/animals"
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path

# Bildverarbeitung vorbereiten
def prepare_data(extract_path):
    img_size = (128, 128)
    batch_size = 32  
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = datagen.flow_from_directory(extract_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
    val_data = datagen.flow_from_directory(extract_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')
    return train_data, val_data

# Modell erstellen und trainieren
def create_and_train_model(train_data, val_data):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(train_data.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save("model/animal_classifier.h5")
    return model

# Modell laden
def load_model():
    model = tf.keras.models.load_model("model/animal_classifier.h5")
    class_names = list(train_data.class_indices.keys())
    return model, class_names
