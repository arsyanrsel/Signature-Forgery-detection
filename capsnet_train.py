import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Reshape, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = os.path.join('Dataset', 'Dataset1')
MODEL_PATH = os.path.join('models', 'capsnet.h5')


def build_capsnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), activation='relu')(inputs)
    x = Conv2D(32, (9, 9), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='validation'
)

caps_model = build_capsnet(input_shape=(224, 224, 1))
caps_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

caps_model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
caps_model.save(MODEL_PATH)

print(f"CapsuleNet model trained and saved to {MODEL_PATH}")