from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input

def build_siamese_model(input_shape=(224, 224, 1)):
    base_cnn = VGG16(include_top=False, input_shape=input_shape)
    base_cnn.trainable = False

    flatten = layers.Flatten()(base_cnn.output)
    dense = layers.Dense(256, activation='relu')(flatten)
    embedding = Model(base_cnn.input, dense)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    encoded_a = embedding(input_a)
    encoded_b = embedding(input_b)

    L1_layer = layers.Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_a, encoded_b])
    output = layers.Dense(1, activation='sigmoid')(L1_distance)

    siamese = Model(inputs=[input_a, input_b], outputs=output)
    return siamese