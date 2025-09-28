import os
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA

from custom_cnn import build_custom_cnn
from autoencoder import build_autoencoder
from siamese_vgg import build_siamese_model
from resnet_svm import extract_features_with_resnet, train_svm
from capsnet import build_capsnet
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Dataset')
DATA_DIR = os.path.abspath(DATA_DIR)
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.02,
    zoom_range=0.02,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
)

val_data = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
)

print("Training: Custom CNN")
custom_model = build_custom_cnn(input_shape=(224, 224, 1))
custom_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
custom_model.fit(train_data, validation_data=val_data, epochs=5)
custom_model.save(os.path.join(MODEL_DIR, 'custom_cnn.h5'))

print("Training: Autoencoder")
autoencoder = build_autoencoder(input_shape=(224, 224, 1))
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(train_data, validation_data=val_data, epochs=5)
autoencoder.save(os.path.join(MODEL_DIR, 'autoencoder.h5'))

print("Training: ResNet + SVM")
resnet_features = extract_features_with_resnet(train_data)
resnet_labels = train_data.classes[:len(resnet_features)]
unique_labels, counts = np.unique(resnet_labels, return_counts=True)
print("SVM Labels:", dict(zip(unique_labels, counts)))
if len(unique_labels) < 2:
    print(" Not enough classes for SVM. Skipping training.")
else:
    pca = PCA(n_components=100)
    resnet_features = pca.fit_transform(resnet_features)
    svm_model = train_svm(resnet_features, resnet_labels)
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'resnet_svm.pkl'))
    print(" ResNet + SVM model saved.")

print("Training: Capsule Network")
caps_model = build_capsnet(input_shape=(224, 224, 1))
caps_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
caps_model.fit(train_data, validation_data=val_data, epochs=5)
caps_model.save(os.path.join(MODEL_DIR, 'capsnet.h5'))

print("Training: Siamese Network")

class SiamesePairGenerator:
    def __init__(self, data_dir, batch_size, img_size):
        self.batch_size = batch_size
        self.img_size = img_size
        self.genuine_dir = os.path.join(data_dir, 'genuine')
        self.forge_dir = os.path.join(data_dir, 'forge')
        self.genuine_imgs = os.listdir(self.genuine_dir)
        self.forge_imgs = os.listdir(self.forge_dir)

    def __iter__(self):
        return self

    def __next__(self):
        pairs, labels = [], []
        for _ in range(self.batch_size):
            if random.random() < 0.5:
                label = 1
                img1, img2 = random.sample(self.genuine_imgs, 2)
                img1_path = os.path.join(self.genuine_dir, img1)
                img2_path = os.path.join(self.genuine_dir, img2)
            else:
                label = 0
                img1 = random.choice(self.genuine_imgs)
                img2 = random.choice(self.forge_imgs)
                img1_path = os.path.join(self.genuine_dir, img1)
                img2_path = os.path.join(self.forge_dir, img2)

            arr1 = self._load_image(img1_path)
            arr2 = self._load_image(img2_path)
            pairs.append((arr1, arr2))
            labels.append(label)

        pair1 = np.array([x[0] for x in pairs])
        pair2 = np.array([x[1] for x in pairs])
        return (pair1, pair2), np.array(labels).astype('float32')

    def _load_image(self, path):
        img = load_img(path, color_mode='grayscale', target_size=self.img_size)
        arr = img_to_array(img) / 255.0
        arr = np.repeat(arr, 3, axis=-1)
        return arr

siamese_gen = SiamesePairGenerator(DATA_DIR, batch_size=16, img_size=IMG_SIZE)
siamese_model = build_siamese_model(input_shape=(224, 224, 3))
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

for i in range(5):
    print(f"Epoch {i+1}/5")
    (pair1, pair2), y = next(siamese_gen)
    siamese_model.fit([pair1, pair2], y, epochs=1, batch_size=16)

siamese_model.save(os.path.join(MODEL_DIR, 'siamese_vgg.h5'))
print("All models trained and saved.")