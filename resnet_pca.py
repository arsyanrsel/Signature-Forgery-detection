import os
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from resnet_svm import extract_features_with_resnet, train_svm
from sklearn.decomposition import PCA

MODEL_DIR = 'models'
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Dataset')
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_data = datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='training'
)

val_data = datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    subset='validation'
)

print("Training: ResNet + SVM")
resnet_features = extract_features_with_resnet(train_data)
resnet_labels = train_data.classes[:len(resnet_features)]

unique_labels, counts = np.unique(resnet_labels, return_counts=True)
print("SVM Labels:", dict(zip(unique_labels, counts)))

if len(unique_labels) < 2:
    print(" Not enough classes for SVM. Skipping training.")
else:
    pca = PCA(n_components=100)
    resnet_features_pca = pca.fit_transform(resnet_features)
    joblib.dump(pca, os.path.join(MODEL_DIR, 'resnet_pca.pkl'))

    svm_model = train_svm(resnet_features_pca, resnet_labels)
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'resnet_svm.pkl'))

    print(" ResNet + SVM (with PCA) model saved.")