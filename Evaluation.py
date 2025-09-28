import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from resnet_svm import extract_features_with_resnet
from siamese_vgg import build_siamese_model


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dataset'))


test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)
y_true = test_data.classes

def print_metrics(y_true, y_pred, name):
    print(f"\n{name} Results:")
    print("-" * 40)
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_true, y_pred, zero_division=0))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

results = {}

cnn = load_model(os.path.join(MODEL_DIR, 'custom_cnn.h5'), compile=False)
cnn_preds = (cnn.predict(test_data) > 0.5).astype(int).flatten()
print_metrics(y_true, cnn_preds, "Custom CNN")
results["Custom CNN"] = accuracy_score(y_true, cnn_preds)

auto = load_model(os.path.join(MODEL_DIR, 'autoencoder.h5'), compile=False)
x_test, _ = test_data[0]
recon = auto.predict(x_test)
errors = np.mean((x_test.reshape(x_test.shape[0], -1) - recon.reshape(recon.shape[0], -1))**2, axis=1)
threshold = np.percentile(errors, 90)
auto_preds = (errors > threshold).astype(int)
print_metrics(y_true[:len(auto_preds)], auto_preds, "Autoencoder")
results["Autoencoder"] = accuracy_score(y_true[:len(auto_preds)], auto_preds)

capsnet = load_model(os.path.join(MODEL_DIR, 'capsnet.h5'), compile=False)
caps_preds = (capsnet.predict(test_data) > 0.5).astype(int).flatten()
print_metrics(y_true, caps_preds, "CapsuleNet")
results["CapsuleNet"] = accuracy_score(y_true, caps_preds)

resnet_features = extract_features_with_resnet(test_data)
pca = joblib.load(os.path.join(MODEL_DIR, 'resnet_pca.pkl'))
resnet_features_pca = pca.transform(resnet_features)
svm_model = joblib.load(os.path.join(MODEL_DIR, 'resnet_svm.pkl'))
svm_preds = svm_model.predict(resnet_features_pca)
print_metrics(y_true[:len(svm_preds)], svm_preds, "ResNet + SVM")
results["ResNet + SVM"] = accuracy_score(y_true[:len(svm_preds)], svm_preds)

siamese = build_siamese_model(input_shape=(224, 224, 3))
siamese.load_weights(os.path.join(MODEL_DIR, 'siamese_vgg.h5'))

ref_path = os.path.join(DATA_DIR, 'test', 'genuine')
ref_img = load_img(os.path.join(ref_path, os.listdir(ref_path)[0]), color_mode='grayscale', target_size=IMG_SIZE)
ref_arr = img_to_array(ref_img) / 255.0
ref_arr = np.repeat(ref_arr, 3, axis=-1)[np.newaxis, ...]

comparison_imgs = []
for path in test_data.filepaths:
    img = load_img(path, color_mode='grayscale', target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.repeat(arr, 3, axis=-1)
    comparison_imgs.append(arr)

comparison_imgs = np.array(comparison_imgs)
siamese_preds = siamese.predict([np.repeat(ref_arr, len(comparison_imgs), axis=0), comparison_imgs])
siamese_bin = (siamese_preds.flatten() < 0.5).astype(int)

print_metrics(y_true[:len(siamese_bin)], siamese_bin, "Siamese VGG")
results["Siamese VGG"] = accuracy_score(y_true[:len(siamese_bin)], siamese_bin)

plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison (All 5 Models)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()