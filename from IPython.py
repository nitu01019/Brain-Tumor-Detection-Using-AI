import numpy as np
import pandas as pd
import cv2
import shutil
import os
import matplotlib.pyplot as plt
import itertools
from keras.models import Sequential
from keras.layers import Flatten, BatchNormalization, Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19, preprocess_input
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from keras.preprocessing import image
import imutils

# Make directories
def create_dirs():
    directories = [
        "TRAIN", "TEST", "VAL", "TRAIN/YES", "TRAIN/NO", 
        "TEST/YES", "TEST/NO", "VAL/YES", "VAL/NO",
        "TRAIN_CROP", "TEST_CROP", "VAL_CROP", "TRAIN_CROP/YES", 
        "TRAIN_CROP/NO", "TEST_CROP/YES", "TEST_CROP/NO", 
        "VAL_CROP/YES", "VAL_CROP/NO"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

create_dirs()

# Define the load data function
def load_data(dir_path, img_size=(100, 100)):
    X, y = [], []
    i = 0
    labels = dict()
    for path in sorted(os.listdir(dir_path)):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(f'{dir_path}/{path}'):
                if not file.startswith('.'):
                    img = cv2.imread(f'{dir_path}/{path}/{file}')
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    y.append(i)
            i += 1
    return np.array(X), np.array(y), labels

# Loading the datasets
IMG_PATH = "../input/brain-tumor-detection-mri/Brain_Tumor_Detection"
TRAIN_DIR = 'TRAIN/'
TEST_DIR = 'TEST/'
VAL_DIR = 'VAL/'

X_train, y_train, labels = load_data(TRAIN_DIR, img_size=(224, 224))
X_test, y_test, _ = load_data(TEST_DIR, img_size=(224, 224))
X_val, y_val, _ = load_data(VAL_DIR, img_size=(224, 224))

# Preprocessing images
def preprocess_imgs(set_name, img_size=(224, 224)):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

X_train_prep = preprocess_imgs(X_train)
X_test_prep = preprocess_imgs(X_test)
X_val_prep = preprocess_imgs(X_val)

# Data augmentation setup
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'TRAIN_CROP/',
    color_mode='rgb',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'VAL_CROP/',
    color_mode='rgb',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Model setup using VGG19
base_Neural_Net = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

model = Sequential()
model.add(base_Neural_Net)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

for layer in base_Neural_Net.layers:
    layer.trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', 'AUC']
)

model.summary()

# Train the model
EPOCHS = 30
es = EarlyStopping(monitor='val_acc', mode='max', patience=6)

history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks=[es]
)

# Evaluation on training set
predictions = model.predict(X_train_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]
train_accuracy = accuracy_score(y_train, predictions)
print(f'Train Accuracy: {train_accuracy:.2f}')

# Confusion matrix
confusion_mtx = confusion_matrix(y_train, predictions)
plot_confusion_matrix(confusion_mtx, classes=['No', 'Yes'], normalize=False)

# Evaluation on validation set
predictions = model.predict(X_val_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]
val_accuracy = accuracy_score(y_val, predictions)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Confusion matrix
confusion_mtx = confusion_matrix(y_val, predictions)
plot_confusion_matrix(confusion_mtx, classes=['No', 'Yes'], normalize=False)

# Evaluate on the test set
predictions = model.predict(X_test_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]
test_accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Confusion matrix
confusion_mtx = confusion_matrix(y_test, predictions)
plot_confusion_matrix(confusion_mtx, classes=['No', 'Yes'], normalize=False)

# Print additional classification metrics
prob_pred = model.predict_proba(X_test_prep)
from sklearn import metrics
print('Accuracy:', metrics.accuracy_score(y_test, predictions))
print('Precision:', metrics.precision_score(y_test, predictions, average='weighted'))
print('Recall:', metrics.recall_score(y_test, predictions, average='weighted'))
print('F1 Score:', metrics.f1_score(y_test, predictions, average='weighted'))
print('ROC AUC Score:', metrics.roc_auc_score(y_test, prob_pred, average='weighted'))
print('Cohen Kappa Score:', metrics.cohen_kappa_score(y_test, predictions))
print('Classification Report:\n', metrics.classification_report(y_test, predictions))

# Clean up the directories after training
shutil.rmtree('TRAIN')
shutil.rmtree('TEST')
shutil.rmtree('VAL')
shutil.rmtree('TRAIN_CROP')
shutil.rmtree('TEST_CROP')
shutil.rmtree('VAL_CROP')
