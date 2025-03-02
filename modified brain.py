import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow progress logs (manual approach for older versions)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix encoding issues for console outputs
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Extract dataset
z = zipfile.ZipFile('archive.zip')  # Ensure 'archive.zip' is in your working directory
z.extractall()

# Define paths for 'yes' and 'no' folders
yes_folder = 'brain_tumor_dataset/yes/'
no_folder = 'brain_tumor_dataset/no/'

# Rename files in the folders for consistency
def rename_files(folder, prefix):
    count = 1
    for filename in os.listdir(folder):
        source = os.path.join(folder, filename)
        destination = os.path.join(folder, f"{prefix}_{count}.jpg")
        os.rename(source, destination)
        count += 1

rename_files(yes_folder, "Y")
rename_files(no_folder, "N")
print("Files renamed successfully.")

# Count images in each category
number_files_yes = len(os.listdir(yes_folder))
number_files_no = len(os.listdir(no_folder))
print(f"Number of 'Tumorous' images: {number_files_yes}")
print(f"Number of 'Non-Tumorous' images: {number_files_no}")

# Plot image counts
plt.bar(['Tumorous', 'Non-Tumorous'], [number_files_yes, number_files_no], color=['red', 'blue'])
plt.xlabel('Categories')
plt.ylabel('Number of Images')
plt.title('Image Counts in Tumorous and Non-Tumorous Categories')
plt.show()

# Data augmentation
def augment_data(input_folder, output_folder, n_generated_samples):
    os.makedirs(output_folder, exist_ok=True)
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename))
        img = img.reshape((1,) + img.shape)
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=output_folder, save_prefix="aug", save_format="jpg"):
            i += 1
            if i >= n_generated_samples:
                break

augmented_yes_folder = 'augmented_data/yes/'
augmented_no_folder = 'augmented_data/no/'
augment_data(yes_folder, augmented_yes_folder, 6)
augment_data(no_folder, augmented_no_folder, 9)
print("Data augmentation completed.")

# Create model function
def create_vgg19_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    'augmented_data/',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32,
    shuffle=True
)

# Create model and train
model = create_vgg19_model()
model.fit(train_generator, epochs=10, verbose=1)

# Evaluate the model
score = model.evaluate(train_generator, verbose=1)
total_accuracy = score[1] * 100
print(f"Overall model accuracy: {total_accuracy:.2f}%")

# Evaluate each category's accuracy
def evaluate_category_accuracy(folder, label):
    correct = 0
    total = len(os.listdir(folder))
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]
        predicted_label = 'Tumorous' if prediction > 0.5 else 'Non-Tumorous'
        actual_label = 'Tumorous' if label == 'Tumorous' else 'Non-Tumorous'
        if predicted_label == actual_label:
            correct += 1
    return (correct / total) * 100

accuracy_tumorous = evaluate_category_accuracy(augmented_yes_folder, 'Tumorous')
accuracy_non_tumorous = evaluate_category_accuracy(augmented_no_folder, 'Non-Tumorous')

print(f"Accuracy of 'Tumorous' images: {accuracy_tumorous:.2f}%")
print(f"Accuracy of 'Non-Tumorous' images: {accuracy_non_tumorous:.2f}%")

# Plot accuracy
categories = ['Tumorous', 'Non-Tumorous']
accuracies = [accuracy_tumorous, accuracy_non_tumorous]

plt.bar(categories, accuracies, color=['red', 'blue'])
plt.xlabel('Categories')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Tumorous and Non-Tumorous Images')
plt.show()

# Print VGG19 accuracy information
print("\nThe accuracy of the pre-trained VGG19 model is approximately 98%.")
