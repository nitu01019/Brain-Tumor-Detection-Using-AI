# Brain-Tumor-Detection-Using-AI
# Brain Tumor Detection Using AI

## Introduction

Hey there! 👋 Welcome to this **Brain Tumor Detection** project, where we use **Artificial Intelligence (AI)** to analyze MRI scans and detect tumors. This project leverages **VGG19**, a powerful deep learning model, to classify images as either **tumorous** or **non-tumorous**—helping in early diagnosis. 🧠

## What This Project Does

- **Loads MRI images** and prepares them for analysis.
- **Uses AI (VGG19)** to learn patterns and classify images.
- **Enhances data** through augmentation (rotating, flipping, zooming, etc.).
- **Evaluates performance** with accuracy, confusion matrices, and other metrics.
- **Provides insights** through graphs and reports. 📊

---

## Getting Started

### What You Need 🛠️

Make sure you have the following installed:

- **Python 3.x**
- **TensorFlow & Keras** (for deep learning)
- **OpenCV** (for image processing)
- **NumPy & Pandas** (for data handling)
- **Matplotlib & Seaborn** (for visualizations)

### Install Everything 📦

Run this command to install the required libraries:

```sh
pip install tensorflow keras numpy pandas opencv-python matplotlib seaborn
```

### Dataset 📁

1. Download the **Brain Tumor Dataset** and place it in your working directory.
2. Ensure the dataset is structured as follows:

```
brain_tumor_dataset/
   ├── yes/  # Contains MRI images with tumors
   ├── no/   # Contains MRI images without tumors
```

3. If your dataset is in a ZIP file, extract it before running the code.

---

## How It Works 🚀

### **Step 1: Preparing the Images**

- Resizes all MRI images to **224x224 pixels**.
- Applies **data augmentation** to improve learning.
- Splits images into **training, validation, and test** sets.

### **Step 2: Training the AI Model**

- Loads **VGG19**, a powerful pre-trained model.
- Adds **custom layers** for classification.
- Uses **Adam optimizer** and **binary cross-entropy** for learning.
- Trains for **10-30 epochs** (fine-tuned for performance).

### **Step 3: Making Predictions & Evaluating Performance**

- Predicts if an MRI scan has a tumor or not. 🤖
- Computes **accuracy, precision, recall, and F1-score**.
- Generates visual reports to analyze results.

---

## Running the Project 🏃‍♂️

### 1️⃣ Extract the Dataset (If Zipped)

```sh
unzip archive.zip
```

### 2️⃣ Run the Main Python Script

```sh
python brain_tumor_detection.py
```

### 3️⃣ Check the Results 🎯

- The script will print accuracy scores.
- Confusion matrices and graphs will be displayed.

---

## Model Performance 🏆

- **Training Accuracy:** \~98%
- **Validation Accuracy:** \~96%
- **Test Accuracy:** \~95%

### Confusion Matrix Breakdown:

✅ **True Positives:** Correctly identified tumors.\
✅ **True Negatives:** Correctly identified non-tumors.\
❌ **False Positives:** Misclassified normal scans as tumors.\
❌ **False Negatives:** Missed tumors.

---

## Future Improvements 🔮

- **Try more AI models** like **ResNet** or **EfficientNet**.
- **Improve data augmentation** for better accuracy.
- **Optimize hyperparameters** to reduce false detections.
- **Develop a user-friendly interface** for real-time scanning.

---

## Meet the Creator 👩‍💻

Developed  by **[NITISH BHARDWAJ]**. If you have any questions, feel free to reach out! 😊

---

## License 📜

This project is for educational and research purposes. Feel free to modify and improve it!

