# 🌾 Rice Leaf Disease Detection using Transfer Learning

This project focuses on detecting diseases in rice leaves using Deep Learning techniques. It compares custom CNN architectures with state-of-the-art Transfer Learning models to achieve high accuracy.

## 🚀 Project Overview
Rice is a staple food for a significant part of the world's population. Early detection of diseases like **Blast, Brownspot, Tungro, and Bacterial Blight** is crucial for crop yield. This project utilizes Computer Vision to classify these diseases automatically.

### ✨ Key Features
* **Data Preprocessing:** Image normalization, resizing (224x224), and splitting (Train/Val/Test).
* **Custom CNN:** A manually designed Convolutional Neural Network as a baseline.
* **Hyperparameter Tuning:** Optimized using **Keras Tuner** (RandomSearch).
* **Transfer Learning:** Comparative analysis of **VGG16, ResNet50, InceptionV3, and MobileNetV2**.
* **Deployment:** A web interface built with **Streamlit** and served via **Ngrok**.

## 📊 Model Performance
After extensive training and testing, **MobileNetV2** was selected as the best performing model due to its balance between accuracy and inference speed.

| Model | Accuracy | MCC | F1-Score |
| :--- | :---: | :---: | :---: |
| **MobileNetV2** | **98%** | **0.97** | **0.98** |
| ResNet50 | 94% | 0.92 | 0.94 |
| Custom CNN | 85% | 0.81 | 0.84 |

*(Note: Values are approximate based on the best training run.)*

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **Optimization:** Keras Tuner
* **Web Framework:** Streamlit, PyNgrok
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab

## 📂 Project Structure
* `Rice_Disease_Training.ipynb`: The main notebook containing data loading, preprocessing, model training (Custom & Transfer Learning), and evaluation.
* `Streamlit_App_Deployment.ipynb`: The code for running the web application using Streamlit and Ngrok.
* `rice_disease_mobilenet_final.h5`: The trained MobileNetV2 model weights.

## 🖼️ Dataset
The dataset consists of images categorized into 4 classes:
1.  Bacterial Blight
2.  Blast
3.  Brown Spot
4.  Tungro

## 🤝 How to Run
1.  Open `Rice_Disease_Training.ipynb` in Google Colab.
2.  Mount your Google Drive containing the dataset.
3.  Run the cells to train the model or load the pre-trained `.h5` file.
4.  To launch the web app, run `Streamlit_App_Deployment.ipynb`.

---
*Created by efecandonmez67
