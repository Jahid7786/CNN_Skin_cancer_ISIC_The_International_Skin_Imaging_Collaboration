# Introduction
Skin cancer is the most common type of cancer. It is the growth of abnormal skin cells on skin exposed to the sun. This type of cancer also can also occur in areas not exposed to sunlight. The most common places for skin cancer to develop are on areas of skin that are exposed to the sun, including the scalp, face, lips, ears, neck, chest, arms and hands, and on the legs in women. However, it can also form in areas that are rarely seen in the light of day: palms, under fingernails or toenails, and areas of the genitals. Currently, deep learning has revolutionised the future as it can solve complex problems. The motivation is to develop a solution that can help dermatologists better support their diagnostic accuracy by ensembling contextual images , reducing the variance of predictions from the model.

# Problem
Early detection is critical for successful treatment, but accurately identifying the type of skin cancer can be challenging. That's where skin classification using CNN multiclass classification comes in. By training a neural network on a large dataset of skin cancer images, we can develop a highly accurate classification system that can differentiate between the 7 different types of skin cancer.

This approach solves several key problems in skin cancer detection. First, it eliminates the need for a human expert to visually inspect every skin lesion, which can be time-consuming and prone to error. Second, it can help identify subtle differences between different types of skin cancer that might not be apparent to the naked eye. And third, it can help ensure that patients receive the correct diagnosis and treatment plan, improving their chances of a positive outcome.

Overall, skin classification using CNN multiclass classification is a powerful tool for improving the accuracy and efficiency of skin cancer detection. By leveraging the power of machine learning, we can help healthcare providers make more informed decisions and ultimately improve patient outcomes.

# Motivation
The primary goal of the project is to develop a highly accurate skin cancer classification model using CNN multiclass classification. By training the neural network on a large dataset of skin cancer images, the model aims to differentiate between all seven types of skin cancer with high precision and accuracy.

The project's focus on accuracy and validation based on images is essential in ensuring that healthcare providers have access to reliable and accurate skin cancer diagnosis tools. The model's ability to accurately classify skin cancer types can aid in identifying the disease in its early stages, leading to more effective treatment and improved patient outcomes.

Overall, the project's focus on developing an accurate skin cancer classification model using advanced image classification technology is a significant step towards improving skin cancer detection and diagnosis, leading to earlier treatment and better patient outcomes.

# Dataset
The HAM10000 dataset is commonly used in skin cancer classification tasks due to its extensive collection of 10,015 dermatoscopic images of pigmented skin lesions, which are classified into seven different types. The dataset provides a representative range of the major diagnostic categories of skin lesions, including Actinic keratoses and intraepithelial carcinoma/Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions such as solar lentigines/seborrheic keratoses and lichen-planus like keratosis (bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions such as angiomas, angiokeratomas, pyogenic granulomas, and haemorrhage, vasc.

## Conclusion:
By using this dataset in skin cancer classification tasks, researchers and practitioners can train and test models to accurately differentiate between the various types of skin cancer. The diversity of the dataset allows for a more comprehensive understanding of skin lesions and better support for dermatological clinical work.

## Technology Used:
import pathlib
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

from tensorflow.keras.preprocessing.image import load_img

