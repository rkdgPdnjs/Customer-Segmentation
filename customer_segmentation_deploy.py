# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:19:29 2022

@author: Alfiqmal
"""

import os 
import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import load_model
import pickle
import warnings
warnings.filterwarnings("ignore")
from customer_segmentation_modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation


#%%

TEST_DATA = os.path.join(os.getcwd(), "dataset", "new_customers.csv")
MODEL_PATH = os.path.join(os.getcwd(), "saved", "model.h5") 
SCALER_PATH = os.path.join(os.getcwd(), "saved", "scaler.pkl")
ENCODER_PATH = os.path.join(os.getcwd(), "saved", "encoder.pkl")
OHE_PATH = os.path.join(os.getcwd(), "saved", "ohe.pkl")
LOG_PATH = os.path.join(os.getcwd(),'log')
PREDICTED_PATH = os.path.join(os.getcwd(), "dataset", "new_customers_prediction.csv")

#%% MODEL LOADING

model = load_model(MODEL_PATH)

model.summary()

scaler_saved = pickle.load(open(SCALER_PATH, "rb"))
encoder_saved = pickle.load(open(ENCODER_PATH, "rb"))
ohe_saved = pickle.load(open(OHE_PATH, "rb"))

#%% DATA LOADING 

df = pd.read_csv(TEST_DATA)

#%% DATA CLEANING

eda = ExploratoryDataAnalysis()

df_copy = df.copy()

df_copy.drop("ID", axis = 1, inplace = True)

df_copy = df_copy[df_copy["Ever_Married"].notna()]
df_copy["Ever_Married"] = df_copy["Ever_Married"].astype("str")

df_copy = df_copy[df_copy["Graduated"].notna()]

df_copy["Profession"].fillna("Unemployed", inplace = True)
df_copy["Profession"] = df_copy["Profession"].astype("str")


df_copy["Work_Experience"].fillna(math.floor(df_copy["Work_Experience"].median()), inplace = True)
# outlier not too big and far, can use mean

df_copy["Family_Size"].fillna(math.floor(df_copy["Family_Size"].mean()), inplace = True)

df_copy = df_copy[df_copy["Var_1"].notna()]


df_copy["Gender"] = eda.label_encoder(df_copy["Gender"])
df_copy["Ever_Married"] = eda.label_encoder(df_copy["Ever_Married"])
df_copy["Graduated"] = eda.label_encoder(df_copy["Graduated"])
df_copy["Profession"] = eda.label_encoder(df_copy["Profession"])
df_copy["Spending_Score"] = eda.label_encoder(df_copy["Spending_Score"])
df_copy["Var_1"] = eda.label_encoder(df_copy["Var_1"])


#%% DATA PREPROCESSING 

X = df_copy.drop(["Segmentation"], axis = 1)

X = eda.min_max_scaler(X)

#%% 

predicted_y = np.empty([len(X), 4])

for index, test in enumerate(X):
    predicted_y[index, :] = model.predict(np.expand_dims(test, axis = 0))

y_pred = np.argmax(predicted_y, axis=1)

y_pred = y_pred.astype("str")

new_y_pred = ohe_saved.inverse_transform(predicted_y)

new_y_pred = pd.DataFrame(new_y_pred)

#%%

df_new = df.copy()

df_new = df_new[df_new["Ever_Married"].notna()]
df_new["Ever_Married"] = df_new["Ever_Married"].astype("str")

df_new = df_new[df_new["Graduated"].notna()]

df_new["Profession"].fillna("Unemployed", inplace = True)
df_new["Profession"] = df_new["Profession"].astype("str")


df_new["Work_Experience"].fillna(math.floor(df_new["Work_Experience"].median()), inplace = True)
# outlier not too big and far, can use mean

df_new["Family_Size"].fillna(math.floor(df_new["Family_Size"].mean()), inplace = True)

df_new = df_new[df_new["Var_1"].notna()]

df_new.drop("Segmentation", axis = 1, inplace = True)

df_new = pd.concat([df_new, new_y_pred], axis = 1, join = "inner")

df_new.to_csv(PREDICTED_PATH, index = False)
