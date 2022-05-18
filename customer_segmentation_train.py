# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:39:21 2022

@author: Alfiqmal
"""

# =============================================================================
# TRAIN SCRIPT
# =============================================================================

#%% PACKAGES

import os 
import pickle
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import datetime
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.utils import plot_model

from customer_segmentation_modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation


import warnings
warnings.filterwarnings("ignore")


#%% PATHS

# =============================================================================
# ROBUST PATHS FOR DATASET, SAVED MODEL AND SAVED SCALER
# =============================================================================

TRAIN_DATA = os.path.join(os.getcwd(), "dataset", "train.csv")
MODEL_PATH = os.path.join(os.getcwd(), "saved", "model.h5") 
SCALER_PATH = os.path.join(os.getcwd(), "saved", "scaler.pkl")
ENCODER_PATH = os.path.join(os.getcwd(), "saved", "encoder.pkl")
OHE_PATH = os.path.join(os.getcwd(), "saved", "ohe.pkl")
LOG_PATH = os.path.join(os.getcwd(),'log')


#%% DATA LOADING 

df = pd.read_csv(TRAIN_DATA)

#%% DATA INSPECTION

df.head()

df.info()

# =============================================================================
# checking for any NaN value so that we can clean them up
# =============================================================================

df.isna().sum()


#%% DATA CLEANING

eda = ExploratoryDataAnalysis()

df_copy = df.copy()

# =============================================================================
# dropping ID because it is not a feature
# =============================================================================

df_copy.drop("ID", axis = 1, inplace = True)

# =============================================================================
# dropping NaN in "Ever_Married" because it wont take out a lot of data from df_copy
# =============================================================================

df_copy = df_copy[df_copy["Ever_Married"].notna()]
df_copy["Ever_Married"] = df_copy["Ever_Married"].astype("str")

# =============================================================================
# dropping NaN in "Graduated" because it wont take out a lot of data from df_copy
# =============================================================================

df_copy = df_copy[df_copy["Graduated"].notna()]

# =============================================================================
# filling NaN in "Profession" with "Unemployed"
# =============================================================================

df_copy["Profession"].fillna("Unemployed", inplace = True)
df_copy["Profession"] = df_copy["Profession"].astype("str")

# =============================================================================
# filling NaN with median for "Work_Experience" because there are big outlier
# =============================================================================

df_copy["Work_Experience"].fillna(math.floor(df_copy["Work_Experience"].
                                             median()), inplace = True)
# outlier not too big and far, can use mean

# =============================================================================
# filling NaN with mean for "Family Size" because the outlier is bearable
# =============================================================================
df_copy["Family_Size"].fillna(math.floor(df_copy["Family_Size"].
                                         mean()), inplace = True)

# =============================================================================
# dropping NaN in Var_1 because it wont tae out a lot of data from df_copy
# =============================================================================

df_copy = df_copy[df_copy["Var_1"].notna()]

# =============================================================================
# label encoding for all categorical columns
# =============================================================================

df_copy["Gender"] = eda.label_encoder(df_copy["Gender"])
df_copy["Ever_Married"] = eda.label_encoder(df_copy["Ever_Married"])
df_copy["Graduated"] = eda.label_encoder(df_copy["Graduated"])
df_copy["Profession"] = eda.label_encoder(df_copy["Profession"])
df_copy["Spending_Score"] = eda.label_encoder(df_copy["Spending_Score"])
df_copy["Var_1"] = eda.label_encoder(df_copy["Var_1"])


#%% DATA PREPROCESSING 

X = df_copy.drop(["Segmentation"], axis = 1)
y = df_copy[["Segmentation"]]

X = eda.min_max_scaler(X)

ohe = OneHotEncoder(sparse = False)

y = ohe.fit_transform(y)
pickle.dump(ohe, open(OHE_PATH, "wb"))

#%% DEEP LEARNING MODEL 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 13)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

model = ModelCreation().model_creation(input_shape = 9, output_shape = 4,
                                       num_nodes = 256, dropout = 0.3)

plot_model(model)

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

# Callbacks 

log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq = 1)

# early stopping callbacks 

early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 80)

# train

hist = model.fit(X_train, y_train, epochs = 200, 
                    validation_data = (X_test, y_test), 
                    callbacks = [tensorboard_callback, early_stopping_callback])

hist.history.keys() 
training_loss = hist.history['loss']
training_acc = hist.history['acc']
validation_loss = hist.history['val_loss']
validation_acc = hist.history['val_acc']

# Model Evaluation

predicted_y = np.empty([len(X_test), 4])

for index, test in enumerate(X_test):
    predicted_y[index,: ] = model.predict(np.expand_dims(test, axis = 0))
    
# MOdel Analysis

y_pred = np.argmax(predicted_y, axis=1)
y_true = np.argmax(y_test, axis=1)

ModelEvaluation().report_metrics(y_true, y_pred)

model.save(MODEL_PATH)


#%% Discussion


'''
As per requestion from our question file, we need to obtain more than 80% 
accuracy and F1 score for our deep learning model. Unfortunately, I only 
managed to train up until approximately max 52% accuracy. This is actually
heartbreaking for me because this is the final assessment for this course, 
but I am pretty sure I have done my very best. Therefore, I would like to 
discuss on how to rescue this model and get a better accuracy!


1. As Deep Learning is a data hungry method, we might want to add more data
and run our model on train to get a better accuracy. Even tho we already have 
thousands of data, but I believe, we need more than that in order to get an
even better accuracy with good F1 score.

2. By increasing the number of nodes, layers or even epochs might help by a
little amount, but it is worth a try.

3. I always believe that the key to a flawless model is always starts from
the data cleaning phase. I need to get better at cleaning the data, dealing
with NaN(s), missing values, null(s), using imputer intelligently, encode 
with care and standardize data when needed. I will pledge to myself to learn 
more for data cleaning. In fact, i have spent most of my time cleaning the 
data and do a lot of trial and error to obtain better accuracy and F1 score.
This assignment is a good exercise for me and in my assumption, for all of 
my batchmates as well.


To conclude everything, this is not the best one yet. I could've done better.
In the future, I hope i can do better by continuously learn :)
    
'''