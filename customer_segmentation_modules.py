# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:41:37 2022

@author: Alfiqmal
"""


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


#%%

class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass
    
    def label_encoder(self, data):        
        lab_enc = LabelEncoder() 
        return lab_enc.fit_transform(data)
    
    def one_hot_encoder(self, data):    
        ohe = OneHotEncoder(sparse=False) 
        return ohe.fit_transform(data)
    
    def min_max_scaler(self, data):        
        mms = MinMaxScaler() 
        return mms.fit_transform(data)
    
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def model_creation(self, input_shape, output_shape, num_nodes, dropout):
        model = Sequential()
        model.add(Input(shape=(input_shape), name='input_layer'))
        model.add(Flatten())
        model.add(Dense(num_nodes, activation='relu', name='hidden_layer_1'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(num_nodes, activation='relu', name='hidden_layer_2'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(num_nodes, activation='relu', name='hidden_layer_3'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(num_nodes, activation='relu', name='hidden_layer_4'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(output_shape, activation='softmax', name='output_layer'))
        model.summary()
        return model
    
class ModelEvaluation():
    
    def report_metrics(self,y_true,y_pred):
       
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))