#!/usr/bin/env python
# coding: utf-8

# Required packages


import numpy as np
import pandas as pd
import os, sys
from IPython.display import display
from pycaret.utils import version


# Getting the data

train = pd.read_csv('db70_train.csv')
test = pd.read_csv('Test_data_all_AMB.csv')

# Setting up Environment in PyCaret


from pycaret.classification import *

clf1 = setup (data = train,
             target = 'Class', test_data = test, preprocess= True, feature_selection = True, feature_selection_threshold= 0.9, feature_selection_method= 'boruta', normalize = True, normalize_method = "robust", transformation = True, log_plots = True)

#Comparing All Models

best = compare_models()

#Create a Model 'lightgbm'

lightgbm = create_model('lightgbm')

#trained model object is stored in the variable:"lightgbm"

print(lightgbm)

#Plot a Model
plot_model(estimator = lightgbm)#AUC
plot_model(estimator = lightgbm, plot = 'confusion_matrix')
plot_model(estimator = lightgbm, plot = 'feature')
#Predict on test / hold-out Sample
predict_model(lightgbm)


#Create a Model 'gbc'
gbc = create_model('gbc')
#trained model object is stored in the variable:"gbc"
print(gbc)
#Plot a Model
plot_model(estimator = gbc)
plot_model(estimator = gbc, plot = 'confusion_matrix')
plot_model(estimator = gbc, plot = 'feature')
#Predict on test / hold-out Sample
predict_model(gbc)


plot_model(estimator = tuned_gbc) #AUC
plot_model(estimator = tuned_gbc, plot = 'confusion_matrix')
plot_model(estimator = tuned_gbc, plot = 'feature')


#Create a Model
et = create_model('et')
#trained model object is stored in the variable:"et"
print(et)

# Plot Model
plot_model(estimator = et)
plot_model(estimator = et, plot = 'confusion_matrix')
plot_model(estimator = tuned_et, plot = 'feature')

#Predict on test / hold-out Sample
predict_model(et)

#Create a Model

rf = create_model('rf')

# trained model object is stored in the variable:"rf"

print(rf)

# Plot Model 
plot_model(estimator = rf) #AUC
plot_model(estimator = rf, plot = 'confusion_matrix')
plot_model(estimator =rf, plot = 'feature')

#Predict on test / hold-out Sample
predict_model(rf)


#Create a Model
catboost = create_model('catboost')

# trained model object is stored in the variable:"catboost"

print(catboost)

# Plot Model 

plot_model(estimator = catboost)
plot_model(estimator = catboost, plot = 'confusion_matrix')
plot_model(estimator = catboost, plot = 'feature')


#Predict on test / hold-out Sample
predict_model(catboost)




