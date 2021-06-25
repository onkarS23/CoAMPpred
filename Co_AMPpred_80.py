#!/usr/bin/env python
# coding: utf-8

# Required packages

import numpy as np
import pandas as pd
import os, sys
from IPython.display import display
from pycaret.utils import version


# Getting the data
train = pd.read_csv('DB_80_Train.csv')
test = pd.read_csv('Test_data_all_AMB.csv')




# Setting up Environment in PyCaret
from pycaret.classification import *

clf1 = setup (data = train,
             target = 'Class', test_data = test, preprocess= True, feature_selection = True, feature_selection_threshold= 0.9, feature_selection_method= 'boruta', normalize = True, normalize_method = "robust", transformation = True)




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

# In[25]:


predict_model(tuned_gbc)


# In[39]:


plot_model(estimator = tuned_gbc, plot = 'feature',save = True, scale = 32)


# In[40]:


plot_model(estimator = tuned_gbc, plot = 'pr',save = True, scale = 32)


# In[19]:


et = create_model('et')


# In[19]:


print(et)


# In[29]:


plot_model(estimator = et)


# In[45]:


plot_model(estimator = et, plot = 'confusion_matrix')


# In[50]:


predict_model(et)


# In[20]:


tuned_et = tune_model(et)


# In[30]:


print(tuned_et)


# In[41]:


plot_model(estimator = tuned_et,save = True, scale = 32)


# In[42]:


plot_model(estimator = tuned_et, plot = 'confusion_matrix',save = True, scale = 32)


# In[24]:


predict_model(tuned_et)


# In[43]:


plot_model(estimator = tuned_et, plot = 'feature',save = True, scale = 32)


# In[44]:


plot_model(estimator = tuned_et, plot = 'pr',save = True, scale = 32)


# In[45]:


plot_model(estimator = tuned_et, plot = 'parameter',save = True, scale = 32)


# In[21]:


rf = create_model('rf')


# In[ ]:





# In[21]:


print(rf)


# In[32]:


plot_model(estimator = rf)


# In[43]:


plot_model(estimator = rf, plot = 'confusion_matrix')


# In[52]:


predict_model(rf)


# In[22]:


tuned_rf = tune_model(rf)


# In[46]:


print(tuned_rf)


# In[47]:


plot_model(estimator = tuned_rf, save = True, scale = 32)


# In[48]:


plot_model(estimator = tuned_rf, plot = 'confusion_matrix',save = True, scale = 32)


# In[49]:


predict_model(tuned_rf)


# In[50]:


plot_model(estimator = tuned_rf, plot = 'feature', save = True, scale = 32)


# In[51]:


plot_model(estimator = tuned_rf, plot = 'pr', save = True, scale = 32)


# In[52]:


plot_model(estimator = tuned_rf, plot = 'parameter', save = True, scale = 32)


# In[64]:


catboost = create_model('catboost')


# In[25]:


print(catboost)


# In[36]:


plot_model(estimator = catboost)


# In[38]:


plot_model(estimator = catboost, plot = 'confusion_matrix' )


# In[56]:


predict_model(catboost)


# In[8]:


tuned_catboost = tune_model(catboost)


# In[26]:


print(tuned_catboost)


# In[53]:


plot_model(estimator = tuned_catboost, save = True, scale = 32)


# In[54]:


plot_model(estimator = tuned_catboost, plot = 'confusion_matrix', save = True, scale = 32)


# In[27]:


predict_model(tuned_catboost)


# In[55]:


plot_model(estimator = tuned_catboost, plot = 'feature', save = True, scale = 32)


# In[56]:


plot_model(estimator = tuned_catboost, plot = 'pr', save = True, scale = 32)


# In[ ]:




