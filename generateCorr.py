# -*- coding: utf-8 -*-

#Import All the packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectFromModel
import seaborn as sn
import os
from sklearn.ensemble import RandomForestRegressor
from skrebate import ReliefF

#Import Dataset

dataset = pd.read_csv('AllData.csv',header=None)
nrows, nnodes = dataset.shape

#Generate Input Data and Target Data

X = dataset.iloc[0:(nrows-1),:].values
Y = dataset.iloc[1:(nrows),:].values

#Define the destination csv files

rf = "CorrMatrix(RF).csv"
relief = "CorrMatrix(Relief).csv"

#Generate Feature Ranking Matrix

Fmatrix1 = np.zeros((nnodes,nnodes))
Fmatrix2 = np.zeros((nnodes,nnodes))

#Prepare RandomForest model
for node in range(0,nnodes):
  print("Calculating RF ranking for node ", node)
  regressor = RandomForestRegressor(n_estimators = 1000, max_features = 'sqrt', random_state = 0)
  #rfmodel = randomForest(f, data = ds, ntree = 1000, mtry = floor(sqrt(nnodes)), importance = TRUE)
  regressor.fit(X,Y[:,node])
  Fmatrix1[node,] = regressor.feature_importances_
#  print (regressor.feature_importances_)
pr = (regressor.feature_importances_)  
p=Y[:,0]

#Prepare ReliefF model

for node in range(0,nnodes):
  print("Calculating ReliefF ranking for node ", node)
  relief = ReliefF()
  #rfmodel = randomForest(f, data = ds, ntree = 1000, mtry = floor(sqrt(nnodes)), importance = TRUE)
  relief.fit(X,Y[:,node])
  Fmatrix2[node,] = relief.feature_importances_
#  Fmatrix[node,] = regressor.feature_importances_
relief.top_features

#Plot two feature ranking matrix generated using two different models
#Matrix1 for Randomforest
sn.heatmap(Fmatrix1, annot=False)
plt.show()

#Fmatrix2 for ReliefF
plt.figure()
sn.heatmap(Fmatrix2, annot=False)
plt.show()

#Save the generated Feature Ranking Matrix

if os.path.isfile(rf and relief):
    os.remove(rf)
    os.remove(relief)
np.savetxt(rf, Fmatrix1, delimiter=",")
np.savetxt(relief, Fmatrix2, delimiter=",")
