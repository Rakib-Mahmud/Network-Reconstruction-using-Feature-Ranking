# -*- coding: utf-8 -*-

#Import All the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
from sklearn.ensemble import RandomForestRegressor
from skrebate import ReliefF
from skrebate.turf import TuRF
from sklearn_relief import RReliefF 
import xgboost as xgb

#Import Dataset
dataset = pd.read_csv('demo.csv',header=None)
nrows, nnodes = dataset.shape

#Generate Input Data and Target Data
X = dataset.iloc[0:(nrows-1),:].values
Y = dataset.iloc[1:(nrows),:].values

#Generate 15 CSV files as CSV can't have multiple sheets
rf = []
xgboost = []
rft = "Company(RF){}.csv"
xgboostt = "Company(XGB){}.csv"
#Define the destination csv files
for i in range(0,15):
    rf.append(rft.format(i))
    xgboost.append(xgboostt.format(i))
    
#Generate Feature Ranking Matrix
Fmatrix1 = np.zeros((15,nnodes,nnodes))#for 15 different years
Fmatrix2 = np.zeros((15,nnodes,nnodes))
Fmatrix3 = np.zeros((15,nnodes,nnodes))

working_days = 252
for itr in range(0,15):
    start = (itr*working_days)-1
    if start < 0:
        start = 0
        
    if itr == 6:
        working_days = 251
        
    end = working_days*(itr+1)
    
    #Prepare RandomForest model
    for node in range(0,nnodes):
      print("Calculating RF ranking for node ", node)
      print("And Round is ",itr)
      regressor = RandomForestRegressor(n_estimators = 300, max_features = 'log2', random_state = 0)
      regressor.fit(X[start:end,:],Y[start:end,node])
      Fmatrix1[itr,node,] = regressor.feature_importances_
    
    #Prepare ReliefF model
#    for node in range(0,nnodes):
#      print("Calculating ReliefF ranking for node ", node)
#      relief = RReliefF()
#    #  relief.fit(X,Y[:,node])
#    #  Fmatrix2[node,] = relief.feature_importances_
#      relief.fit_transform(X[start:end,:],Y[start:end,node])
#      Fmatrix2[itr,node,] = relief.w_
#    
    
    #Prepare XGBOOST model
    for node in range(0,nnodes):
      print("Calculating XGBoost ranking for node ", node)
      print("And Round is ",itr)
      xg_reg = xgb.XGBRegressor() 
      xg_reg.fit(X[start:end,:],Y[start:end,node])
      Fmatrix3[itr,node,] = xg_reg.feature_importances_



##Plot two feature ranking matrix generated using two different models
#Matrix1 for Randomforest
sn.heatmap(Fmatrix1[1,:,:], annot=False)
plt.show()

#Fmatrix2 for ReliefF
plt.figure()
sn.heatmap(Fmatrix2, annot=False)
plt.show()

#Fmatrix3 for XGBoost
plt.figure()
sn.heatmap(Fmatrix3[10,:,:], annot=False)
plt.show()

#Save the generated Feature Ranking Matrix
if os.path.isfile(rf and xgboost):
    os.remove(rf)
#    os.remove(relief)
    os.remove(xgboost)
for i in range(0,15):
    np.savetxt(rf[i], Fmatrix1[i,:,:], delimiter=",")
    #np.savetxt(relief, Fmatrix2, delimiter=",")
    np.savetxt(xgboost[i], Fmatrix3[i,:,:], delimiter=",")





#for node in range(0,nnodes):
#      print("Calculating RF ranking for node ", node)
##      print("And Round is ",itr)
#      regressor = RandomForestRegressor(n_estimators = 300, max_features = 'log2', random_state = 0)
#      regressor.fit(X,Y[:,node])
#      Fmatrix1[node,] = regressor.feature_importances_
#    
#sn.heatmap(Fmatrix1, annot=False)
#plt.show()
