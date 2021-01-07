import numpy as np
import pandas as pd
import os

#Import Dataset

dataset = pd.read_csv('FinalAllRawData.csv',header=None)
nrows, ncols = dataset.shape
X = dataset.iloc[:,:].values
#Define the destination csv files
log_r = "FinalLogReturn.csv"

#Generate Matrix to store log return
Fmatrix = np.zeros((nrows-1,ncols))

for i in range(1,nrows):
    for j in range(0,ncols):
        Fmatrix[i-1,j] = np.log(X[i,j])-np.log(X[i-1,j])

if os.path.isfile(log_r):
    os.remove(log_r)
np.savetxt(log_r, Fmatrix, delimiter=",")
