#Import All the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
plt.style.use('fivethirtyeight')
#retrive datasets
dataset = []
for x in range(0,15):
    dataset.append(pd.read_csv('Company(XGB){}.csv'.format(x),header=None))

#Find mean of the data of each years
years = [x for x in range(1998,2013)]
avg = []

for y in range(0,15):
    temp = 0
    temp=(dataset[y].median(axis=0))
    temp = temp.tolist()
    avg.append(statistics.mean(temp))


ymin = np.min(avg)
ymax = np.max(avg)
plt.figure(figsize=(9, 3))
plt.ylim(ymin-1e-5,ymax+1e-5)
plt.title('Year-wise mean ranking(XGB)')
plt.xlabel('Year')
plt.ylabel('Mean')
plt.bar(years,avg,width=0.5)
plt.show()# -*- coding: utf-8 -*-

