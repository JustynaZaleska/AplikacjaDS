# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 05:42:09 2022


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import pickle

np.random.seed(10)

# %

df_raw = pd.read_csv('data.csv', index_col=0)
# print(df_raw)

#%%
#Budowanie modelu na podstawie wybranych atrybutów

columns= ['Year', 'Fuel_Type','Transmission', 'Engine','Power', 'Seats', 'Price']

df = df_raw.copy()
df = df[columns]
#print(df)

#przetworzenie column Engine i Power na kolumny numeryczne

df.Engine = df.Engine.str.split(' ').str[0]
df.Power = df.Power.str.split(' ').str[0].replace('null', np.nan)

#%%
#df.isnull().sum()  sprawdza czy są wiersze w których wystepuja braki

df = df.dropna()  # usuwa wiersze z brakami

df.Engine = df.Engine.astype('float32')
df.Power = df.Power.astype('float32')

#%%
#przekodowanie danych kategorycznych

df = pd.get_dummies(df, drop_first=True)

# %%
df.to_csv('./datasets/data_cleaned.csv')

#%% 
#przygotowanie danych do modelu

X= df.copy()
y = X.pop('Price')

#%%
# uzycie biblioteki sklearn from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)

#%%
# import alforytmu lasów losowych from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()  #tworzenie instancji modelu
reg.fit(X_train, y_train)  #metoda fit - przetrenowanie modelu na danych

print(reg.score(X_test, y_test))

#%%
# wykorzystanie klasy GridSearchCVdo znalezienia najlepszego modelu

param_grid = [{'max_depth': [3,4,5,6,7,8,10,20],
               'min_samples_leaf': [3,4,5,10,15]}]

model = RandomForestRegressor()
gs = GridSearchCV(model, param_grid=param_grid, scoring='r2')
gs.fit(X_train, y_train)

#%%
gs_score = gs.score(X_test, y_test)

#%%
model = gs.best_estimator_

with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)

print()