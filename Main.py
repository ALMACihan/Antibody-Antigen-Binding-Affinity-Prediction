import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import requests
from numpy import mean, std
from pandas import Series
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ast import literal_eval
from tdc.multi_pred import AntibodyAff
from xgboost import XGBRFClassifier, train, XGBRFRegressor
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import tensorflow as tf
from sklearn.model_selection import KFold

data = AntibodyAff(name = 'Protein_SAbDab')
split = data.get_split(method = 'random')

df = pd.DataFrame(data.get_data())

df['Antibody'] = df['Antibody'].apply(literal_eval) #convert to list type
antibodyExploded = df['Antibody'].explode()

print(antibodyExploded.count())

df_clean = antibodyExploded.to_frame()

antigen_repeat = pd.Series(df['Antigen'])
antigen_repeat = antigen_repeat.repeat(2)

Y_repeat = pd.Series(df['Y'])
Y_repeat = Y_repeat.repeat(2)

df_clean['Antigen'] = antigen_repeat
df_clean['Y'] = Y_repeat

df_clean.reset_index(drop=True, inplace=True)
print(df_clean)

antibody = df_clean['Antibody'].apply(lambda x:pd.Series(list(x)))
antigen = df_clean['Antigen'].apply(lambda x:pd.Series(list(x)))

antibody = pd.get_dummies(antibody)
antigen = pd.get_dummies(antigen)

for i in antibody.columns:
    antibody = antibody.rename(index=str, columns={i: 'antibody_'+i})

antibody = antibody.reset_index(drop=True)
antigen = antigen.reset_index(drop=True)
Y_repeat = Y_repeat.reset_index(drop=True)

data_frame = pd.concat([antibody, antigen, Y_repeat], axis=1, ignore_index=True)
final_frame = data_frame.reset_index(drop=True)

#final_frame.to_csv('final_frame.csv', index=False)

print(final_frame)

kfold = KFold(n_splits=10, shuffle=True)

fold_no = 1
for train, test in kfold.split(final_frame):

  print(train.shape)
  print(test.shape)
  train = final_frame.iloc[:, 0:16948].values
  test = final_frame.iloc[:, 16948].values


  count_true = 0
  count_false = 0

  X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=0)

  train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
  test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

  y_test = tf.cast(y_test, tf.float32)
  y_train = tf.cast(y_train, tf.float32)
  train_X = tf.cast(train_X, tf.float32)
  test_X = tf.cast(test_X, tf.float32)

  model = Sequential()
  model.add(Bidirectional(LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2]))))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='mse', optimizer='adam')

  history = model.fit(train_X, y_train, epochs=16, batch_size=32, validation_data=(test_X, y_test),  shuffle=False)

  yhat = model.predict(test_X)

  for i in range(len(y_test)):
    if y_test[0] < 1*10^(-6) and yhat[0] < 1*10^(-6):
      count_true += 1
    elif y_test[0] > 1*10^(-6) and yhat[0] > 1*10^(-6):
      count_true += 1

  print(count_true)



