from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# data = pd.read_csv("./data/Impairment_Total_b.csv")
data = pd.read_csv("./data/total_data_b.csv", encoding='latin1')
df = pd.DataFrame(data)
print(df)
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':  # 범주형 열에만 적용
        df[col] = le.fit_transform(df[col])

print(df)

df_corr = df.corr()
df_corr_sort = df_corr.sort_values("ImpairmentPeople", ascending = False)
print(df_corr_sort["d"].head(10))
#
cols = ['Years', 'Location', 'AllPeople' ]

X_train_pre = df[cols]
y = df['ImpairmentPeople'].values

X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience=30)

modelpath = "./data/model/Ch15-uncommon6.hdf5"

Checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose = 0, save_best_only = True)

history = model.fit(X_train, y_train, validation_split=0.25, epochs = 2000, batch_size=50, callbacks=[early_stopping_callback, Checkpointer])


PredictPeople = []
RealPeople = []
X_num = []

n_iter = 0
Y_prediction = model.predict(X_test).flatten()
for i in range(25):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제 장애인 수 : {:.2f}명, 예상 장애인 수 : {:.2f}명".format(real, prediction))
    RealPeople.append(real)
    PredictPeople.append(prediction)
    n_iter = n_iter+1
    X_num.append(n_iter)

plt.plot(X_num, PredictPeople, label='Predicted Population')
plt.plot(X_num, RealPeople, label='Real Population')
plt.show()
plt.legend()

colormap = plt.cm.gist_heat
plt.figure(figsize=(12,12))

sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show()
