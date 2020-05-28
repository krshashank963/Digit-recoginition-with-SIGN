import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense , Flatten, Dropout
from keras.optimizers import Adam


from sklearn.model_selection import train_test_split
X = np.load("X.npy")
Y = np.load("Y.npy")
plt.figure(figsize=(24,8))

plt.subplot(2,5,1)

plt.imshow(X[0])
plt.subplot(2,5,2)

plt.imshow(X[1200])
plt.subplot(2,5,3)

plt.imshow(X[1500])
plt.subplot(2,5,4)

plt.imshow(X[1700])

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
x_train = x_train.reshape(-1,64,64,1)
x_test = x_test.reshape(-1,64,64,1)

model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(5,5),activation="relu",padding="same",input_shape=(64,64,1)))
model.add(Conv2D(filters=32,kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Conv2D(filters=32,kernel_size=(4,4),activation="relu",padding="same"))
model.add(Conv2D(filters=32,kernel_size=(4,4),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Dropout(0.2))

model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Dropout(0.2))

model.add(Conv2D(filters=32,kernel_size=(2,2),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(40,activation="relu"))

model.add(Dense(10,activation="softmax"))

model.compile(optimizer=Adam(lr=0.0002),loss=keras.losses.categorical_crossentropy,metrics=["accuracy"])

results = model.fit(x_train,y_train,epochs=25,validation_data=(x_test,y_test))
plt.plot(results.history["val_acc"],label="validation_accuracy",c="red",linewidth=4)
plt.plot(results.history["acc"],label="training_accuracy",c="green",linewidth=4)
