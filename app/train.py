
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split


data =  pd.read_csv(r"/My_pr/letter_number.csv").astype('float32')
del data[data.columns[0]]
data.head(10)

# delete unnecessary characters for VIN code
indexLetters = data[ (data['label'] == 8) | (data['label'] == 14) | (data['label'] == 16)].index
data.drop(indexLetters , inplace=True)

print(data.shape)

X = data.drop('label',axis = 1)
y = data['label']

# plot the number of characters in the dataset
characters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H', 8: 'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O', 15:'P', 16: 'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
y_int = np.int0(y)
count = np.zeros(36, dtype='int')
for i in y_int:
    count[i] +=1
characters = []
for i in characters_dict.values():
    characters.append(i)
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.barh(characters, count)
plt.xlabel("Number of elements ")
plt.ylabel("Characters")
plt.grid()
plt.show()

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# data preprocessing
X_train = np.reshape(X_train.values, (X_train.shape[0], 28,28))
X_test = np.reshape(X_test.values, (X_test.shape[0], 28,28))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
print("New shape of train data: ", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)
print("New shape of train data: ", X_test.shape)

y_train = to_categorical(y_train, num_classes = 36, dtype='int')
print("New shape of train labels: ", y_train.shape)
y_test = to_categorical(y_test, num_classes = 36, dtype='int')
print("New shape of test labels: ", y_test.shape)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(36,activation ="softmax"))

model.summary()

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 1)

model.save(r"app/model.h5")

loss, accuracy = model.evaluate(X_test,y_test)