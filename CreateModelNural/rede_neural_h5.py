import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import os
import pandas as pd 
import numpy as np 
import random

from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import matplotlib.image as mpimg
from imgaug import augmenters as iaa

import cv2
import matplotlib.image as mpimg

from matplotlib import pyplot as plt

path = "data/AUTONOMOUS_CAR"
path_imgs = path + "IMG/"

df = pd.read_csv(path + "/driving_log.csv",  names=['center','left','right','steering','throttle','brake','speed'])

print(df.head())

# PRE-PROCESSAMENTO
def getName(filePath):
    return filePath.split("\\")[-1]


columns = ['center','left','right','steering','throttle','brake','speed']
df = pd.read_csv(os.path.join(path, "driving_log.csv"),  names=columns)
df['center'] = df['center'].apply(getName)
df['left'] = df['left'].apply(getName)
df['right'] = df['right'].apply(getName)

print(df['center'][0])

print("Total de imagens importadas : {}".format(df.shape[0]))

# BALANCIAMENTO DE DADOS

nBins = 31
samplesPerBin = 2000

hist, bins = np.histogram(df['steering'], nBins)
# print(bins)
    
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.06)
plt.plot((-1,1), (samplesPerBin,samplesPerBin))
# plt.show()

#EXCLUIINDO DADOS DESNECEESSARIOS

removeindexList = []
for j in range(nBins):
    binDataList = []
    for i in range(len(df['steering'])):
        if df['steering'][i] >= bins[j] and df['steering'][i] <= bins[j + 1]:
            binDataList.append(i)
    binDataList = shuffle(binDataList)
    binDataList = binDataList[samplesPerBin:]
    removeindexList.extend(binDataList)

# print('Removed Images:', len(removeindexList))
df.drop(df.index[removeindexList], inplace=True)
# print('Remaining Images:', len(df))

hist, _ = np.histogram(df['steering'], (nBins))
plt.bar(center, hist, width=0.06)
plt.plot((np.min(df['steering']), np.max(df['steering'])), (samplesPerBin, samplesPerBin))
# plt.show()

def loadData(path, data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]

        choice = np.random.choice(3)
        if choice == 0:
            imagesPath.append(os.path.join(path, 'IMG' ,indexed_data[0]))
            steering.append(float(indexed_data[3]))

        elif choice == 1:
            imagesPath.append(os.path.join(path, 'IMG' ,indexed_data[1]))
            steering.append(float(indexed_data[3])+0.2)

        else:
            imagesPath.append(os.path.join(path, 'IMG' ,indexed_data[1]))
            steering.append(float(indexed_data[3])-0.2)

    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)

    return imagesPath, steering

X, y = loadData(path, df)

# CRIANDO CONJUNTOS DE TREINO E TESTE

xTrain, xVal, yTrain, yVal = train_test_split(X, y, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

# DATA AUGMENTATION

def preProcess(img):
    img = img[60:135,:,:] #CROP
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #RGB TO YUV
    img = cv2.GaussianBlur(img,  (3, 3), 0) # BLUR
    img = cv2.resize(img, (200, 66)) #RESIZE
    img = img/255
    return img 

def augmentImage(imgPath,steering):
    img = mpimg.imread(imgPath)

    #PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    #ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    #BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)

    #FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)

            if trainFlag: #APLICA DATA AUG NO TREINO
                img, steering = augmentImage(imagesPath[index], steeringList[index])

            else: # CARREGA IMAGEM NA VALIDAÇÃO
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]

            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield (np.asarray(imgBatch), np.asarray(steeringBatch))


#HIPERPARÂMETROS


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

EPOCHS = 50
BATCH_SIZE = 64
alpha  = 1e-5

#CONVOLUTIONAL NEURAL NETWORK

model = Sequential()
model.add(layers.Conv2D(64, (5, 5), activation='elu', padding="same", input_shape=INPUT_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128,(3, 3), activation='elu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, (3, 3), activation='elu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, (3, 3), activation='elu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(512, (1, 1), activation='elu', padding="same"))
model.add(layers.BatchNormalization())

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(258, activation="elu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae', 'mse'])

# KERAS CALLBACK

filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=alpha, patience=5, verbose=1)
stopping = EarlyStopping(monitor="val_loss", min_delta=alpha, patience=15, verbose=1)
callbacks = [checkpoint, lr_reduce, stopping]

history = model.fit(batchGen(xTrain, yTrain,batchSize=100, trainFlag=1),

                    steps_per_epoch=300,
                    epochs=50,
                    validation_data=batchGen(xVal, yVal, batchSize=100, trainFlag=0),
                    validation_steps=200,
                    callbacks = callbacks
                    )