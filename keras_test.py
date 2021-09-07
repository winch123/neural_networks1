#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy, sys

numpy.random.seed(42)
img_cols = 32
img_rows = 32
img_channels = 3
nb_classes = 10

### https://habr.com/ru/post/352678/

def InitData():
  from keras import datasets
  from keras.utils import np_utils
  (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
  #print X_train.dtype, y_train.dtype
  #print X_train.shape  

  # Преобразование размерности изображений
  #X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
  #X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

  X_train = X_train.astype('float32') / 255
  X_test = X_test.astype('float32') / 255

  # метки в категории
  #print y_train.shape
  Y_train = np_utils.to_categorical(y_train, nb_classes)
  #print Y_train.shape
  #sys.exit()
  Y_test = np_utils.to_categorical(y_test, nb_classes)

  return X_train, Y_train, X_test, Y_test

def TrainNewModel(FileName):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_rows, img_cols, img_channels), activation='relu'))    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))    
    model.add(MaxPooling2D(pool_size=(2, 2))) # подвыборка
    model.add(Dropout(0.25)) #регуляризация
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) # Третий сверточный
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    from keras.optimizers import SGD
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    print model.summary()
    open(FileName + ".json", "w").write(model.to_json())
    #return

    model.fit(X_train, Y_train, batch_size=32, epochs=22, validation_split=0.1, shuffle=True, verbose=1)
    model.save_weights(FileName + ".h5")
    
    print "тестирование...."
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print "Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100)
    
def LoadModel(FileName, doPrint=True):
  from keras.models import model_from_json
  m = model_from_json(open(FileName + ".json", "r").read())
  m.load_weights(FileName + ".h5")
  m.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
  if doPrint:
    print m.summary()
  return m

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = InitData()
  #TrainNewModel('cifar100')
  model = LoadModel('cifar100', False)  
  scores = model.evaluate(X_test, Y_test, verbose=1)
  print "Точность работы загруженной сети на тестовых данных: %.2f%%" % (scores[1]*100)
  sys.exit()
  
  from keras.applications.vgg16 import preprocess_input, decode_predictions
  from keras.preprocessing import image  
  img = image.load_img('bus.jpg', target_size=(img_rows, img_cols, img_channels))
  x = image.img_to_array(img)
  x = numpy.expand_dims(x, axis=0)
  #x = preprocess_input(x)
  pred = model.predict(x)
  print pred
  print 'Результаты распознавания:', decode_predictions(pred, top=3)[0]


