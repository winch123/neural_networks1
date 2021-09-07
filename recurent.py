#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, SpatialDropout1D
from keras.datasets import imdb
from keras.utils import to_categorical
import sys

'''
# Устанавливаем seed для повторяемости результатов
np.random.seed(42)
# Максимальное количество слов (по частоте использования)
max_features = 5000
# Максимальная длина рецензии в словах
maxlen = 80

# Загружаем данные
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Заполняем или обрезаем рецензии
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

model = Sequential()

model.add(Embedding(max_features, 32))		# Слой для векторного представления слов
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 	# Слой долго-краткосрочной памяти
model.add(Dense(1, activation="sigmoid"))	# Полносвязный слой

# Копмилируем модель
model.compile(loss='binary_crossentropy',  optimizer='adam',  metrics=['accuracy'])

# Обучаем модель
model.fit(X_train, y_train, batch_size=64, epochs=7,  validation_data=(X_test, y_test), verbose=2)
# Проверяем качество обучения на тестовых данных
scores = model.evaluate(X_test, y_test, batch_size=64)
print "Точность на тестовых данных: %.2f%%" % (scores[1] * 100)
'''

## https://neurohive.io/ru/tutorial/nejronnaya-set-keras-python/



(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
#print training_data.shape
#print training_targets.shape

data = np.concatenate((training_data, testing_data), axis=0)
print data.shape
#for i in data:
#  print len(i)
print "Number of unique words:", len(np.unique(np.hstack(data)))  

#targets = np.concatenate((training_targets, testing_targets), axis=0)
#print targets.shape
#print "Categories:", np.unique(targets)

length = [len(i) for i in data]
print "Average Review length:", np.mean(length)
print "Standard Deviation:", round(np.std(length))


def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
      results[i, sequence] = 1
    return results
 
data = vectorize(data)
print data
print data.shape
sys.exit()

targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

sys.exit()

model = models.Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
results = model.fit( train_x, train_y, epochs= 2, batch_size = 500, validation_data = (test_x, test_y))
print "Test-Accuracy:", np.mean(results.history["val_acc"])

