import tensorflow as tf
from tensorflow import keras

from helpers import encode_review, decode_review, pad

import numpy as np

# train and test dataset
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# print(train_data[0])
# len(train_data[0]), len(train_data[1])

train_data = pad(train_data)
test_data = pad(test_data)

# len(train_data[0]), len(train_data[1])
# print(train_data[0])

# build model
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
# evaluate model
val_loss, val_acc = model.evaluate(test_data, test_labels)
print("Val acc: {}, Val loss: {}".format(val_acc, val_loss))

# save model
model.save('movie_review.model')
















