import keras
from keras import layers

encoding_dim = 32
px_num  = 784
input_img = keras.Input(shape=(px_num,))

encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

decoded = layers.Dense(px_num, activation='sigmoid')(encoded)

autoencoder = keras.Model(input_img, decoded)


encoder = keras.Model(input_img, encoded)

encoded_input = keras.Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

from keras.datasets import mnist
import numpy as np

(x_train,_), (x_test,_) = mnist.load_data()

x_test = x_test[0]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = np.array([x_test.flatten()])
print(x_test.shape)



# Bug below this line
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs.shape)
print(decoded_imgs.shape)
