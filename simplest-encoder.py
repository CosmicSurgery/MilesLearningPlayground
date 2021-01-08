import keras
from keras import layers

# the size of our encoded representations

encoding_dim = 32

# Our input image
input_img = keras.Input(shape=(625,))

# encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

# decoded is the lossy reconstruction of the input
decoded = layers.Dense(625, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# seperate encoder model:
encoder = keras.Model(input_img,encoded)

# and a decoder model:

# our encoded (32 - dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

'''

'''

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train,_),(x_test,_) = mnist.load_data()

mydata = np.load('train.npy')
x_train = mydata[:50000]
x_test = mydata[50000:]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32') 
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n = 10 # number of digits
plt.figure(figsize=(20,4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(25,25))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(25,25))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

























