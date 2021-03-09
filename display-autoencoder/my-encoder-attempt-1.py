from keras.datasets import mnist
import numpy as np
(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_test[0].shape)


'''
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_test[0].reshape(28,28))

plt.show()

'''
