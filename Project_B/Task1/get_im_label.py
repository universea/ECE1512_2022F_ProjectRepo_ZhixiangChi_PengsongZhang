import numpy as np
import cv2

# CIFAR10
c = np.load('CIFAR10.npz')["images"]
label = np.argmax(np.load('CIFAR10.npz')['labels'], axis=1)
b = c.transpose((0, 2, 3, 1))

big_im = np.zeros((320, 320, 3))
for i in range(100):
    row = int(i / 10)
    col = i % 10
    im = b[i]
    im_norm = (im - im.min()) / (im.max() - im.min())
    big_im[row * 32: (row + 1) * 32, col * 32: (col + 1) * 32, :] = im_norm
print(np.min(big_im), np.max(big_im))


cv2.imwrite('cifar10_condensed.png',np.uint8(big_im * 255.0))

# MNIST
c = np.load('MNIST.npz')["images"]
label = np.argmax(np.load('MNIST.npz')['labels'], axis=1)
b = c.transpose((0, 2, 3, 1))

big_im = np.zeros((280, 280, 1))
for i in range(100):
    row = int(i / 10)
    col = i % 10
    im = b[i]
    im_norm = (im - im.min()) / (im.max() - im.min())
    big_im[row * 28: (row + 1) * 28, col * 28: (col + 1) * 28, :] = im_norm
print(np.min(big_im), np.max(big_im))

cv2.imwrite('mnist_condensed.png',np.uint8(big_im * 255.0))