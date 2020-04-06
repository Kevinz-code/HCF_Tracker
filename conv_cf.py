from __future__ import division
import numpy as np
import cv2
import time
import torch


def real(x):
    # real value of a complex
    return x[:, :, 0]


def fft(area):
    # forward
    return cv2.dft(np.float32(area), flags=cv2.DFT_COMPLEX_OUTPUT)


def ifft(area):
    # backward and remember the scale
    return cv2.dft(np.float32(area), flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))


def complex_division(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1.0 / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2 + 0.0000001)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] - a[:, :, 0] * b[:, :, 1]) * divisor
    return res

def mycomplex_division(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1.0 / (b[:, :, :, 0] ** 2 + b[:, :, :, 1] ** 2 )

    res[:, :, :, 0] = (a[:, :, :, 0] * b[:, :, :, 0] + a[:, :, :, 1] * b[:, :, :, 1]) * divisor
    res[:, :, :, 1] = (a[:, :, :, 1] * b[:, :, :, 0] - a[:, :, :, 0] * b[:, :, :, 1]) * divisor
    return res


class HCF(object):
    def __init__(self, lamda, adapt=0.01, TargetGaussianBand=30.0):

        # important parameters
        self.lamda = lamda
        self.s = TargetGaussianBand
        self.fixed_size = [32, 16, 8, 256]  # conv3 conv4 conv5
        self.l = 0  # layers 0, 1, 2
        self.tmp_size = 0
        self.D = 0  # dimension

        # other essentials
        self.A = 0.0
        self.B = 0.0
        self.adapt = adapt
        self.hann = 0.0

        self.ac_conv = 0.0
        self.conv = 0.0

    def init_conv(self, conv, D, layers):
        conv = conv[0, :, :, :].cpu()   # extracted to cpu
        conv = conv.numpy()  # extracted to numpy
        conv = np.transpose(conv, axes=(1, 2, 0))  # swap axis

        self.ac_conv = conv
        self.conv = conv
        self.D = D
        self.l = layers - 3

        self.tmp_size = self.fixed_size[self.l]
        self.A = np.zeros((self.tmp_size, self.tmp_size, self.D, 2))
        self.B = np.zeros((self.tmp_size, self.tmp_size, self.D, 2))

        self.create_hanning()
        self.ac_conv = self.ac_conv * self.hann

    def create_hanning(self):
        N_2 = self.tmp_size
        N_1 = self.tmp_size
        hann2t, hann1t = np.ogrid[0:N_2, 0:N_1]

        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (N_2 - 1)))
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (N_1 - 1)))
        hann2d = hann2t * hann1t

        self.hann = hann2d[:, :, None]
        self.hann = self.hann.astype(np.float32)

    def create_target(self, len_y, len_x):
        half_y = (len_y - 0) / 2.0
        half_x = (len_x - 0) / 2.0

        # calculate the bandwidth
        target_sigma = np.sqrt(len_y*len_x) / self.s  # float array
        bandwidth = (-2.0) * (target_sigma**2)

        # generate grid
        y_vector, x_vector = np.ogrid[0:len_y, 0:len_x]

        # do not forget the sequence
        y_vector = (y_vector - half_y)**2
        x_vector = (x_vector - half_x)**2
        target = np.exp((y_vector + x_vector) / bandwidth)

        return target

    def train(self, x, y):
        tmp_A = cv2.mulSpectrums(fft(y), fft(x), flags=0)
        tmp_B = cv2.mulSpectrums(fft(x), fft(x), flags=0, conjB=True)
        return tmp_A, tmp_B

    def get_response_map(self, cur_conv):
        # to cpu and
        # pre operation
        cur_conv = cur_conv[0, :, :, :].cpu()   # extracted to cpu
        cur_conv = cur_conv.numpy()             # extracted to numpy
        cur_conv = np.transpose(cur_conv, axes=(1, 2, 0))   #swap axis

        # pre
        # operation
        cur_conv = cur_conv[:, :] * self.hann
        self.ac_conv = (1.0 - self.adapt) * self.ac_conv + self.adapt * self.conv
        target_y = self.create_target(self.tmp_size, self.tmp_size)

        # training and
        # get A, B
        A = np.zeros((self.tmp_size, self.tmp_size, self.D, 2))
        B = np.zeros((self.tmp_size, self.tmp_size, self.D, 2))
        for i in range(self.D):
            A[:, :, i, :], B[:, :, i, :] = self.train(x=self.ac_conv[:, :, i], y=target_y)
        self.A = (1.0 - self.adapt) * self.A + self.adapt * A
        B = np.sum(B, axis=2)
        B = np.asarray([B for i in range(self.D)])
        B = np.transpose(B, (1, 2, 0, 3))
        self.B = (1 - self.adapt) * self.B + self.adapt * B
        W = mycomplex_division(self.A, self.B + self.lamda).astype(np.float32)  # # important

        # detect
        # process
        response = np.zeros((self.tmp_size, self.tmp_size, self.D, 2))
        for i in range(self.D):
            response[:, :, i, :] = cv2.mulSpectrums(fft(cur_conv[:, :, i]), W[:, :, i, :], flags=0, conjB=True)
        response = np.sum(response, axis=2)
        response = ifft(response)
        response = real(response)

        self.conv = cur_conv

        '''
        cur_conv = cur_conv * self.hann
        target_y = self.create_target(self.tmp_size, self.tmp_size)

        # training and get A, B
        A, B = self.train(x=self.conv, y=target_y)

        self.A = (1.0 - self.adapt) * self.A + self.adapt * A
        self.B = (1 - self.adapt) * self.B + self.adapt * B
        print(A.shape, B.shape)
        W = complex_division(self.A, self.B + self.lamda).astype(np.float32)  # # important

        print(W.shape)
        print(fft(cur_conv).shape)

        # detect process
        response = cv2.mulSpectrums(fft(cur_conv), W, flags=0, conjB=True)
        response = ifft(response)
        response = real(response)

        self.conv = cur_conv

        '''

        return response



