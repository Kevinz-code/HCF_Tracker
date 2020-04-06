import torch
import cv2
import numpy as np
import time
from scipy.fftpack import fft2
from conv_cf import fft


def get_peak(response, l):
    pos = int(np.argmax(response) + 1)
    pos_y = int(np.ceil(pos / l))  # row
    pos_x = int(pos % l)  # line
    if pos_x == 0:
        pos_x = l
    peak = response[pos_y - 1, pos_x - 1]

    return pos_y, pos_x, peak



a = np.array([[2, 55, 45], [1, 7.6, 3.5], [3, 9, 2]], dtype=float)
print(get_peak(a,l=3),a)
exit()

b = np.array([[1, 5, 6.5], [3, 9, 2.3]], dtype=float)
print(fft(a))
print("\n\n")
print(fft(b))

print("\n\n")
print(cv2.mulSpectrums(fft(a), fft(b), flags=0, conjB=True))
exit()

a = torch.tensor(np.arange(64*64*2, dtype=float).reshape((64,64,2)))
b = np.arange(64*64, dtype=float).reshape(64,64)

c1 = np.random.randn(4, 4)
c2 = np.random.randn(8, 8)
c3 = np.random.randn(16,16)
t1 = time.time()
for i in range(512):
    c1 = cv2.resize(c1,dsize=(128,128))
    c2 = cv2.resize(c2,dsize=(128,128))
    c3 = cv2.resize(c3,dsize=(128,128))
t2 = time.time()
print(1000*float(t2 - t1),"ms")
exit()



t1 = time.time()
for i in range(10000):
    torch.fft(a, signal_ndim=2)
t2 = time.time()
print(1000*float(t2 - t1),"ms")

t1 = time.time()
for i in range(10000):
    cv2.dft(b, flags=cv2.DFT_COMPLEX_OUTPUT)
t2 = time.time()
print(1000*float(t2 - t1),"ms")

t1 = time.time()
for i in range(10000):
    np.fft.fft(b)
t2 = time.time()
print(1000*float(t2 - t1),"ms")

t1 = time.time()
for i in range(10000):
    fft2(b)
t2 = time.time()
print(1000*float(t2 - t1),"ms")

