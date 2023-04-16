import random
from idna import valid_string_length
import numpy as np
import os
import glob
import math
import json
import cv2
from neuralnet.SignalCollection import SignalCollection

# ------- [ PROCESSING CONSTANTS ] --------------
c = 0.3  # Light velocity, m/ns
freq0 = 3.1  # Start frequency, GHz
freqMax = 3.6  # Stop frequency, GHz
d = 0.061  # MIMO array step, m
T = 10  # Chirp duration
P = 1  # A pause between chirps
N = 1024  # No. of samples in a chirp
rxAz = 5
Nn = 4096  # padded samples
azPadding = 64  # Azimuth angle padding
Tc = T + P  # Chirp repetition period
lambda0 = c / freq0  # Wavelength, m
radarToVideoRatio = (T + P) / 50.0  # 20 FPS is assumed, which is what is measured in the
# first iteration of the data acquisition
chirpsPerFrame = 32
removeVelocities = 0
log_mean = 20.25
log_variance = 7.46
rxNum = 8
rangePoints = 256
rangePadding = 4096


def chirp_to_compressed(chirp: np.ndarray, rangePoints):
    chirpPadded = np.zeros((rangePadding, rxAz))
    chirpPadded[:N, :rxAz] = chirp[:, :rxAz]
    ffted = np.fft.fft(chirpPadded, axis=0)[:rangePoints]  # FFT & truncating to 256 x 5

    return ffted


def frame_to_compressed(frame: np.ndarray, rangePoints=256):
    compressed = np.zeros((rangePoints, rxAz, chirpsPerFrame), dtype=np.complex64)  # 256 x 5 x 32
    for chirpIndex in range(chirpsPerFrame):
        # Ghetto way to unpack it and change format to 1024 x 8 from 8 x 1024, row-major
        chirp = np.swapaxes(frame[chirpIndex].reshape(rxNum, N), 1, 0)
        compressed[:, :, chirpIndex] = chirp_to_compressed(chirp, rangePoints)
    return compressed


def compressed_to_raddet(compressed: np.ndarray):
    paddedFrame = np.zeros((rangePoints, azPadding, chirpsPerFrame), dtype=np.complex64)  # 256 x 64 x 32

    paddedFrame[:, :rxAz, :] = compressed

    # 2-D FFT & FFT-shift on the azimuth & doppler axes (last 2)
    frame = np.fft.fftshift(np.fft.fft2(paddedFrame, axes=(1, 2)), axes=(1, 2))

    middleIndex = chirpsPerFrame // 2
    # removing unwanted velocities
    frame[:, :, middleIndex - removeVelocities:middleIndex + removeVelocities + 1] = 0.0 + 0.0j  # removing data with 0 velocity

    return frame


def raddet_to_box(raddet: np.ndarray):
    box = np.log(np.abs(raddet) + 1e-9)
    box -= log_mean
    box *= 1/log_variance

    return box


class RawDataLoader:
    def __init__(self, filePath):

        self.filePath = filePath
        self.collection = SignalCollection(self.filePath, raw=True).to_numpy()

        self.totalFrames = len(self.collection) // chirpsPerFrame

        print(f"There are {self.totalFrames} frames in {filePath}")
        self.currentFrameIndex = 0

    def produce_frame(self):
        whichFrame = self.currentFrameIndex
        self.currentFrameIndex += 1
        if self.currentFrameIndex >= self.totalFrames:
            self.currentFrameIndex = 0
        rawFrame = self.collection[whichFrame * chirpsPerFrame: (whichFrame + 1)*chirpsPerFrame]
        compressed = frame_to_compressed(rawFrame)
        raddet = compressed_to_raddet(compressed)
        return raddet, self.dummy_gt()

    @staticmethod
    def dummy_gt():
        boxes3d = np.ones((0, 6))
        boxes2d = np.ones((0, 4))
        classes = []

        return {"boxes": boxes3d,
                "classes": classes,
                "cart_boxes": boxes2d}

    def get_total_samples(self):
        return self.totalFrames

    def yield_next_data(self):
        while True:
            yield self.produce_frame()
