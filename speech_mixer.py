# coding = utf-8
import numpy as np
from read_sphere_wav import read_sphere_wav
from scipy.io import wavfile
from feature_extractor import *

from matplotlib import pyplot as plt


def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))


def signal_by_db(x1, x2, snr, handle_method):
    x1 = x1.astype(np.int32)
    x2 = x2.astype(np.int32)
    l1 = x1.shape[0]
    l2 = x2.shape[0]
    if l1 != l2:
        if handle_method == 'cut':
            ll = min(l1, l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            ll = max(l1, l2)
            if l1 < ll:
                x1 = np.append(x1, x1[:ll-l1])
            if l2 < ll:
                x2 = np.append(x2, x2[:ll-l1])

    from numpy.linalg import norm
    x2 = x2 / norm(x2) * norm(x1) / (10.0 ** (0.05 * snr))
    mix = x1 + x2

    return mix


if __name__ == '__main__':
    speech_data, wav_header = read_sphere_wav(u"/media/neo/000C6F0F00042510/Doctor/dataset/TIMIT/train/dr1/fcjf0/sa1.wav")
    fs, noise_data = wavfile.read('/media/neo/000C6F0F00042510/Doctor/dataset/DEMAND/PCAFETER/ch01.wav')
    plt.figure()
    spect = log_power_spectrum_extractor(speech_data, 320, 160, 'hanning', True)
    plt.subplot(311)
    plt.imshow(spect)
    noisy_speech = signal_by_db(speech_data, noise_data, 5, 'cut')
    spect = log_power_spectrum_extractor(noisy_speech, 320, 160, 'hanning', True)
    plt.subplot(312)
    plt.imshow(spect)
    noisy_speech = signal_by_db(speech_data, noise_data, 0, 'cut')
    spect = log_power_spectrum_extractor(noisy_speech, 320, 160, 'hanning', True)
    plt.subplot(313)
    plt.imshow(spect)
    plt.figure()
    noisy_speech = signal_by_db(speech_data, noise_data, -5, 'cut')
    spect = log_power_spectrum_extractor(noisy_speech, 320, 160, 'hanning', True)
    plt.subplot(211)
    plt.imshow(spect)
    noisy_speech = signal_by_db(speech_data, noise_data, -10, 'cut')
    spect = log_power_spectrum_extractor(noisy_speech, 320, 160, 'hanning', True)
    plt.subplot(212)
    plt.imshow(spect)
    plt.show()
    #sd.play(noisy_speech.astype(np.int32), fs, blocking=True)
