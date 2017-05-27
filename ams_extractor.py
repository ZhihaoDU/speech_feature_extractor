# coding = utf-8

import numpy as np
from feature_extractor import stft_extractor
import matplotlib.pyplot as plt
from scipy.io import wavfile


# axis=0 for col vectors
# axis=1 for row vectors
def normalize_matrix(m, axis=0):
    norm = np.linalg.norm(m, axis=axis, keepdims=True)
    return m / np.repeat(norm, norm.shape[axis], axis=axis)


# axis=0 for col vectors
# axis=1 for row vectors
def uniformize_matrix(m, axis=None):
    if axis is None:
        maxm = np.ones(m.shape, dtype=m.dtype) * np.max(m)
        minm = np.ones(m.shape, dtype=m.dtype) * np.min(m)
    else:
        maxm = np.repeat(np.max(m, axis=axis, keepdims=True), m.shape[axis], axis=axis)
        minm = np.repeat(np.min(m, axis=axis, keepdims=True), m.shape[axis], axis=axis)
    return (m-minm) / (maxm-minm)


def freq2bark(f):
    return 7*np.log(f/650+np.sqrt(np.power(1+(f/650), 2)))


def bark2freq(b):
    return 650*np.sinh(b/7)


def get_fft_bark_mat(sr, fft_len, barks, min_frq=20, max_frq=None):
    if max_frq is None:
        max_frq = sr // 2
    fft_frqs = np.arange(0, fft_len//2+1) / (1.*fft_len) * sr
    min_bark = freq2bark(min_frq)
    max_bark = freq2bark(max_frq)
    bark_bins = bark2freq(min_bark + np.arange(0, barks+2) / (barks + 1.) * (max_bark - min_bark))
    wts = np.zeros((barks, fft_len//2+1))
    for i in range(barks):
        fs = bark_bins[[i+0, i+1, i+2]]
        loslope = (fft_frqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fft_frqs) / (fs[2] - fs[1])
        wts[i, :] = np.maximum(0, np.minimum(loslope, hislope))
    return wts


def ams_extractor(x, sr, win_len, shift_len, barks, inner_win, inner_shift, win_type, method_version):
    x_spectrum = stft_extractor(x, win_len, shift_len, win_type)
    coef = get_fft_bark_mat(sr, win_len, barks, 20, sr//2)
    bark_spect = np.matmul(coef, x_spectrum)
    ams = np.zeros((barks, inner_win//2+1, (bark_spect.shape[1] - inner_win)//inner_shift))
    for i in range(barks):
        channel_stft = stft_extractor(bark_spect[i, :], inner_win, inner_shift, 'hanning')
        if method_version == 'v1':
            ams[i, :, :] = 20 * np.log(np.abs(channel_stft[:inner_win//2+1, :(bark_spect.shape[1] - inner_win)//inner_shift]))
        elif method_version == 'v2':
            channel_amplitude = np.abs(channel_stft[:inner_win//2+1, :(bark_spect.shape[1] - inner_win)//inner_shift])
            channel_angle = np.angle(channel_stft[:inner_win//2+1, :(bark_spect.shape[1] - inner_win)//inner_shift])
            channel_angle = channel_angle - (np.floor(channel_angle / (2.*np.pi)) * (2.*np.pi))
            ams[i, :, :] = np.power(channel_amplitude, 1./3.) * channel_angle
        else:
            ams[i, :, :] = np.abs(channel_stft)
    return ams


if __name__ == '__main__':
    fft_2_bark = get_fft_bark_mat(16000, 320, 24, 20, 8000)
    fs, x = wavfile.read('clean.wav')
    ams_feature = ams_extractor(x, fs, int(fs*0.02), int(fs*0.01), 24, 32, 1, 'hanning', 'v1')
    ams_feature = uniformize_matrix(ams_feature)
    plt.subplot(311)
    plt.imshow(ams_feature[0, :, :])
    plt.subplot(312)
    plt.imshow(ams_feature[5, :, :])
    plt.subplot(313)
    plt.imshow(ams_feature[10, :, :])
    plt.show()
