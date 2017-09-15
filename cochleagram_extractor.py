# coding = utf-8
import numpy as np
from read_sphere_wav import read_sphere_wav
from matplotlib import pyplot as plt
from scipy.signal import lfilter
from spectrum_extractor import spectrum_extractor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize


def erb_space(low_freq=50, high_freq=8000, n=64):
    ear_q = 9.26449
    min_bw = 24.7

    cf_array = -(ear_q * min_bw) + np.exp(np.linspace(1,n,n) * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw)) / n) \
                * (high_freq + ear_q * min_bw)
    return cf_array


def make_erb_filters(fs, num_channels, low_freq):
    t = 1. / fs
    cf = erb_space(low_freq, fs // 2, num_channels)

    ear_q = 9.26449
    min_bw = 24.7
    order = 4

    erb = np.power(np.power(cf/ear_q, order) + (min_bw ** order), 1./order)
    b = 1.019 * 2 * np.pi * erb

    a0 = t
    a2 = 0
    b0 = 1
    b1 = -2 * np.cos(2 * cf * np.pi * t) / np.exp(b*t)
    b2 = np.exp(-2 * b * t)

    a11 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) + 2 * np.sqrt(3+2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2
    a12 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) - 2 * np.sqrt(3+2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2
    a13 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) + 2 * np.sqrt(3-2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2
    a14 = -(2 * t * np.cos(2*cf*np.pi*t) / np.exp(b*t) - 2 * np.sqrt(3-2**1.5) * t * np.sin(2*cf*np.pi*t) / np.exp(b*t))/2

    p1 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) - np.sqrt(3 - 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p2 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) + np.sqrt(3 - 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p3 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) - np.sqrt(3 + 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p4 = (-2*np.exp(4j*cf*np.pi*t)*t + 2*np.exp(-(b*t) + 2j*cf*np.pi*t) * t *
         (np.cos(2*cf*np.pi*t) + np.sqrt(3 + 2**(3/2))* np.sin(2*cf*np.pi*t)))
    p5 = np.power(-2 / np.exp(2*b*t) - 2 * np.exp(4j*cf*np.pi*t) + 2 * (1 + np.exp(4j*cf*np.pi*t)) / np.exp(b*t), 4)
    gain = np.abs(p1 * p2 * p3 * p4 / p5)

    allfilts = np.ones((np.size(cf, 0), 1), dtype=np.float32)
    fcoefs = np.column_stack((a0*allfilts, a11, a12, a13, a14, a2*allfilts, b0*allfilts, b1, b2, gain))
    return fcoefs, cf


def erb_frilter_bank(x, fcoefs):
    a0 = fcoefs[:, 0]
    a11 = fcoefs[:, 1]
    a12 = fcoefs[:, 2]
    a13 = fcoefs[:, 3]
    a14 = fcoefs[:, 4]
    a2 = fcoefs[:, 5]
    b0 = fcoefs[:, 6]
    b1 = fcoefs[:, 7]
    b2 = fcoefs[:, 8]
    gain = fcoefs[:, 9]

    output = np.zeros((np.size(gain, 0), np.size(x, 0)))

    for chan in range(np.size(gain, 0)):
        y1 = lfilter(np.array([a0[chan] / gain[chan], a11[chan] / gain[chan], a2[chan] / gain[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), x)
        y2 = lfilter(np.array([a0[chan], a12[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y1)
        y3 = lfilter(np.array([a0[chan], a13[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y2)
        y4 = lfilter(np.array([a0[chan], a14[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y3)

        output[chan, :] = y4
    return output


def cochleagram_extractor(xx, sr, win_len, shift_len, channel_number, win_type):
    fcoefs, f = make_erb_filters(sr, channel_number, 50)
    fcoefs = np.flipud(fcoefs)
    xf = erb_frilter_bank(xx, fcoefs)

    if win_type == 'hanning':
        window = np.hanning(channel_number)
    elif win_type == 'hamming':
        window = np.hamming(channel_number)
    elif win_type == 'triangle':
        window = (1 - (np.abs(channel_number - 1 - 2 * np.arange(1, channel_number + 1, 1)) / (channel_number + 1)))
    else:
        window = np.ones(channel_number)
    window = window.reshape((channel_number, 1))

    xe = np.power(xf, 2.0)
    frames = 1 + ((np.size(xe, 1)-win_len) // shift_len)
    cochleagram = np.zeros((channel_number, frames))
    for i in range(frames):
        one_frame = np.multiply(xe[:, i*shift_len:i*shift_len+win_len], np.repeat(window, win_len, 1))
        cochleagram[:, i] = np.sqrt(np.mean(one_frame, 1))

    cochleagram = np.where(cochleagram == 0.0, np.finfo(float).eps, cochleagram)
    return cochleagram


def cochleagram_fft_coefs(sr, win_len, channel_number):
    min_freq = 50.0
    max_freq = sr//2
    max_len = win_len
    nfilts = channel_number
    nfft = win_len

    wts = np.zeros((nfilts, nfft // 2 + 1))
    ear_q = 9.26449
    min_bw = 24.7
    order = 1.0
    cfreqs = -(ear_q * min_bw) + np.exp(np.arange(1, nfilts+1, 1) * (-np.log(max_freq+ear_q*min_bw) + np.log(min_freq + ear_q*min_bw)) / nfilts) * (max_freq + ear_q*min_bw)
    cfreqs = np.flipud(cfreqs)
    GTord = 4.0
    ucirc = np.exp(2j * np.pi * np.arange(0, nfft//2+1, 1)/nfft)

    for i in range(nfilts):
        cf = cfreqs[i]
        erb = 1.0 * np.power((np.power(cf/ear_q, order) + min_bw ** order), 1.0/order)
        b = 1.019 * 2 * np.pi * erb
        r = np.exp(-b / sr)
        theta = 2 * np.pi * cf / sr
        pole = r * np.exp(1j * theta)

        t = 1. / sr

        a11 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) + 2 * np.sqrt(3 + 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2
        a12 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) - 2 * np.sqrt(3 + 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2
        a13 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) + 2 * np.sqrt(3 - 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2
        a14 = -(2 * t * np.cos(2 * cf * np.pi * t) / np.exp(b * t) - 2 * np.sqrt(3 - 2 ** 1.5) * t * np.sin(
            2 * cf * np.pi * t) / np.exp(b * t)) / 2

        zros = -1 * np.column_stack((a11, a12, a13, a14))/t
        p1 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p2 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) + np.sqrt(3 - 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p3 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p4 = (-2 * np.exp(4j * cf * np.pi * t) * t + 2 * np.exp(-(b * t) + 2j * cf * np.pi * t) * t *
              (np.cos(2 * cf * np.pi * t) + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(2 * cf * np.pi * t)))
        p5 = np.power(
            -2 / np.exp(2 * b * t) - 2 * np.exp(4j * cf * np.pi * t) + 2 * (1 + np.exp(4j * cf * np.pi * t)) / np.exp(
                b * t), 4)
        gain = np.abs(p1 * p2 * p3 * p4 / p5)

        wts[i, :] = ((t ** 4) / gain) * np.abs(ucirc - zros[:, 0]) * np.abs(ucirc - zros[:, 1]) * \
                    np.abs(ucirc - zros[:, 2]) * np.abs(ucirc - zros[:, 3]) * \
                    np.power(np.abs((pole - ucirc) * (np.conj(pole) - ucirc)), -1*GTord)

    return wts

if __name__ == '__main__':
    wav_data, wav_header = read_sphere_wav(u'sa1.wav')
    sr = 16000
    #wav_data, wav_header = read_sphere_wav(u"/media/neo/000C6F0F00042510/Doctor/dataset/TIMIT/train/dr1/fcjf0/sa1.wav")
    cochlea = cochleagram_extractor(wav_data, sr, 320, 160, 64, 'hanning')
    plt.imshow(np.sqrt(np.flipud(cochlea)))
    plt.show()
    fft2gammatone_coef = cochleagram_fft_coefs(sr, 320, 64)
    spect = spectrum_extractor(wav_data, 320, 160, 'hanning', False)
    plt.imshow(np.flipud(np.sqrt(np.matmul(fft2gammatone_coef, spect))))
    plt.show()
    # min = fft2gammatone_coef.min()
    # max = fft2gammatone_coef.max()
    # color_map = cm.RdYlGn
    # scalarMap = cm.ScalarMappable(norm=Normalize(vmin=min, vmax=max), cmap=color_map)
    # C_colored = scalarMap.to_rgba(fft2gammatone_coef)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.arange(1, 162, 1)
    # y = np.arange(1, 65, 1)
    # X, Y = np.meshgrid(x, y)
    # ax.plot_surface(X, Y, fft2gammatone_coef, rstride=1, cstride=1, facecolors=C_colored, antialiased=True)
    # for i in range(10):
    #     plt.hold(True)
    #     plt.plot(fft2gammatone_coef[i, :])
    # plt.subplot(211)
    # plt.imshow(fft2gammatone_coef)
    # plt.subplot(212)
    # plt.hold(True)
    # for i in range(24):
    #     plt.plot(fft2gammatone_coef[40+i,:])
    #
    # plt.show()


