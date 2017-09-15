import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.io import wavfile


def calc_normalized_autocorrelation(x, win_len, shift_len, Tn):
    frame_number = (len(x) - win_len) // shift_len
    A = np.zeros(shape=(win_len // Tn, frame_number))
    for i in range(frame_number):
        one_frame = x[i*shift_len: i*shift_len+win_len]
        for t in range(1, win_len // Tn-1):
            n = np.arange(t*Tn, win_len, Tn)
            A[t, i] = np.sum(one_frame[n]*one_frame[n - t*Tn]) / (norm(one_frame[n]) * norm(one_frame[n - t*Tn]))
    return A


if __name__ == '__main__':
    sr, wav_data = wavfile.read("clean.wav")
    A = calc_normalized_autocorrelation(wav_data, int(sr * 0.02), int(sr * 0.01), 2)
    plt.figure()
    plt.subplot(311)
    plt.imshow(A[int(2.5*sr/1000):int(15.*sr/1000), :])
    plt.subplot(312)
    Ats = np.max(A[int(2.5*sr/1000):int(15.*sr/1000), :], 0)
    plt.plot(Ats)
    ts = np.argmax(A[int(2.5 * sr / 1000):int(15. * sr / 1000), :], 0)
    plt.subplot(313)
    plt.plot(1./(ts*1./sr))
    plt.show()
