# coding = utf-8


import numpy as np
from scipy.io import wavfile
import ctypes
from feature_extractor import stft_extractor


def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))


def mix_by_db(x1, x2, snr, handle_method):
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


def read_sphere_wav(file_name):
    wav_file = open(file_name, 'rb')
    raw_header = wav_file.read(1024).decode('utf-8')
    raw_data = wav_file.read()
    sample_count = len(raw_data) // 2

    wav_data = np.zeros(shape=[sample_count], dtype=np.int32)

    for i in range(sample_count):
        wav_data[i] = ctypes.c_int16(ord(raw_data[2 * i + 1]) << 8).value + ctypes.c_int16(ord(raw_data[2 * i])).value

    header_list = raw_header.split("\n")
    sphere_header = {}
    for s in header_list:
        if len(s) > 0 and s != "end_head":
            tmp = s.split(" ")
            if 0 < len(tmp) < 3:
                sphere_header['Name'] = tmp[0]
            elif len(tmp[0]) > 0:
                sphere_header[tmp[0]] = tmp[2]

    return wav_data, sphere_header


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


def ideal_binary_mask(noisy_speech, clean_speech, snr, sr):
    win_len = sr * 0.02
    shift_len = sr * 0.01
    noisy_spect = stft_extractor(noisy_speech, win_len, shift_len, 'hanning')
    clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
    spect_snr = np.power(np.abs(clean_spect), 2.0) / np.power(np.abs(noisy_spect - clean_spect), 2.0)
    ibm = np.where(spect_snr > 10**(0.1*snr), 1, 0)

    return ibm


def ideal_ratio_mask(noisy_speech, clean_speech, sr):
    win_len = sr * 0.02
    shift_len = sr * 0.01
    noisy_spect = stft_extractor(noisy_speech, win_len, shift_len, 'hanning')
    clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
    irm = np.abs(clean_spect) / np.abs(noisy_spect)

    return irm


def synthesis_speech(noisy_speech, ideal_mask, win_type, win_len, shift_len, syn_method='A&R'):
    samples = noisy_speech.shape[0]
    frames = (samples - win_len) // shift_len

    if win_type == 'hanning':
        window = np.hanning(win_len)
    elif win_type == 'hamming':
        window = np.hamming(win_len)
    elif win_type == 'rectangle':
        window = np.ones(win_len)
    to_ifft = np.zeros(win_len, dtype=np.complex64)
    clean_speech = np.zeros((frames-1)*shift_len+win_len, dtype=np.float32)
    window_sum = np.zeros((frames-1)*shift_len+win_len, dtype=np.float32)
    for i in range(frames):
        one_frame = noisy_speech[i * shift_len: i * shift_len + win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft = np.fft.fft(windowed_frame, win_len)
        masked_abs = np.abs(stft[:win_len//2+1]) * ideal_mask[:, i]
        to_ifft[:win_len//2+1] = masked_abs * np.exp(1j * np.angle(stft[:win_len//2+1]))
        to_ifft[win_len//2+1:] = np.conj(to_ifft[win_len//2-1:0:-1])
        speech_seg = np.real(np.fft.ifft(to_ifft, win_len))

        if syn_method == 'A&R' or syn_method == 'ALLEN & RABINER':
            clean_speech[i*shift_len:i*shift_len+win_len] += speech_seg
            window_sum[i*shift_len:i*shift_len+win_len] += window

        elif syn_method == 'G&L' or syn_method == 'GRIFFIN & LIM':
            speech_seg = np.multiply(speech_seg, window)
            clean_speech[i * shift_len:i * shift_len + win_len] += speech_seg
            window_sum[i * shift_len:i * shift_len + win_len] += np.power(window, 2.)
        # if i > 0:
        #     clean_speech[i*shift_len: (i-1)*shift_len+win_len] *= 0.5
    window_sum = np.where(window_sum < 1e-2, 1e-2, window_sum)
    return clean_speech / window_sum


'''
          |    old    |  new
----------+-----------+-----------
Say "old" |    TP     |     FP
Say "new" |    FN     |     TN

HIT rate            = TP/(TP+FN)
False Alarm rate    = FP/(FP+TN)

'''
def calc_decision_matrix(predict_y, grand_truth):
    TP = np.sum(np.sum(np.where(predict_y == 1 and grand_truth == 1, 1, 0)))  # true positive
    FP = np.sum(np.sum(np.where(predict_y == 1 and grand_truth == 0, 1, 0)))  # false positive
    FN = np.sum(np.sum(np.where(predict_y == 0 and grand_truth == 1, 1, 0)))  # false negative
    TN = np.sum(np.sum(np.where(predict_y == 0 and grand_truth == 0, 1, 0)))  # true negative
    return TP, FP, TN, FN


def calc_hit_rate(predict_ibm, ibm):
    TP, FP, TN, FN = calc_decision_matrix(predict_ibm, ibm)
    return TP / (TP + FN)


def calc_false_alarm(predict_ibm, ibm):
    TP, FP, TN, FN = calc_decision_matrix(predict_ibm, ibm)
    return FP / (FP + TN)


def get_one_third_octave_bands(min_freq=None, max_freq=None):
    if min_freq is None and max_freq is None:
        fcenter = 2. ** (np.arange(-18, 14, 1) / 3.) * (10 ** 3)
    else:
        n = np.ceil(3. * np.log2(max_freq / min_freq)) + 1.
        fcenter = min_freq * 2 ** (1./3.) * np.arange(0., n, 1.)
    fd = 2. ** (1. / 6.)
    fupper = fcenter * fd
    flower = fcenter / fd
    return flower, fupper


def get_octave_bands(min_freq=None, max_freq=None):
    if min_freq is None and max_freq is None:
        fcenter = 2. ** np.arange(-6, 5, 1) * (10 ** 3)
    else:
        n = np.ceil(np.log2(max_freq / min_freq)) + 1.
        fcenter = min_freq * 2. ** np.arange(0., n, 1.)
    fd = 2. ** (1. / 2.)
    fupper = fcenter * fd
    flower = fcenter / fd
    return flower, fupper


def calc_stoi_from_wav(clean_speech, degraded_speech):
    pass


def get_fft_octave_mat(nfft, sr, min_freq, max_freq):
    fft_bins = np.linspace(0, sr//2, nfft//2+1)
    fft_center_freq = (fft_bins[:-1] + fft_bins[1:]) / 2.
    octave_lower, octave_upper = get_one_third_octave_bands(min_freq, max_freq)
    wts = np.zeros((len(octave_lower), len(fft_center_freq)), dtype=np.float32)
    j = 0
    for i in range(len(fft_center_freq)):
        while not (octave_lower[j] < fft_center_freq[i] < octave_upper[j]):
            j += 1
        wts[j, i] = 1

    return wts


def calc_stoi_from_spec(clean_spec, degraded_spec, analysis_len=30):
    freq_bins = np.size(clean_spec, 0)
    frames = np.size(clean_spec, 1)
    x = np.zeros((freq_bins, frames - analysis_len + 1, analysis_len), dtype=np.float32)
    y = np.zeros((freq_bins, frames - analysis_len + 1, analysis_len), dtype=np.float32)
    for j in range(0, freq_bins):
        for m in range(analysis_len - 1, frames, 1):
            x[j, m] = clean_spec[j, m - analysis_len + 1:m + 1]
            y[j, m] = degraded_spec[j, m - analysis_len + 1:m + 1]
            y[j, m] = np.minimum(np.linalg.norm(x[j,m,:])/np.linalg.norm(y[j,m,:])*y[j,m,:],
                                 (1.+np.power(10., 15./20.))*x[j,m,:])  # y is normalized and clipped
    x_mean = np.mean(x, axis=(0, 1))
    y_mean = np.mean(y, axis=(0, 1))
    score = 0.
    for j in range(0, freq_bins):
        for m in range(analysis_len - 1, frames, 1):
            score += np.dot(x[j, m, :] - x_mean, y[j, m, :] - y_mean) / \
                     (np.linalg.norm(x[j, m, :] - x_mean) * np.linalg.norm(y[j, m, :] - y_mean))
    score /= (freq_bins * analysis_len)
    return score
