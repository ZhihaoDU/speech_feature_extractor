# coding = utf-8

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import ctypes
from feature_extractor import *
import subprocess, time

EPSILON = np.finfo(np.float32).eps


def SNR(x1, x2):
    return 10 * np.log10(np.sum(x1 ** 2.) / np.sum(x2 ** 2.))


def write_wav_to_file(wav, sr, file_path):
    scalar = (np.iinfo(np.int16).max / 4.) / np.max(np.abs(wav))
    wavfile.write(file_path, sr, (wav * scalar).astype(np.int16))


def random_mix_speech_noise(clean_speech, noise, snr, noise_from, noise_to):
    from numpy.linalg import norm
    to_mix_speech = np.array(clean_speech)
    to_mix_noise = np.array(noise[noise_from: noise_to])
    if len(clean_speech) < (noise_to - noise_from):
        noise_start = np.random.randint(noise_from, noise_to - len(clean_speech))
        to_mix_noise = noise[noise_start: noise_start+len(clean_speech)]
        to_mix_speech = clean_speech
    elif len(clean_speech) > (noise_to - noise_from):
        segs = len(clean_speech) // (noise_to - noise_from)
        to_mix_noise[:(segs-1)*noise_to] = np.repeat(noise[noise_from: noise_to], segs)
        noise_start = np.random.randint(noise_from, noise_to - (len(clean_speech)-(segs-1)*(noise_to - noise_from)))
        to_mix_noise[(segs-1)*noise_to:] = noise[noise_start: noise_start+(len(clean_speech)-(segs-1)*(noise_to - noise_from))]

    # This calculate snr method will lead log(x+1) mismatch so much!!!!
    # to_mix_noise = to_mix_noise / norm(to_mix_noise)
    # to_mix_speech = to_mix_speech / norm(to_mix_speech) * np.sqrt(10.0 ** (0.1 * snr))
    to_mix_noise = to_mix_noise / norm(to_mix_noise) * norm(to_mix_speech) / np.sqrt(10.0 ** (0.1 * snr))
    check_snr = 10*np.log10(np.square(norm(to_mix_speech)/norm(to_mix_noise)))
    if abs(check_snr - snr) > 1e-6:
        print "FATAL ERROR: snr calculate error!!!!"
        exit(-1)
    mix = to_mix_noise + to_mix_speech
    return mix, to_mix_speech


def read_sphere_wav(file_name):
    wav_file = open(file_name, 'rb')
    raw_header = wav_file.read(1024).decode('utf-8')
    raw_data = wav_file.read()
    wav_file.close()
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


def ideal_binary_mask(noisy_speech, clean_speech, local_snr, win_len, shift_len, delta_size=0):
    coch_speech = cochleagram_extractor_wdl(clean_speech, 16000, win_len, shift_len, 64, 'ones')
    coch_noise = cochleagram_extractor_wdl(noisy_speech - clean_speech, 16000, win_len, shift_len, 64, 'ones')
    ibm = np.where(10.*np.log10(coch_speech/coch_noise) >= local_snr, 1., 0.)
    if delta_size > 0:
        ibm = ibm[:, delta_size: -delta_size]
    return ibm


def fft_mag(noisy_speech, clean_speech, win_len, shift_len, delta_siz, method='log_percent'):
    clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
    # log compress
    if method.find('log') > 0:
        clean_spect = np.log(np.square(np.abs(clean_spect)))
    # percent normalization
    if method.find('percent') > 0:
        clean_spect = (clean_spect - np.min(clean_spect, axis=1, keepdims=True)) / \
                      (np.max(clean_spect, axis=1, keepdims=True) - np.min(clean_spect, axis=1, keepdims=True))

    clean_spect = clean_spect[:, delta_siz : -delta_siz]
    return clean_spect

def fft_ibm(noisy_speech, clean_speech, win_len, shift_len, local_snr, delta_size, method='naive'):
    noise_spect = stft_extractor(noisy_speech-clean_speech, win_len, shift_len, 'hanning')
    clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
    ibm = np.where(10. * np.log10(clean_spect / noise_spect) >= local_snr, 1., 0.)
    if delta_size > 0:
        ibm = ibm[:, delta_size: -delta_size]
    return ibm

def fft_irm(noisy_speech, clean_speech, win_len, shift_len, delta_size, method='naive'):
    if method == 'naive':
        noise_spect = stft_extractor(noisy_speech-clean_speech, win_len, shift_len, 'hanning')
        clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
        mask = np.sqrt(np.square(np.abs(clean_spect)) / (np.square(np.abs(noise_spect)) + np.square(np.abs(clean_spect))))
        if delta_size > 0:
            mask = mask[:, delta_size: -delta_size]
        return mask
    if method == 'clip':
        noisy_spect = stft_extractor(noisy_speech, win_len, shift_len, 'hanning')
        clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
        mask = np.abs(clean_spect) / (np.abs(noisy_spect) + EPSILON)
        mask = mask[:, delta_size: -delta_size]
        mask = np.where(mask > 2., 2., mask)
        return mask
    if method == 'log_sigmoid':
        noisy_spect = stft_extractor(noisy_speech, win_len, shift_len, 'hanning')
        clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
        mask = np.abs(clean_spect) / (np.abs(noisy_spect) + EPSILON)
        mask = np.log(mask) + 1.25
        sigmoid_mask = 1. / (1. + np.exp(-1. * mask))
        return sigmoid_mask


def ideal_ratio_mask(noisy_speech, clean_speech, win_len, shift_len, delta_size, method='naive'):
    coch_speech = cochleagram_extractor_wdl(clean_speech, 16000, win_len, shift_len, 64, 'ones')
    coch_noise = cochleagram_extractor_wdl(noisy_speech - clean_speech, 16000, win_len, shift_len, 64, 'ones')
    # coch_noise = cochleagram_extractor_wdl(noisy_speech, 16000, win_len, shift_len, 64, 'ones')
    irm = np.sqrt(coch_speech / (coch_noise + coch_speech))
    if delta_size > 0:
        irm = irm[:, delta_size: -delta_size]
    return irm


def ideal_abs_complex_ratio(noisy_speech, clean_speech, win_len, shift_len, delta_size):
    noise_spect = stft_extractor(noisy_speech - clean_speech, win_len, shift_len, 'hanning')
    noise_spect = noise_spect[:, delta_size:-delta_size]
    clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
    clean_spect = clean_spect[:, delta_size:-delta_size]
    real_ratio = np.abs(np.real(clean_spect)) / (np.abs(np.real(noise_spect)) + np.abs(np.real(clean_spect)) + EPSILON)
    imag_ratio = np.abs(np.imag(clean_spect)) / (np.abs(np.imag(noise_spect)) + np.abs(np.imag(clean_spect)) + EPSILON)
    abs_complex_irm = np.concatenate([real_ratio, imag_ratio], axis=0)
    # complex_irm = 10 * (1 - np.exp(-0.1*complex_irm)) / (1 + np.exp(-0.1*complex_irm))
    return abs_complex_irm


def ideal_complex_ratio(noisy_speech, clean_speech, win_len, shift_len, delta_size=0, method='clip'):
    noise_spect = stft_extractor(noisy_speech - clean_speech, win_len, shift_len, 'hanning')
    noise_spect = noise_spect[:, delta_size:-delta_size]
    clean_spect = stft_extractor(clean_speech, win_len, shift_len, 'hanning')
    clean_spect = clean_spect[:, delta_size:-delta_size]
    if method == 'clip':
        real_ratio = 0.001 * np.real(clean_spect) / (np.real(noise_spect) + np.real(clean_spect) + EPSILON)
        real_ratio = np.where(real_ratio > 100., 100., real_ratio)
        real_ratio = np.where(real_ratio < -100., -100., real_ratio)
        imag_ratio = 0.001 * np.imag(clean_spect) / (np.imag(noise_spect) + np.imag(clean_spect) + EPSILON)
        imag_ratio = np.where(imag_ratio > 100., 100. , imag_ratio)
        imag_ratio = np.where(imag_ratio < -100., 100., imag_ratio)
        complex_irm = np.concatenate([real_ratio, imag_ratio], axis=0)
        complex_irm = 1. / (1. + np.exp(-1. * complex_irm))
        return complex_irm
    if method == 'log':
        tmp = np.real(noise_spect) + np.real(clean_spect)
        tmp = np.where(np.sign(tmp) == 0., EPSILON, tmp)
        tmp = np.sign(tmp) * (np.abs(tmp) + EPSILON)
        real_ratio = np.real(clean_spect) / tmp
        real_ratio = np.sign(real_ratio) * np.log(np.abs(real_ratio) + 1.)

        tmp = np.imag(noise_spect) + np.imag(clean_spect)
        tmp = np.where(np.sign(tmp) == 0., EPSILON, tmp)
        tmp = np.sign(tmp) * (np.abs(tmp) + EPSILON)
        imag_ratio = np.imag(clean_spect) / tmp
        imag_ratio = np.sign(imag_ratio) * np.log(np.abs(imag_ratio) + 1.)
        complex_irm = np.concatenate([real_ratio, imag_ratio], axis=0)
        complex_irm = 10. * (1. - np.exp(-1. * complex_irm)) / (1. + np.exp(-1. * complex_irm))
        return complex_irm

def expand_frame(frames, extend_number):
    feature_dim = frames.shape[0]
    frame_width = 2 * extend_number + 1
    extended_frames = np.zeros([frames.shape[1] - frame_width + 1, feature_dim, frame_width], dtype=np.float32)
    frame_number = frames.shape[1]
    for i in range(frame_width):
        extended_frames[:, :, i] = np.transpose(frames[:, i: frame_number-frame_width+i+1])
    return extended_frames

def istft(stft, win_type, win_len, shift_len, syn_method='A&R'):
    frames = stft.shape[1]
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
        masked_abs = np.abs(stft[:win_len // 2 + 1, i])
        to_ifft[:win_len // 2 + 1] = masked_abs * np.exp(1j * np.angle(stft[:win_len // 2 + 1, i]))
        to_ifft[win_len // 2 + 1:] = np.conj(to_ifft[win_len // 2 - 1:0:-1])
        speech_seg = np.real(np.fft.ifft(to_ifft, win_len))

        if syn_method == 'A&R' or syn_method == 'ALLEN & RABINER':
            clean_speech[i * shift_len:i * shift_len + win_len] += speech_seg
            window_sum[i * shift_len:i * shift_len + win_len] += window

        elif syn_method == 'G&L' or syn_method == 'GRIFFIN & LIM':
            speech_seg = np.multiply(speech_seg, window)
            clean_speech[i * shift_len:i * shift_len + win_len] += speech_seg
            window_sum[i * shift_len:i * shift_len + win_len] += np.power(window, 2.)
    window_sum = np.where(window_sum < 1e-2, 1e-2, window_sum)
    return clean_speech / window_sum


def synthesis_speech(noisy_speech, ideal_mask, win_type, win_len, shift_len, frame_offset=0, syn_method='A&R'):
    samples = noisy_speech.shape[0]
    frames = ideal_mask.shape[1]
    result_speech = np.zeros_like(noisy_speech)

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
        one_frame = noisy_speech[(i+frame_offset)*shift_len: (i+frame_offset)*shift_len+win_len]
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
    result_speech[frame_offset*shift_len: frame_offset*shift_len+len(clean_speech)] = clean_speech / window_sum
    return result_speech
    # return clean_speech / window_sum

def mag_synthesis_speech(noisy_speech, spect_mag, win_type, win_len, shift_len, frame_offset=0, syn_method='A&R'):
    frames = spect_mag.shape[1]
    result_speech = np.zeros_like(noisy_speech)

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
        one_frame = noisy_speech[(i+frame_offset)*shift_len: (i+frame_offset)*shift_len+win_len]
        windowed_frame = np.multiply(one_frame, window)
        stft = np.fft.fft(windowed_frame, win_len)
        masked_mag = spect_mag[:, i]
        to_ifft[:win_len//2+1] = masked_mag * np.exp(1j * np.angle(stft[:win_len//2+1]))
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
    result_speech[frame_offset*shift_len: frame_offset*shift_len+len(clean_speech)] = clean_speech / window_sum
    return result_speech

def coch_synthesis(noisy, mask, sr, win_len, shift_len, frame_offset):
    low_fq = 50
    r = np.zeros_like(noisy)
    [channel, frames] = mask.shape
    coswin = (1. + np.cos(2. * np.pi * np.arange(0., win_len, 1.) / win_len - np.pi)) / 2.
    fcoefs, f = make_erb_filters(sr, channel, low_fq)
    fcoefs = np.flipud(fcoefs)
    x = erb_frilter_bank(noisy, fcoefs)

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

    for chan in range(np.size(gain, 0)):
        y1 = lfilter(np.array([a0[chan] / gain[chan], a11[chan] / gain[chan], a2[chan] / gain[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), x[chan, ::-1])
        y2 = lfilter(np.array([a0[chan], a12[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y1)
        y3 = lfilter(np.array([a0[chan], a13[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y2)
        y4 = lfilter(np.array([a0[chan], a14[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y3)

        y4 = y4[::-1]
        weight = np.zeros_like(noisy)
        for f in range(frames):
            weight[(f+frame_offset)*shift_len: (f+frame_offset)*shift_len+win_len] += mask[chan, f] * coswin
        r += y4 * weight
    return r


def get_pesq_mos(speech, degraded, dlabel):
    rand_number = np.random.randint(0, 100)
    if len(speech) != len(degraded):
        print "Fatal Error: the length is mismatch: speech: %d, degraded: %d" % (len(speech), len(degraded))
        exit(-1)
    write_wav_to_file(speech, 16000, "/data/duzhihao/tmp/%s/clean_%d.wav" % (dlabel, rand_number))
    write_wav_to_file(degraded, 16000, "/data/duzhihao/tmp/%s/degraded_%d.wav" % (dlabel, rand_number))
    # your pesq.exe path here.
    pesq = "/data/duzhihao/pesq/PESQ"
    output = subprocess.Popen([pesq, "+16000", "/data/duzhihao/tmp/%s/clean_%d.wav" % (dlabel, rand_number), "/data/duzhihao/tmp/%s/degraded_%d.wav" % (dlabel, rand_number)], stdout=subprocess.PIPE).communicate()[0]
    try:
        pesq_mos = float(output.split(' ')[-1])
    except:
        print "Fatal Error: the output of pesq is:", output
        return None
    return pesq_mos


'''
          |    old    |    new
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
        fcenter = 2. ** (np.arange(-19, 15, 1) / 3.) * (10 ** 3)
    else:
        n = np.ceil(3. * np.log2(max_freq / min_freq)) + 1.
        fcenter = min_freq * 2 ** (1./3. * np.arange(-1, n+1, 1))
    n = len(fcenter)
    flower = np.sqrt(fcenter[:n-2] * fcenter[1:n-1])
    fupper = np.sqrt(fcenter[1:n-1] * fcenter[2:n])
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


def get_fft_octave_mat(nfft, sr, min_freq, max_freq):
    fft_bins = np.linspace(0, sr, nfft+1)
    fft_bins = fft_bins[:nfft//2+1]
    octave_lower, octave_upper = get_one_third_octave_bands(min_freq, max_freq)
    wts = np.zeros((len(octave_lower), len(fft_bins)), dtype=np.float32)
    for i in range(len(octave_lower)):
        left = np.argmin(np.square(fft_bins - octave_lower[i]))
        right = np.argmin(np.square(fft_bins - octave_upper[i]))
        wts[i, left:right] = 1.
    return wts
'''
The result is same as up one.
def get_fft_octave_mat(nfft, sr, min_freq, max_freq):
    fft_bins = np.linspace(0, sr, nfft+1)
    fft_bins = fft_bins[:nfft//2+2]
    fft_center_freq = (fft_bins[:-1] + fft_bins[1:]) / 2.
    octave_lower, octave_upper = get_one_third_octave_bands(min_freq, max_freq)
    wts = np.zeros((len(octave_lower), len(fft_center_freq)), dtype=np.float32)
    for i in range(len(octave_lower)):
        for j in range(len(fft_center_freq)):
            if octave_lower[i] < fft_center_freq[j] < octave_upper[i]:
                wts[i, j] = 1.
    return wts
'''

def vad_mask(clean_speeh, dyn_range, win_len, shift_len):
    win = get_window(win_len, 'hanning')
    frames = (len(clean_speeh) - win_len) // shift_len + 1
    mask = np.zeros([frames], dtype=np.float32)
    for i in range(frames):
        one_frame = clean_speeh[i * shift_len: i * shift_len + win_len]
        mask[i] = 20 * np.log10(np.linalg.norm(one_frame * win) / np.sqrt(win_len))
    mask = np.where((mask - np.max(mask) + dyn_range) > 0, 1, 0)
    return mask.reshape([1, frames])


def remove_silence_frames(x, y, dyn_range, win_len, shift_len):
    win = get_window(win_len, 'hanning')
    frames = (len(x) - win_len) // shift_len + 1
    mask = np.zeros([frames], dtype=np.float32)
    for i in range(frames):
        one_frame = x[i*shift_len: i*shift_len+win_len]
        mask[i] = 20 * np.log10(np.linalg.norm(one_frame * win) / np.sqrt(win_len))
    mask = np.where((mask - np.max(mask) + dyn_range) > 0, 1, 0)
    x_out = np.zeros_like(x)
    y_out = np.zeros_like(y)
    count = 0
    for i in range(frames):
        if mask[i] == 1:
            x_out[count*shift_len: count*shift_len+win_len] += x[i*shift_len: i*shift_len+win_len] * win
            y_out[count*shift_len: count*shift_len+win_len] += y[i*shift_len: i*shift_len+win_len] * win
            count += 1
    x_out = x_out[:(count-1)*shift_len+win_len]
    y_out = y_out[:(count-1)*shift_len+win_len]
    return x_out, y_out


def my_resample(x, origin_fs, target_fs):
    from scipy.signal import resample
    origin_l = len(x)
    target_l = np.int32((1. * origin_l / origin_fs * target_fs))
    xx = resample(x, target_l)
    return xx


def taa_corr(x, y):
    xn = x - np.mean(x)
    xn = xn / np.sqrt(np.sum(np.square(xn)))
    yn = y - np.mean(y)
    yn = yn / np.sqrt(np.sum(np.square(yn)))
    rho = sum(xn * yn)
    return rho

def calc_stoi(speech, degraded, fs, analysis_len=30):
    if len(speech) != len(degraded):
        return 0.
    if fs != 10000:
        speech = my_resample(speech, fs, 10000)
        degraded = my_resample(degraded, fs, 10000)
    speech, degraded = remove_silence_frames(speech, degraded, 40, 256, 128)
    wts = get_fft_octave_mat(512, 10000, 150, 5000)
    wts = wts[:15, :]
    speech_spect = stft_extractor(speech, 256, 128, 'hanning', 512)
    degraded_spect = stft_extractor(degraded, 256, 128, 'hanning', 512)
    X = np.sqrt(np.matmul(wts, np.square(np.abs(speech_spect))))
    Y = np.sqrt(np.matmul(wts, np.square(np.abs(degraded_spect))))
    c = 10. ** (15./20.)
    d_interm = np.zeros(shape=[X.shape[0], X.shape[1]-analysis_len])
    for i in range(analysis_len, X.shape[1]):
        X_seg = X[:, i-analysis_len: i]
        Y_seg = Y[:, i-analysis_len: i]
        alpha = np.sqrt(np.sum(np.square(X_seg), 1, keepdims=True) / np.sum(np.square(Y_seg), 1, keepdims=True))
        aY_seg = Y_seg * alpha
        for j in range(X.shape[0]):
            Y_prime = np.minimum(aY_seg[j, :], X_seg[j,:]+X_seg[j, :]*c)
            d_interm[j, i-analysis_len] = taa_corr(X_seg[j, :], Y_prime)
    return np.mean(d_interm)

def print_with_time(str):
    str_time = time.strftime('%Y-%m-%d %X', time.localtime())
    print "%s %s" % (str_time, str)


if __name__ == '__main__':
    fs, speech = wavfile.read('synthesis_test/synthesis_speech.wav')
    fs, degraded = wavfile.read('synthesis_test/noisy.wav')
    fs, ibm_speech = wavfile.read('synthesis_test/ibm_coch_synthesis.wav')
    fs, irm_speech = wavfile.read('synthesis_test/irm_coch_synthesis.wav')
    from numpy.linalg import norm
    print 'noisy stio: %.6f' % calc_stoi(speech / norm(speech), degraded / norm(degraded), fs)
    print 'IBM stio: %.6f' % calc_stoi(speech / norm(speech), ibm_speech / norm(ibm_speech), fs)
    print 'IRM stio: %.6f' % calc_stoi(speech / norm(speech), irm_speech / norm(irm_speech), fs)
