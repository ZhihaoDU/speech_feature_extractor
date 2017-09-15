# coding = utf-8

import numpy as np
import ctypes


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
            if len(tmp) < 3 and len(tmp) > 0:
                sphere_header['Name'] = tmp[0]
            elif len(tmp[0]) > 0:
                sphere_header[tmp[0]] = tmp[2]

    return wav_data, sphere_header


if __name__ == '__main__':
    wav_data, wav_header = read_sphere_wav(u"/media/neo/000C6F0F00042510/Doctor/dataset/TIMIT/train/dr1/fcjf0/sa1.wav")
    print(wav_data.shape)
    print(wav_header)
    print(wav_data[0:100])
    print(wav_data.min())
    print(wav_data.max())


