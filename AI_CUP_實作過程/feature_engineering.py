import math
import numpy as np
import csv
from pathlib import Path
import os

def FFT(xreal, ximag):
    n = 2
    while(n*2 <= len(xreal)):
        n *= 2
    p = int(math.log(n, 2))
    for i in range(0, n):
        a = i
        b = 0
        for j in range(0, p):
            b = int(b*2 + a%2)
            a = a/2
        if(b > i):
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]
    wreal = []
    wimag = []
    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))
    wreal.append(float(1.0))
    wimag.append(float(0.0))
    for j in range(1, int(n/2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)
    m = 2
    while(m < n + 1):
        for k in range(0, n, m):
            for j in range(0, int(m/2), 1):
                index1 = k + j
                index2 = int(index1 + m / 2)
                t = int(n * j / m)
                treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
                timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
                ureal = xreal[index1]
                uimag = ximag[index1]
                xreal[index1] = ureal + treal
                ximag[index1] = uimag + timag
                xreal[index2] = ureal - treal
                ximag[index2] = uimag - timag
        m *= 2
    return n, xreal, ximag

def FFT_data(input_data, swinging_times):
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength
    for num in range(len(swinging_times)-1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num+1]):
            a.append(math.sqrt(max(0, math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2))))
            g.append(math.sqrt(max(0, math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2))))
        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(g) / len(g))
    return a_mean, g_mean

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    allsum = []
    mean = []
    var = []
    rms = []
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    for i in range(len(input_data)):
        if i==0:
            allsum = input_data[i]
            a.append(math.sqrt(max(0, math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2))))
            g.append(math.sqrt(max(0, math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2))))
            continue
        a.append(math.sqrt(max(0, math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2))))
        g.append(math.sqrt(max(0, math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2))))
        allsum = [allsum[feature_index] + input_data[i][feature_index] for feature_index in range(len(input_data[i]))]
    mean = [allsum[feature_index] / len(input_data) for feature_index in range(len(input_data[i]))]
    for i in range(len(input_data)):
        if i==0:
            var = input_data[i]
            rms = input_data[i]
            continue
        var = [var[feature_index] + math.pow((input_data[i][feature_index] - mean[feature_index]), 2) for feature_index in range(len(input_data[i]))]
        rms = [rms[feature_index] + math.pow(input_data[i][feature_index], 2) for feature_index in range(len(input_data[i]))]
    var = [math.sqrt(max(0, var[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    rms = [math.sqrt(max(0, rms[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    a_var = math.sqrt(max(0, math.pow((var[0] + var[1] + var[2]), 2)))
    for i in range(len(input_data)):
        a_s1 = a_s1 + math.pow((a[i] - a_mean[0]), 4)
        a_s2 = a_s2 + math.pow((a[i] - a_mean[0]), 2)
        g_s1 = g_s1 + math.pow((g[i] - g_mean[0]), 4)
        g_s2 = g_s2 + math.pow((g[i] - g_mean[0]), 2)
        a_k1 = a_k1 + math.pow((a[i] - a_mean[0]), 3)
        g_k1 = g_k1 + math.pow((g[i] - g_mean[0]), 3)
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    a_kurtosis = [a_s1 / (a_s2 + 1e-10)]
    g_kurtosis = [g_s1 / (g_s2 + 1e-10)]
    a_skewness = [a_k1 / (a_k2 + 1e-10)]
    g_skewness = [g_k1 / (g_k2 + 1e-10)]
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.sqrt(max(0, a_psd[-1])))
        e3.append(math.sqrt(max(0, g_psd[-1])))
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    for i in range(cut):
        e2 += math.sqrt(max(0, a_psd[i]))
        e4 += math.sqrt(max(0, g_psd[i]))
    for i in range(cut):
        val_a = e1[i] / (e2 + 1e-10)
        val_a = max(1e-10, val_a)
        entropy_a.append(val_a * math.log(val_a))
        val_g = e3[i] / (e4 + 1e-10)
        val_g = max(1e-10, val_g)
        entropy_g.append(val_g * math.log(val_g))
    a_entropy_mean = sum(entropy_a) / len(entropy_a)
    g_entropy_mean = sum(entropy_g) / len(entropy_g)
    output = mean + var + rms + a_max + a_mean + a_min + g_max + g_mean + g_min + [a_fft_mean] + [g_fft_mean] + [a_psd_mean] + [g_psd_mean] + a_kurtosis + g_kurtosis + a_skewness + g_skewness + [a_entropy_mean] + [g_entropy_mean]
    writer.writerow(output)

def data_generate(datapath, tar_dir):
    os.makedirs(tar_dir, exist_ok=True)
    pathlist_txt = Path(datapath).glob('**/*.txt')
    for file in pathlist_txt:
        with open(file) as f:
            All_data = []
            count = 0
            for line in f.readlines():
                if line == '\n' or count == 0:
                    count += 1
                    continue
                num = line.split(' ')
                if len(num) > 5:
                    tmp_list = []
                    for i in range(6):
                        tmp_list.append(int(num[i]))
                    All_data.append(tmp_list)
        swing_index = np.linspace(0, len(All_data), 28, dtype=int)
        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']
        with open('./{dir}/{fname}.csv'.format(dir=tar_dir, fname=Path(file).stem), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i == 0:
                        continue
                    feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
            except Exception as e:
                print('錯誤於:', Path(file).stem, str(e))
                continue
data_generate('./train_data', 'tabular_data_train')
data_generate('./test_data', 'tabular_data_test')
    