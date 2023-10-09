import numpy as np
import scipy
import matplotlib.pyplot as plt
import loradecode
from rtlsdr import RtlSdr

def specto(sig, bins, graph):
    testsignal = sig
    fft_size = 2**bins
    num_rows = int(np.floor(len(testsignal)/fft_size))
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(testsignal[i*fft_size:(i+1)*fft_size])))**2)

    if graph == True:
        plt.imshow(spectrogram, aspect='auto', extent = [0, fft_size, len(testsignal), 0])
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        plt.show()

    return spectrogram

def specto_custom(sig, bins, graph, rowdivis):
    testsignal = sig
    fft_size = 2**bins
    num_rows = int(np.floor(len(testsignal)/rowdivis))
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(testsignal[i*fft_size:(i+1)*fft_size], fft_size)))**2)

    if graph == True:
        plt.imshow(spectrogram, aspect='auto', extent = [0, fft_size, num_rows, 0])
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Time [s]")
        plt.show()

    return spectrogram

def init_radio(Fs, frf):
    sdr = RtlSdr()
    sdr.sample_rate = Fs
    sdr.center_freq = frf     
    sdr.gain = 'auto'
    return sdr

def get_radio_samples(sdr, samplenum):
    sm = sdr.read_samples(samplenum)
    return sm

def detect_lora_signal(sm, Fs, sf, airtime):
    spectrogram = specto(sm, sf, graph = False)
    fft_size = 2**sf
    num_rows = int(np.floor(len(sm)/fft_size))
    
    ilst = []
    binlst = []
    for i in range(num_rows):
        rowmax = np.max(spectrogram[i,:])
        if rowmax > 30:
            ilst.append(i)
            symbin = np.argmax(spectrogram[i,:])
            binlst.append(symbin)

    try:
        smlora = sm[ilst[0]*fft_size:ilst[-1]*fft_size]

        loraairtime = get_time_from_samples(smlora, Fs)
        if loraairtime > airtime:
            smlora = sm[ilst[0]*fft_size - 5000:ilst[-1]*fft_size + 5000]
            return smlora
    except:
        pass

    return np.array([])

def get_time_from_samples(arr, samplerate):
    if type(arr) != list:
        arrlength = arr.size
    else:
        arrlength = len(arr)

    return arrlength / samplerate

def close_radio(sdr):
    sdr.close

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def dechirp(samples, chirptype, x, binnum, samplenum, fftlen):
    ft = np.fft.fft(samples[x : x + samplenum] * chirptype, fftlen)
    ft_ = abs(ft[1: binnum]) + abs(ft[fftlen - binnum + 1 : fftlen])
    pk = np.argmax(10*np.log(ft_))
    return pk

def dechirp_normal(samples, chirptype, x, binnum, samplenum, fftlen, graph = None):
    ft = np.fft.fft(samples[x : x + samplenum] * chirptype, fftlen)
    pk = np.argmax(10*np.log(ft))
    if graph != None:
        plt.plot(ft)
        plt.show()
    return pk

def dechirp_pow(samples, chirptype, x, binnum, samplenum, fftlen, fftplot = None):
    try:
        ft = np.fft.fft(samples[x : x + samplenum] * chirptype, fftlen)
        ft_ = abs(ft[1: binnum]) + abs(ft[fftlen - binnum + 1 : fftlen])
        ft_ = moving_average(ft_, 50)
        ft_log = 10*np.log(ft_)
        pkpow = np.max(ft_log)
        pksum = np.sum(ft_log) / ft_log.size


        if fftplot != None:
            plt.plot(ft_log)
            plt.show()
        #if no real peak return zero 
        if pkpow - 5 > pksum:
            return  pkpow

        return 0

    except:
        return 0

def generate_chirp(N, bw):
    headersamplenum = N
    bwchirp = bw/2
    t = np.linspace(0, headersamplenum/bw, int(headersamplenum))
    i = scipy.signal.chirp(t, f0 = -bwchirp, f1 = bwchirp, t1 = headersamplenum/bw, phi = 90)
    q = scipy.signal.chirp(t, f0 = -bwchirp, f1 = bwchirp, t1 = headersamplenum/bw, phi = 0)

    iq = i + 1j * q

    dwnchp = np.conj(iq)
    return dwnchp

def dynamic_compensation(syms, cfo, frf, sf):
    symlen = np.array(range(1, len(syms) + 1))
    sfodrift = (1 + symlen) * 2**sf * cfo / frf
    symbols = (syms - sfodrift) % 2**sf
    return symbols

Fs = 250000
sf = 8
bw = 125000
frf = 915e6
airtime = .06

zero_padding_ratio = 10
N = 2 ** sf

sdr = init_radio(Fs, frf)
smlora = np.array([])
while smlora.size == 0:
    sm = get_radio_samples(sdr, 200000)
    smlora = detect_lora_signal(sm, Fs, sf, airtime)


close_radio(sdr)
smlora = scipy.signal.decimate(smlora, 2)
specto(smlora, sf-3, graph = True)


bwchirp = bw/2
t = np.linspace(0, N/bw, int(N))
i = scipy.signal.chirp(t, f0 = -bwchirp, f1 = bwchirp, t1 = N/bw, phi = 90)
q = scipy.signal.chirp(t, f0 = -bwchirp, f1 = bwchirp, t1 = N/bw, phi = 0)

iq = i + 1j * q

dwnchp = np.conj(iq)

upchp = iq


ii = 0
preamble_len = 5
pk_bin_list = []
while ii < smlora.size - N * preamble_len:
    if len(pk_bin_list) == preamble_len:
        x = ii - round(pk_bin_list[-1]/20)
        break

    pk0 = dechirp(smlora, dwnchp, ii, N, N, N)

    if pk_bin_list != []:
        bin_diff = (pk_bin_list[-1] - pk0) % N
        if bin_diff > N/2:
            bin_diff = N - bin_diff
        if bin_diff <= zero_padding_ratio:
            pk_bin_list.append(pk0)
        else:
            pk_bin_list = []
            pk_bin_list.append(pk0)
    else:
        pk_bin_list.append(pk0)

    ii = ii + N

#sync
notsync = True
while notsync == True:
    while x < smlora.size - N:
        smlorademod = smlora[x : x + N]
        lorafft = 10*np.log10(np.abs(np.fft.fft(smlorademod * dwnchp * np.blackman(N))**2))
        upmax = np.max(lorafft)
        up_peak = np.argmax(lorafft)
        upmean = np.mean(lorafft)
        #print('Upmax: ' + str(upmax) + ' Upmean: ' + str(upmean) + ' Sample Num: ' + str(x))
        #plt.plot(lorafft)
        #plt.show()

        smlorademod = smlora[x : x + N]
        lorafft = 10*np.log10(np.abs(np.fft.fft(smlorademod * upchp * np.blackman(N))**2))
        dwnmax = np.max(lorafft)
        down_peak = np.argmax(lorafft)
        dwnmean = np.mean(lorafft)
        #print('Downmax: ' + str(dwnmax) + ' Downmean: ' + str(dwnmean) + ' Sample Num: ' + str(x))
        #plt.plot(lorafft)
        #plt.show()

        #if down_peak > up_peak and dwnmax > upmax:
        if dwnmax - 10 > dwnmean and upmean > upmax - 10:
            if x < ii + (preamble_len + 6) * N: 
                notsync = False
                break
        x += N

    if notsync == True:
        print('Changing Window')
        ii += 10
        x = ii


xsecondsync = x + N
notsync = True
while notsync == True:
    smlorademod = smlora[xsecondsync : xsecondsync + N]
    lorafft = 10*np.log10(np.abs(np.fft.fft(smlorademod * dwnchp * np.blackman(N))**2))
    up_peak = np.argmax(lorafft)
    upmax = np.max(lorafft)

    smlorademod = smlora[xsecondsync : xsecondsync + N]
    lorafft = 10*np.log10(np.abs(np.fft.fft(smlorademod * upchp * np.blackman(N))**2))
    down_peak = np.argmax(lorafft)
    dwnmax = np.max(lorafft)

    if down_peak > up_peak and dwnmax > upmax:
        notsync = False  

    if notsync == True:
        xsecondsync -= 10
        print('Aligning Section Sync Symbol')

x = xsecondsync - N
print(x + 2.25 * N)

###cfo
pk = dechirp_normal(smlora, dwnchp, x - 4 * N, N, N, N)
preamble_bin = pk

if preamble_bin > N / 2:
    cfo = (preamble_bin - N - 1) * bw / N
else:
    cfo = (preamble_bin - 1) * bw / N

# start of payload is 9.25 symbols from start
Data_frame_st = int(x + 2.25 * N) 

num_of_symbs = 16
symbarr = []
#getting symbols
for i in range(num_of_symbs):
    try:
        pk = dechirp_normal(smlora, dwnchp, Data_frame_st, N, N, N)
        symbarr.append(pk)

        #specto(smlorademod, 9)

    except:
        pass
    
    Data_frame_st += N
        
#17, 13, 509, 33, 1, 73, 485, 286, 306, 410, 57, 167, 446, 401, 338, 251, 155, 64
symbadjust = preamble_bin 

symlst = []
for s in symbarr:
    symlst.append((s + N - symbadjust - 1) % N)

symbols = dynamic_compensation(symlst, cfo, frf, sf)
symbols = list(np.mod(np.round(symbols), 2**sf)) 
print(symlst)
print(symbols)

headerinfo = loradecode.lora_decode_header(symbols[0:8], sf)
print('Coding Rate: ' + str(headerinfo[0]))
print('Payload Length: ' + str(headerinfo[1]))

decodesym = loradecode.lora_decoder(symbols, sf)
print(decodesym)

try:
    if decodesym != []:
        for d in decodesym:
            print(chr(int(d)))
except:
    print('Failed')

