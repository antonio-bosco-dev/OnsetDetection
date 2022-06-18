########## NON MODIFICARE #######
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.io import wavfile
from scipy import signal
import librosa
import IPython.display as ipd
import pandas as pd
sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6



########################
#####LETTURA FILE#######
#########################
#Leggi file audio 16Bit mono
f_wav= 'drum.wav'
sr, X_int = wavfile.read(f_wav)
print ("sr :", sr)

# trasforma in formato "normale" tra -1.0 e 1.0
X = X_int/32768.0
print('max abs amp input:', max(abs(X)))
##############################


#########################
#####Funzioni Plot#######
#########################
def plot_wav_spectrogram(DataIn, sr, xlim=None, audio=True):

    plt.figure(figsize=(8))
    ax = plt.subplot(1,2,1)
    libfmp.b.plot_signal(DataIn, sr, ax=ax)
    if xlim!=None: plt.xlim(xlim)
    ax = plt.subplot(1,2,2)
    N, H = 512, 256
    X_stft = librosa.stft(DataIn, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Y_stft = np.log(1 + 10 * np.abs(X_stft))
    libfmp.b.plot_matrix(Y_stft, sr=sr/H, sr_F=N/sr, ax=[ax], colorbar=False)
    plt.ylim([0,5000])
    if xlim is not None: plt.xlim(xlim)
    plt.tight_layout()
    plt.show()
    if audio: ipd.display(ipd.Audio(DataIn, rate=sr))
    return

def disegna(Data, sr):
        """
        Input:
            myArray: np array a 64 bit

        Output:
            disegna il grafico dell'array
        """
        numSmpls = len(Data)

        #sintassi linspace start, stop, numero di punti
        # genera asse dei tempi
        Time = np.linspace(0, numSmpls/sr, numSmpls)

        # crea in memoria il disegno con asse x Time ed y myArray
        plt.plot(Time, Data)

        # aggiungi griglia
        plt.grid()

        # mostra video il disegno
        plt.show()

        return

###############################



#########################
###Funzioni Utilities####
########################
def ms2smpls(t_ms, sr):
  """
    Input:
        t_ms:  tempo in ms

    Output:
        numero di campioni equivalenti
  """
  return round(sr * t_ms/1000)

def smpls2ms(numSmpls, sr):
  """
    Input:
        numSmpls:  numero di campioni

    Output:
        tempo in ms equivalente
  """

  return (1000.0 * numSmpls/sr)

def normalizza(Data):

    return(Data / max(abs(Data)))
#################################



#########################
###Funzioni Filtro#######
#########################
def lowPass1(Data, sr, fc):
    # preso da pirkle pag 165

    # filter coesr
    thetaC = 2 *  np.pi*fc / sr
    gamma = 2 - np.cos(thetaC)
    b1 = np.sqrt(gamma ** 2 - 1) - gamma
    a0 = 1 + b1

    # init memories
    y1 = 0

    for i in range(len(Data)):
        x = Data[i]

        # eq LP filter
        # y[n] = a0 * x[n]  - b1 * y[n-1]
        y = a0 * x  - b1 * y1

        # update memories
        y1 = y

        Data[i] = y

    return Data

def LP_ma(DataIn, sr, N):

   """
    # ESEMPIO CON N = 4
    N = 4

    for i in range(len(Data)):
      Y[i] = (Data[i] + Data[i - 1] + Data[i - 2] + Data[i - 3])/4

   """
   # crea un vettore vuoto
   DataOut = np.zeros(len(DataIn))

   DataOut = np.concatenate((np.zeros(N), DataIn))

   for i in range(len(DataIn)):
    for j in range(N):
      DataOut[i] += DataIn[i - j]


   return(DataOut / N)

def  envelope(DataIn, L):

    DataIn =  np.concatenate((np.zeros(L), DataIn))
    DataOut = np.zeros(len(DataIn))

    '''
    for i in range(len(DataIn)-L):
        for j in range(L):
            DataOut[i] += DataIn[i+j]'''

    for i in range(len(DataIn)-L):
        DataOut[i] = np.sum(abs(DataIn[i:i+L]))

    return DataOut/L

def comb(DataIn, sr, delMs, gain, delMaxMs):

    """
    COMB FILTER FIR / IIR

    Inputs:
        DataIn: input numpy array
        sr: sampling rate
        delyMs: gain in ms
        gain: reflection gain
        delayMaxMs: max delay line length line
        combType: comb FIR / IIR  [0, 1]

    Output:
        DataOut: comb filter output
	"""

    if (delMs > delMaxMs):
        return(DataIn)


    DataOut = np.zeros(len(DataIn))


    delaySmpls = ms2smpls(delMs, sr)
    delayMaxSmpls = ms2smpls(delMaxMs, sr)


    tmpBuf = np.zeros(delayMaxSmpls)


    for i in range(len(DataIn)):


        x = DataIn[i]


        y = x + gain * tmpBuf[-delaySmpls]


        DataOut[i] = y


        tmpBuf = np.append(tmpBuf, x)


        tmpBuf =  np.delete(tmpBuf, 0)

    return(DataOut)
##########################



#########################
###Funzioni Custom#######
#########################
def trim(Data, start, end):

     return Data[start:end]

def cut(Data, start, end):

    return np.concatenate((Data[:start],Data[end:]))
##########################




#########################
#########  MAIN   #######
#########################
#X = normalizza(X_int)
#X = lowPass1(X, sr, )
#X= fir(X_int, sr, 100, 2, 200)
#X= LP_ma(X, sr, 50)
plot_wav_spectrogram(f_wav, sr)
#X = envelope(X,10000)
disegna(X, sr)


#########################
### Funzioni Template ###
#########################
# NORMALIZZAZIONE ... paracadute
Y = normalizza(X)

#  stampa la massima ampiezza in valore assoluto
print('max abs amp output:', max(abs(Y)))
"""
# trasforma in 16 bit (opzionale, se non la metto salva a 32 bit float)
Y_int = np.round(Y * 32768)
Y_int = Y_int.astype(int)
wavfile.write("ouput.wav", sr, Y_int)
"""
# scrivi File su disco float 32
wavfile.write("out.wav", sr, Y)

# disegna output (va posizionato qui altrimenti matplot lib blocca la prosecuzione in repl)
disegna(Y,sr)

#################################


'''def detection(x, pre_max, post_max, pre_avg, post_avg, delta, wait):

    #max dalla finestra




    max_length = pre_max + post_max
    max_origin = np.ceil(0.5 * (pre_max - post_max))
    mov_max = []
    mov_max = scipy.ndimage.filters.maximum_filter1d(x, max_length, -1, None, mode="constant", cval = x.min(), origin=max_origin)
    #avg dalla finestra
    avg_length = pre_avg + post_avg
    avg_origin = np.ceil(0.5 * (pre_avg - post_avg))
    mov_avg = scipy.ndimage.filters.uniform_filter1d(x, avg_length, -1, None, mode="nearest", origin = avg_origin)


    n = 0

    while n - pre_avg < 0 and n < x.shape[0]:
        # Regola 1
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start : n + post_avg])
        n += 1
    # Correzione finestra alla fine per leakage
    n = x.shape[0] - post_avg


    # First mask out all entries not equal to the local max
    detections = x * (x == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections * (detections >= (mov_avg + delta))

    # Initialize peaks array, to be filled greedily
    peaks = np.zeros(len(x))

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i

    return np.array(peaks)

def plots(DataIn):
    D = np.abs(librosa.stft(DataIn))
    times = librosa.times_like(D)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),y_axis='log', x_axis='time', ax=ax[0]),ax[0].set(title='Power spectrogram'),ax[0].label_outer()
    C = np.abs(librosa.cqt(DataIn, sr=sr))
    ax[1].plot(times, C / C.max(), alpha=0.8, label='Mean (CQT)'), ax[1].legend(),ax[1].set(ylabel='Normalized strength', yticks=[])
'''
