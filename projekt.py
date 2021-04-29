import matplotlib.pyplot as plt
import scipy.signal as signal
from math import pi
import numpy as np
import cv2

t_symbol = 1.0  # czas trwania pojedyńczego symbolu
fs = 100  # częstotliwość próbkowania
f = 6.0  # częstotliwość nośnej
# np.random.seed(19680801)  # ziarno losowości
noise_level = 2  # poziom szumu, przy poziomie 8 się psuje
black = [0, 0, 0]
white = [255, 255, 255]

img = cv2.imread('10px_b_w_img.png')
shape = img.shape
height = shape[0]
width = shape[1]

bit_array = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        px = img[i, j]
        if px[0] == 255 and px[1] == 255 and px[2] == 255:
            bit_array[i, j] = 1

size = bit_array.size

t = np.arange(0.0, t_symbol * size, 1 / fs)  # dziedzina czasu

noise = np.random.randn(len(t))  # generowanie szumu białego
bit_samples = np.repeat(bit_array, fs)  # powielamy każdy bit, by wytworzyć tablice próbek

carrier = np.cos(2.0 * pi * f * t)  # nośna

bpsk_sig = np.cos(2.0 * pi * f * t + (pi * bit_samples))
bpsk_sig_noised = bpsk_sig + noise * noise_level

# Konstrukcja filtra środkowoprzepustowego, częstotliwości graniczne [f / 2, 1.5 * f]
# scipy.signal.ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None)
# (N - rząd filtru,
# rp - maksymalne zafalowanie w paśmie przepustowym (maksymalna utrata wzmocnienia w tym paśmie), w decybelach,
# rs - minimalne tłumienie wymagane w paśmie zaporowym. Podane w decybelach jako liczba dodatnia,
# Wn - Sekwencja z krytycznymi częstotliwościami, w przypadku naszych filtrów jest to punkt w paśmie przejściowym,
# w którym wzmocnienie najpierw spada ponieżej - rp.
# W przypadku filtrów cyfrowych Wn są w tych samych jednostkach co fs.
# Domyślnie fs to 2 pół cykli / próbkę, więc są one znormalizowane od 0 do 1, gdzie 1 to częstotliwość Nyquista.
# (Wn jest zatem w (jednostkach) połowie cykli / próbkę).
# b - licznik, a - mianownik - wielomiany filtra IIR
# Filtr rzędu N ma N+1 współczynników b w liczniku
# i N współczynników a w mianowniku.
[b11, a11] = signal.ellip(5, 0.5, 60, [f / 2 * 2 / fs, f * 1.5 * 2 / fs], btype='bandpass', analog=False, output='ba')

# Konstrukcja filtra dolnoprzepustowego, częstotliwość odcięcia pasma przepustowego wynosi f / 2

[b12, a12] = signal.ellip(5, 0.5, 60, (f / 2 * 2 / fs), btype='lowpass', analog=False, output='ba')

# Filtruj szum poza pasmem za pomocą filtra środkowoprzepustowego

bpsk_sig_bandpass = signal.filtfilt(b11, a11, bpsk_sig_noised) * (-1)

# Koherentna demodulacja, pomnożona przez spójną nośną w fazie o tej samej częstotliwości

bandpass_demodulated = bpsk_sig_bandpass * (carrier * 2)

# Filtr dolnoprzepustowy

bpsk_sig_lowpass = signal.filtfilt(b12, a12, bandpass_demodulated)

# Sprawdzenie próbek

bit_array_received = np.zeros(size)

for i in range(size):
    temp = 0
    for j in range(fs):
        temp = temp + bpsk_sig_lowpass[i * fs + j]

    if temp > 0:
        bit_array_received[i] = 1

    else:
        bit_array_received[i] = 0

bit_samples_received = np.repeat(bit_array_received, fs)  # powielamy każdy bit, by wytworzyć tablice próbek

# wykresy
fig, axs = plt.subplots(3, 3)
axs[0, 0].plot(t, bit_samples)
axs[0, 0].set_xlabel('Czas [s]')
axs[0, 0].set_ylabel('Amplituda ')
axs[0, 0].set_title('Pierwotny sygnał')
axs[0, 0].grid(True)

axs[1, 0].plot(t, bit_samples_received)
axs[1, 0].set_xlabel('Czas [s]')
axs[1, 0].set_ylabel('Amplituda')
axs[1, 0].set_title('Odebrany sygnał')
axs[1, 0].grid(True)

axs[2, 0].axis('off')

axs[0, 1].plot(t, carrier)
axs[0, 1].set_xlabel('Czas [s]')
axs[0, 1].set_ylabel('Amplituda')
axs[0, 1].set_title('Nośna')
axs[0, 1].grid(True)

axs[1, 1].plot(t, bpsk_sig)
axs[1, 1].set_xlabel('Czas [s]')
axs[1, 1].set_ylabel('Amplituda')
axs[1, 1].set_title('Sygnał')
axs[1, 1].grid(True)

axs[2, 1].plot(t, bpsk_sig_noised)
axs[2, 1].set_xlabel('Czas [s]')
axs[2, 1].set_ylabel('Amplituda')
axs[2, 1].set_title('Sygnał z szumem')
axs[2, 1].grid(True)

axs[0, 2].plot(t, bpsk_sig_bandpass)
axs[0, 2].set_xlabel('Czas [s]')
axs[0, 2].set_ylabel('Amplituda')
axs[0, 2].set_title('Sygnał z szumem po filtrze środkowoprzepustowym')
axs[0, 2].grid(True)

axs[1, 2].plot(t, bandpass_demodulated)
axs[1, 2].set_xlabel('Czas [s]')
axs[1, 2].set_ylabel('Amplituda')
axs[1, 2].set_title('Sygnał z szumem po demodulacji nośną')
axs[1, 2].grid(True)

axs[2, 2].plot(t, bpsk_sig_lowpass)
axs[2, 2].set_xlabel('Czas [s]')
axs[2, 2].set_ylabel('Amplituda')
axs[2, 2].set_title('Sygnał z szumem po filtrze dolnoprzepustowym')
axs[2, 2].grid(True)

fig.tight_layout()  # czytelne rozłożenie wykresów

# Przetworzenie listy z bitami na obraz
# --------------- działający kod ----------------------------------------
detection_bpsk_reshaped = np.reshape(bit_array_received, (height, width))

img_end = np.zeros([height, width, 3])

for i in range(height):
    for j in range(width):
        if detection_bpsk_reshaped[i, j] == 1:
            img_end[i, j] = white
# -----------------------------------------------------------------------

# --------------- zabawny kod zależny od printa i str -------------------
# nie działa jeśli print i str są zakomentowane
# zadziała po odkomentowaniu jednego z nich
#
# detection_bpsk_reshaped = np.reshape(bit_array_received, (height, width))
#
# # print(detection_bpsk_reshaped)
# # str(detection_bpsk_reshaped)
#
# img_end = np.empty([height, width])
#
# for i in range(height):
#     for j in range(width):
#         if detection_bpsk_reshaped[i, j] == 1:
#             np.append(img_end, white)
# -----------------------------------------------------------------------

# wyświetlenie obrazów
cv2.namedWindow('Sygnal orginalny', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sygnal orginalny', 800, 400)
cv2.imshow('Sygnal orginalny', img)

cv2.namedWindow('Sygnal odebrany', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sygnal odebrany', 800, 400)
cv2.imshow('Sygnal odebrany', img_end)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
