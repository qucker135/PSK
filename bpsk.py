import matplotlib.pyplot as plt
import scipy.signal as signal
from math import pi
import numpy as np

t_symbol = 1.0  # czas trwania pojedyńczego symbolu
fs = 100  # częstotliwość próbkowania
f = 6.0  # częstotliwość nośnej
np.random.seed(19680801)  # ziarno losowości
noise_level = 6  # poziom szumu, przy poziomie 8 się psuje

bit_array = np.array([1, 0, 0, 1, 0])  # przykładowa tablica "bitów" do przesłania

size = bit_array.size

# diagram konstelacyjny
diagram_bits = np.cos(pi * bit_array) + 0j #1j * np.sin(pi * bit_array) #taka ciekawostka

# BPSK
t = np.arange(0.0, t_symbol * size, 1 / fs)  # dziedzina czasu

bit_samples = np.repeat(bit_array, fs)  # powielamy każdy bit, by wytworzyć tablice próbek

carrier = np.cos(2.0 * pi * f * t)  # nośna

bpsk_sig = np.cos(2.0 * pi * f * t + (pi * bit_samples))

noise = np.random.randn(len(t))  # generowanie szumu białego
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
[b11, a11] = signal.ellip(5, 0.7, 110, [f / 2 * 2 / fs, f * 1.5 * 2 / fs], btype='bandpass', analog=False, output='ba')

# Konstrukcja filtra dolnoprzepustowego, częstotliwość odcięcia pasma przepustowego wynosi f / 2

[b12, a12] = signal.ellip(5, 0.7, 110, (f / 2 * 2 / fs), btype='lowpass', analog=False, output='ba')

# Filtruj szum poza pasmem za pomocą filtra środkowoprzepustowego

bpsk_sig_bandpass = signal.filtfilt(b11, a11, bpsk_sig_noised) * (-1)

# Koherentna demodulacja, pomnożona przez spójną nośną w fazie o tej samej częstotliwości

bandpass_demodulated = bpsk_sig_bandpass * (carrier * 2)

# Filtr dolnoprzepustowy

bpsk_sig_lowpass = signal.filtfilt(b12, a12, bandpass_demodulated)

# Sprawdzenie próbek

bit_array_received = np.zeros(size, dtype=int)

for i in range(size): # petla po kazdym otrzymanym bicie
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

# diagramy konstelacji
fig2, axs2 = plt.subplots(1,2)
axs2[0].plot(np.real(diagram_bits), np.imag(diagram_bits), '.')
axs2[0].set_xlabel('Real')
axs2[0].set_ylabel('Imag')
axs2[0].set_title('Diagram konstelacyjny (sygnał wysłany)')
axs2[0].grid(True)

axs2[1].plot(bpsk_sig_lowpass, np.zeros(bpsk_sig_lowpass.size), '.')
axs2[1].set_xlabel('Real')
axs2[1].set_ylabel('Imag')
axs2[1].set_title('Diagram konstelacyjny (sygnał odebrany)')
axs2[1].grid(True)

fig2.tight_layout()

plt.show()