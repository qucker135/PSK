import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import pi

s = 1.0  # normalizacja jednostek miar
Hz = 1.0 / s
t_symbol = 1.0 * s  # czas trwania pojedyńczego symbolu
fs = 100 * Hz  # częstotliwość próbkowania
f = 6.0 * Hz  # częstotliwość nośnej
np.random.seed(19680801)  # ziarno losowości
noise_level = 7  # poziom szumu

bit_array = np.array([1, 0, 0, 1, 0])  # przykładowa tablica "bitów" do przesłania

size = len(bit_array)

t = np.arange(0.0, t_symbol * size, 1 / fs)  # dziedzina czasu

noise = np.random.randn(len(t))  # generowanie szumu białego
bit_samples = np.repeat(bit_array, fs)  # powielamy każdy bit, by wytworzyć tablice próbek

carrier = np.cos(2.0 * pi * f * t)  # nośna

bpsk_sig = np.cos(2.0 * pi * f * t + (pi * bit_samples))
bpsk_sig_noised = bpsk_sig + noise * noise_level

# wykresy
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, bit_samples)
axs[0, 0].set_xlabel('Czas')
axs[0, 0].set_ylabel('Amplituda')
axs[0, 0].set_title('Pierwotny sygnał')
axs[0, 0].grid(True)

axs[0, 1].plot(t, carrier)
axs[0, 1].set_xlabel('Czas')
axs[0, 1].set_ylabel('Amplituda')
axs[0, 1].set_title('Nośna')
axs[0, 1].grid(True)

axs[1, 0].plot(t, bpsk_sig)
axs[1, 0].set_xlabel('Czas')
axs[1, 0].set_ylabel('Amplituda')
axs[1, 0].set_title('Sygnał')
axs[1, 0].grid(True)

axs[1, 1].plot(t, bpsk_sig_noised)
axs[1, 1].set_xlabel('Czas')
axs[1, 1].set_ylabel('Amplituda')
axs[1, 1].set_title('Sygnał z szumem')
axs[1, 1].grid(True)

fig.tight_layout()  # czytelne rozłożenie wykresów
plt.show()