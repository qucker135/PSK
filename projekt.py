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

fig, ax = plt.subplots() #wykresy
ax.plot(t, bit_samples)
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t, carrier)
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(t, bpsk_sig)
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(t, bpsk_sig_noised)
plt.show()