import numpy as np
from math import pi, sqrt, asin, sin, acos, cos
import matplotlib.pyplot as plt
import scipy.signal as signal


t_Symbol = 1.0 #czas trwania pojedynczego symbolu
fs = 100       #czestotliwosc probkowania
f = 6.0        #czestotliwosc nosnej
noise_level = 0.0

bit_array = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # przykładowa tablica "bitów" do przesłania

#stworzenie tablicy, reprezentujacej pary bitów (cyfry w systemie czwórkowym)
symbol_array_qpsk = []
for i in range((len(bit_array)+1)//2):
	symbol_array_qpsk.append(bit_array[2*i]+2*(bit_array[2*i+1] if 2*i+1 < len(bit_array) else False))

#utworzenie obiektu np.array na podstawie istniejącej już listy
symbol_array_qpsk = np.array(symbol_array_qpsk)

size = len(symbol_array_qpsk)

t = np.arange(0.0, t_Symbol*len(symbol_array_qpsk), 1/fs)
carrier = np.cos(2.0 * pi * f * t)  # nośna
symbol_array_qpsk_samples = np.repeat(symbol_array_qpsk, t_Symbol*fs)
signal_modulated = np.cos(2*pi*f*t + pi/2*symbol_array_qpsk_samples + asin(1.0/sqrt(10.0))) #pi/6)

noise = np.random.randn(len(t))
signal_modulated_with_noise = signal_modulated + noise * noise_level

#Listy niezbędne do diagramu konstelacji
diagram_radians = pi/2.0*symbol_array_qpsk_samples + asin(1.0/sqrt(10.0))
diagram_symbols = np.cos(diagram_radians) + 1j * np.sin(diagram_radians)

[b11, a11] = signal.ellip(5, 0.7, 110, [f / 2 * 2 / fs, f * 1.5 * 2 / fs], btype='bandpass', analog=False, output='ba')

# Konstrukcja filtra dolnoprzepustowego, częstotliwość odcięcia pasma przepustowego wynosi f / 2

[b12, a12] = signal.ellip(5, 0.7, 110, (f / 2 * 2 / fs), btype='lowpass', analog=False, output='ba')

# Filtruj szum poza pasmem za pomocą filtra środkowoprzepustowego

qpsk_sig_bandpass = signal.filtfilt(b11, a11, signal_modulated_with_noise) #* (-1)

# Koherentna demodulacja, pomnożona przez spójną nośną w fazie o tej samej częstotliwości

bandpass_demodulated = qpsk_sig_bandpass * (carrier * 2)

# Filtr dolnoprzepustowy

qpsk_sig_lowpass = signal.filtfilt(b12, a12, bandpass_demodulated)

# Sprawdzenie próbek

symbol_array_received = np.zeros(size, dtype = int)

for i in range(size):
	temp = 0
	for j in range(fs):
		temp = temp + qpsk_sig_lowpass[i * fs + j]
	
	temp/=fs
	

	#wersja dla fazy modulacji = arcsin(1.0/sqrt(10.0)) (równomierne rozłożenie cosinusów, najmniejszy średni błąd)
	#temp - srednia dla danego symbolu - trzeba sprawdzic, ktorej z wartosci 3.0/sqrt(10.0), 1.0/sqrt(10.0), -1.0/sqrt(10.0), -3.0/sqrt(10.0) jest najblizsza
	if temp > 0:
		if temp > 2.0/sqrt(10.0):  #przyjmujemy, ze 3.0/sqrt(10.0)
			symbol_array_received[i] = 0
		else: #przyjmujemy, że 1.0/sqrt(10.0)
			symbol_array_received[i] = 3

	else:
		if temp < -2.0/sqrt(10.0):  #przyjmujemy, ze -3.0/sqrt(10.0)
			symbol_array_received[i] = 2
		else: #przyjmujemy, że -1.0/sqrt(10.0)
			symbol_array_received[i] = 1


#listy do diagramu konstelacyjnego
symbol_samples_received = np.repeat(symbol_array_received, t_Symbol*fs)  # powielamy każdy symbol, by wytworzyć tablice próbek

qpsk_sig_lowpass_2nd_coor = np.zeros(qpsk_sig_lowpass.size)

for i in range(qpsk_sig_lowpass.size):
	if qpsk_sig_lowpass[i] > 2.0/sqrt(10.0) or ( qpsk_sig_lowpass[i] < 0.0 and qpsk_sig_lowpass[i] > -2.0/sqrt(10.0)):
		qpsk_sig_lowpass_2nd_coor[i] = sin(acos(min(qpsk_sig_lowpass[i], 1.0)))
	else:
		qpsk_sig_lowpass_2nd_coor[i] = -1.0 * sin(acos(max(qpsk_sig_lowpass[i], -1.0)))

diagram_samples_received = np.zeros(qpsk_sig_lowpass.size, dtype=complex)
for i in range(qpsk_sig_lowpass.size):
	diagram_samples_received[i] = max(min(1.0,qpsk_sig_lowpass[i]),-1.0) + 1j * qpsk_sig_lowpass_2nd_coor[i]

#przetworzenie kodu czwórkowego na bity
bit_array_received = np.array([], dtype=int)
for i in range(symbol_array_received.size):
	bit_array_received = np.append(bit_array_received, [symbol_array_received[i]%2, symbol_array_received[i]//2])

# wykresy
fig, axs = plt.subplots(3, 3)
axs[0, 0].plot(t, symbol_array_qpsk_samples)
axs[0, 0].set_xlabel('Czas [s]')
axs[0, 0].set_ylabel('Amplituda ')
axs[0, 0].set_title('Pierwotny sygnał')
axs[0, 0].grid(True)

axs[1, 0].plot(t, symbol_samples_received)
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

axs[1, 1].plot(t, signal_modulated)
axs[1, 1].set_xlabel('Czas [s]')
axs[1, 1].set_ylabel('Amplituda')
axs[1, 1].set_title('Sygnał')
axs[1, 1].grid(True)

axs[2, 1].plot(t, signal_modulated_with_noise)
axs[2, 1].set_xlabel('Czas [s]')
axs[2, 1].set_ylabel('Amplituda')
axs[2, 1].set_title('Sygnał z szumem')
axs[2, 1].grid(True)

axs[0, 2].plot(t, qpsk_sig_bandpass)
axs[0, 2].set_xlabel('Czas [s]')
axs[0, 2].set_ylabel('Amplituda')
axs[0, 2].set_title('Sygnał z szumem po filtrze środkowoprzepustowym')
axs[0, 2].grid(True)

axs[1, 2].plot(t, bandpass_demodulated)
axs[1, 2].set_xlabel('Czas [s]')
axs[1, 2].set_ylabel('Amplituda')
axs[1, 2].set_title('Sygnał z szumem po demodulacji nośną')
axs[1, 2].grid(True)

axs[2, 2].plot(t, qpsk_sig_lowpass)
axs[2, 2].set_xlabel('Czas [s]')
axs[2, 2].set_ylabel('Amplituda')
axs[2, 2].set_title('Sygnał z szumem po filtrze dolnoprzepustowym')
axs[2, 2].grid(True)

fig.tight_layout()  # czytelne rozłożenie wykresów

# diagramy konstelacji
fig2, axs2 = plt.subplots(1,2)
axs2[0].plot(np.real(diagram_symbols), np.imag(diagram_symbols), '.')
axs2[0].set_xlabel("Real")
axs2[0].set_ylabel("Imag")
axs2[0].set_title("Diagram konstelacyjny (sygnał wysłany)")
axs2[0].grid(True)

axs2[1].plot(np.real(diagram_samples_received), np.imag(diagram_samples_received), '.')
axs2[1].set_xlabel("Real")
axs2[1].set_ylabel("Imag")
axs2[1].set_title("Diagram konstelacyjny (sygnał odebrany)")
axs2[1].grid(True)

fig2.tight_layout()

plt.show()
