import matplotlib
import matplotlib.pyplot as plt
import numpy as np

s = 1.0    #normalizacja jednostek miar
Hz = 1.0/s 

tSymbol = 1.0 * s #czas trwania pojedynczego symbolu
fpr = 100.0 * Hz      #czestotliwosc probkowania
f = 10.0 * Hz         #czestotliwosc nosnej

highState = 1 #umowne poziomy (napiecia?) dla odpowiednich bitow
lowState = 0

bitarr = np.array([True, False, False, True, False]) #przykladowa tablica "bitow" do przeslania

bitarrLvld =np.array([highState if i else lowState for i in bitarr]) #nadajemy bitom wlasciwe poziomy napiecia 

t = np.arange(0.0, tSymbol*len(bitarr) ,1/fpr) #dziedzina czasu

bitSamples = np.repeat(bitarrLvld, fpr) #powielamy kazdy bit, by wytworzyc tablice probek

carrier = np.sin(2.0 * np.pi * f * t) #nosna

signal = np.sin(2.0 * np.pi * f * t + (np.pi*bitSamples))

fig, ax = plt.subplots() #wykresy
ax.plot(t,bitSamples)
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t, carrier)
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(t, signal)
plt.show()
