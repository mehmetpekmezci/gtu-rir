import time

import numpy as np
import pyaudio
import matplotlib.pyplot as plt

#p = pyaudio.PyAudio()

volume = 1  # range [0.0, 1.0]
fs = 22050  # sampling rate, Hz, must be integer
duration = 0.001  # in seconds, may be float
f = 44.0  # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)


fig, (ax1) = plt.subplots(1)
#fig.suptitle('Vertically stacked subplots')

# Creating x axis with range and y axis with Sine
# Function for Plotting Cosine Graph
x = np.arange(0, samples.shape[0])
line = np.zeros(x.shape)
y = samples
ax1.plot(x, y, color='black')
ax1.plot(x, line, color='gray')
ax1.set(xlabel='t', ylabel='x(t)')
ax1.set_title('Impulse Signal')
fig.tight_layout(pad=2.0)

plt.show()




## per @yahweh comment explicitly convert to bytes sequence
#output_bytes = (volume * samples).tobytes()
#
## for paFloat32 sample values must be in range [-1.0, 1.0]
#stream = p.open(format=pyaudio.paFloat32,
#                channels=1,
#                rate=fs,
#                output=True)
#
## play. May repeat with different volume values (if done interactively)
#
#start_time = time.time()
#stream.write(output_bytes)
#print("Played sound for {:.2f} seconds".format(time.time() - start_time))
#
#stream.stop_stream()
#stream.close()
#
#p.terminate()
