import noisereduce as nr
from scipy.io import wavfile
import sys
import librosa
import numpy as np
import wave


fname=sys.argv[1]

# load data
#data,rate = librosa.load(fname+'.wav',dtype='int16',mono=True)
#rate,data=wavfile.read(fname+'.wav')
data,rate=librosa.load(fname,sr=44100,mono=True)
data=np.array(data).astype(np.float32)
print("data.dtype="+str(data.dtype))
print("data.len="+str(len(data)))
print("data[15]="+str(data[15]))
# select section of data that is noise

noise_sampling_part = data[0:2*rate]
# perform noise reduction
#reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noise_sampling_part, verbose=False)
reduced_noise=np.array(reduced_noise).astype(np.float32)
print("reduced_noise.dtype="+str(reduced_noise.dtype))

#librosa.output.write_wav(fname+'.clean.wav', reduced_noise, rate)

wavfile.write(fname,rate,np.array(reduced_noise).astype(np.float32))



