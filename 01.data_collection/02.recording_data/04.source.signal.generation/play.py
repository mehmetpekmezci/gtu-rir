import pyaudio
import wave
import os

## pacmd list-sinks | grep name:


filename = 'ess_sginal.wav'

chunk = 1024

os.system("pacmd set-default-sink bluez_sink.3D_45_41_74_43_09.a2dp_sink")

# Set chunk size of 1024 samples per data frame

# Open the sound file
wf = wave.open(filename, 'rb')
# Create an interface to PortAudio
p = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# Read data in chunks
data = wf.readframes(chunk)
# Play the sound by writing the audio data to the stream
#while data != '':
while len(data) > 0 :
    stream.write(data)
    data = wf.readframes(chunk)

# Close and terminate the stream
stream.close()
p.terminate()


os.system("pacmd set-default-sink bluez_sink.F9_3D_F7_F1_45_66.a2dp_sink")


# Open the sound file
wf = wave.open(filename, 'rb')
# Create an interface to PortAudio
p = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# Read data in chunks
data = wf.readframes(chunk)
# Play the sound by writing the audio data to the stream
#while data != '':
while len(data) > 0 :
    stream.write(data)
    data = wf.readframes(chunk)

# Close and terminate the stream
stream.close()
p.terminate()


