from header import *
from speaker_numbers import *
import scipy.io.wavfile as wavfile

## channel0 == leftSignal
## channel1 == rightSignal

## SPEAKER PHYSICAL LOCATIONS :

###       Speaker1.channel0     ---      Speaker1.channel1    ---   
###                                                                Speaker0.channel1 ---
###                                                                                       Speaker0.channel0 ---
###
###           3                                 2                         1                       0


## pacmd list-sinks | grep name:

def transmitSignal(speakerNo,essSignal,TEST_MODE,format =pyaudio.paInt16):


    
    
    if TEST_MODE == "TEST_MODE_FAST_DEVICE_MOC" :
       time.sleep(1)
       #time.sleep(TRANSMIT_SILENCE+TRANSMIT_DURATION+TRANSMIT_SILENCE)
       ## SONG_TRANSMIT_DURATION
       return
       
      
    
    if TEST_MODE == "TEST_MODE_DEVICE_MOC" :
       print("NOT SETTING SPEAKER, USING DEFAULT ONE AS DEVICE MOC")
    else :
       os.system("pacmd set-default-sink "+SPEAKERS[speakerNo])

    p = pyaudio.PyAudio()
    #essSignalSampleWidth=2 ## from ess.py
    essSignalChannels=2 ## either left or right is 0, other channel is ess signal itself
    essSignalSampleRate=44100 ## ess.py , opts.rate
    stream = p.open(format = format,channels =essSignalChannels,rate = essSignalSampleRate,output = True)
    #stream.write(essSignal.tostring())
    s=essSignal.tostring()
    stream.write(s)
    stream.close()
    p.terminate()
    

def generate_left_signal(signal):
    left=signal
    right=np.copy(signal)
    right[:]=0
    return np.array([left, right]).transpose()    


def generate_right_signal(signal):
    
    right=signal
    left=np.copy(signal)
    left[:]=0
    return np.array([left, right]).transpose()




def get_song_signal():
    song,rate=librosa.load(SCRIPT_DIR+'/mono.song.ist.kant.alt.wav',sr=44100, mono=True)
    return song



