from header import *

## pacmd list-sources | grep name:
#arecord -L | grep hw:CARD| grep -v plughw


def generateSimpleSinusWave(SAMPLE_RATE,AMPLITUDE,FREQUENCY,DURATION):
     sinusoidal_sound_wave_data=AMPLITUDE*np.sin(2*np.pi*np.arange(DURATION)*FREQUENCY/SAMPLE_RATE).astype(np.float32)
     return sinusoidal_sound_wave_data

def generateSampleWave(MICROPHONE_ID):
     MAX_AMPLITUDE=10 # 10/10 in fact
     NUMBER_OF_SINUS_WAVES_PER_SAMPLE=10
     
     wave=np.zeros((int(RECORD_DURATION*SAMPLE_RATE),)) 
     
     for i in range(NUMBER_OF_SINUS_WAVES_PER_SAMPLE):
         starting_position=int(((MICROPHONE_ID)*(MICROPHONE_ID)*6731*i)*np.random.rand()%RECORD_DURATION+10)
         duration=int(((MICROPHONE_ID+1)*(MICROPHONE_ID+1)*113+(MICROPHONE_ID+1)*313+(i+1)*1113)%RECORD_DURATION)
         frequency=int(((MICROPHONE_ID+1)*(MICROPHONE_ID+1)*(i+1)*273+(i+1)*10+(i+1)*100)*np.random.rand()%MAXFREQUENCY)
         amplitude=(((MICROPHONE_ID+1)*(MICROPHONE_ID+1)*(i+1)*3+(i+1)*2+(i+1)*7)*np.random.rand()%MAX_AMPLITUDE+0.5)/MAX_AMPLITUDE
         generatedSingleSignal=generateSimpleSinusWave(SAMPLE_RATE,amplitude,frequency,duration)
         wave[starting_position:starting_position+duration]+=generatedSingleSignal
         
     #print(f"wave={wave}")
     #print(f"wave.shape={wave.shape}")
     return wave

def receiveESSSignal(microphoneNo,record,TEST_MODE): 
    logger.info("ffmpeg "+ " -f "+ " alsa "+ " -ac " + " 1 "+ " -sample_rate "+ str(SAMPLE_RATE)+" -i hw:"+str(micport[microphoneNo])+",0 -t "+str(RECORD_DURATION)+" -f "+" f32le  deneme."+str(microphoneNo)+".wav")
    audioChannels=1
    if TEST_MODE == "TEST_MODE_FAST_DEVICE_MOC" :
       #sleep(RECORD_DURATION)
       audio_array = generateSampleWave(microphoneNo)
       record.appendReceivedSignal(microphoneNo,audio_array)
       return audio_array
    #if microphoneNo == 0 :
    #   audioChannels=2
    
    MICROPHONE_PORT=str(micport[microphoneNo])
    
    if TEST_MODE == "TEST_MODE_DEVICE_MOC" :
       MICROPHONE_PORT=0
    
    microphoneInput = subprocess.Popen(["ffmpeg", "-f", "alsa", "-ac", str(audioChannels), "-sample_rate", str(SAMPLE_RATE),"-i","hw:"+str(MICROPHONE_PORT)+",0", "-t",str(RECORD_DURATION),"-f","f32le", "pipe:1"],stdout=subprocess.PIPE)
    stdout_data = microphoneInput.stdout.read()
    audio_array = np.fromstring(stdout_data, dtype="float32")
    record.appendReceivedSignal(microphoneNo,audio_array)
    return audio_array




def receiveSongSignal(microphoneNo,record,TEST_MODE): 
    logger.info("ffmpeg "+ " -f "+ " alsa "+ " -ac " + " 1 "+ " -sample_rate "+ str(SAMPLE_RATE)+" -i hw:"+str(micport[microphoneNo])+",0 -t "+str(SONG_RECORD_DURATION)+" -f "+" f32le  deneme.song."+str(microphoneNo)+".wav")
    audioChannels=1
    if TEST_MODE == "TEST_MODE_FAST_DEVICE_MOC" :
       #sleep(RECORD_DURATION)
       audio_array = generateSampleWave(microphoneNo)
       record.appendReceivedSongSignal(microphoneNo,audio_array)
       return audio_array
    #if microphoneNo == 0 :
    #   audioChannels=2
    
    MICROPHONE_PORT=str(micport[microphoneNo])
    
    if TEST_MODE == "TEST_MODE_DEVICE_MOC" :
       MICROPHONE_PORT=0
    
    microphoneInput = subprocess.Popen(["ffmpeg", "-f", "alsa", "-ac", str(audioChannels), "-sample_rate", str(SAMPLE_RATE),"-i","hw:"+str(MICROPHONE_PORT)+",0", "-t",str(SONG_RECORD_DURATION),"-f","f32le", "pipe:1"],stdout=subprocess.PIPE)
    stdout_data = microphoneInput.stdout.read()
    audio_array = np.fromstring(stdout_data, dtype="float32")
    record.appendReceivedSongSignal(microphoneNo,audio_array)
    return audio_array





