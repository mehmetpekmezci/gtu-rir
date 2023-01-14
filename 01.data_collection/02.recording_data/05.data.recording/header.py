#!/usr/bin/env python3
##
## IMPORTS
##
import importlib
logging     = importlib.import_module("logging")
tarfile     = importlib.import_module("tarfile")
csv         = importlib.import_module("csv")
glob        = importlib.import_module("glob")
sys         = importlib.import_module("sys")
os          = importlib.import_module("os")
argparse    = importlib.import_module("argparse")
np          = importlib.import_module("numpy")
librosa     = importlib.import_module("librosa")
time        = importlib.import_module("time")
random      = importlib.import_module("random")
datetime    = importlib.import_module("datetime")
pyaudio     = importlib.import_module("pyaudio")
wave        = importlib.import_module("wave")
threading   = importlib.import_module("threading")
pprint      = importlib.import_module("pprint")
#sa          = importlib.import_module("simpleaudio") #  import simpleaudio as sa
lywsd03mmc  = importlib.import_module("lywsd03mmc") 
cv2         = importlib.import_module("cv2") 
subprocess  = importlib.import_module("subprocess")
scipy       = importlib.import_module("scipy")
configparser= importlib.import_module("configparser")
math        = importlib.import_module("math")
shutil      = importlib.import_module("shutil")




SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME=os.path.basename(SCRIPT_DIR)
DATA_DIR = SCRIPT_DIR+'/../../../data/'
LOG_DIR=SCRIPT_DIR+"/../../../logs/"+SCRIPT_NAME
ROOM_NUMBER=0
RECORD_TIMESTAMP=str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
## CONFUGRE LOGGING
logger=logging.getLogger('rir')
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
# create file handler and set level to debug
loggingFileHandler = logging.FileHandler(LOG_DIR+'/rir-'+RECORD_TIMESTAMP+'.log')
loggingFileHandler.setLevel(logging.DEBUG)
# create console handler and set level to debug
loggingConsoleHandler = logging.StreamHandler()
loggingConsoleHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
loggingFileHandler.setFormatter(formatter)
loggingConsoleHandler.setFormatter(formatter)
logger.addHandler(loggingFileHandler)
logger.addHandler(loggingConsoleHandler)
#logger.debug('debug message')
#logger.info('info message')
#logger.warn('warn message')
#logger.error('error message')
#logger.critical('critical message')

##
## DATA CONSTANTS
##
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE=44100 # Hz
TRANSMIT_SILENCE=2
TRANSMIT_DURATION=10 # 10 seconds signal
RECORD_DURATION=4+TRANSMIT_SILENCE+TRANSMIT_DURATION+TRANSMIT_SILENCE+6 #  2 seconds for silence , 14 seconds signal ( 2silence+10 ess signal + 2silence), 4 seconds silence and latency
MAXFREQUENCY=15e3#20e3==20KHz
MINFREQUENCY=60.0#25Hz

SONG_TRANSMIT_DURATION=30
SONG_RECORD_DURATION=4+TRANSMIT_SILENCE+SONG_TRANSMIT_DURATION+TRANSMIT_SILENCE+6

FNULL = open(os.devnull, 'w')

          
#SPEAKERS=["alsa_output.pci-0000_00_1b.0.analog-stereo","bluez_sink.3D_45_41_74_43_09.a2dp_sink","bluez_sink.F9_3D_F7_F1_45_66.a2dp_sink"] ## index-0 : laptop's own speaker, index-1 : lower speaker, index-2: upper speaker
#SPEAKERS=["alsa_output.pci-0000_00_1b.0.analog-stereo","alsa_output.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-stereo"] ## index-0 : laptop's own speaker, index-1 : usb sound card related speaker


## pacmd list-sinks| grep name:
#SPEAKERS=["alsa_output.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-stereo","alsa_output.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-stereo.2"] ## index-0 : usb sound card related speaker



usbport=dict()
micport=dict()

#cat /proc/asound/cards 

NUMBER_OF_MICROPHONES=6

R_SPEAKER=[]
Z_SPEAKER=[]
R_MIC=[]
Z_MIC=[]

## ABSOLUTE COORDINATE DEFINITIONS ACCORDING TO THE ROOM
# [0,0,0] is the corner where the floor corner of the room closest to the door.
########################
##     Z
##     |_ Y
##     /       CENTER=Center of the microphone stand
##    X
#######################

MICROPHONE_DIRECTION_PROPERTIES={ 
  0:  { "x" :  0, "y" :  1, "z" :  0 } , 
  1 : { "x" :  0, "y" :  0, "z" : -1 } , 
  2 : { "x" :  1, "y" :  0, "z" :  0 } , 
  3 : { "x" :  0, "y" : -1, "z" :  0 } , 
  4 : { "x" :  0, "y" :  0, "z" : -1 } , 
  5 : { "x" :  1, "y" :  0, "z" :  0 } , 
} 

MICROPHONE_DIRECTION_LABELS={ 
  0 : "RIGHT", 
  1 : "DOWN" , 
  2 : "OUT_TO_FACE" , 
  3 : "LEFT" , 
  4 : "DOWN" , 
  5 : "OUT_TO_FACE" , 
} 


SPEAKER_DRECTION_LABELS={ 
  0 : "TOP LEFT", 
  1 : "TOP CENTER" , 
  2 : "BOTTOM CENTER" , 
  3 : "BOTTOM LEFT" , 
} 

#ONE_STEP_DEGREE=36

MAX_NUMBER_OF_MIC_ITERATION=10
MAX_NUMBER_OF_SPEAKER_ITERATION=10


ROOM_DIM_WIDTH=-1
ROOM_DIM_DEPTH=-1
ROOM_DIM_HEIGHT=-1


