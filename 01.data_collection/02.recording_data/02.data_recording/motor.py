from header import *

#TIC_T500="00309901" ## ticcmd --list|grep T500| cut -d, -f1
#TIC_T825="00282604" ## ticcmd --list|grep T825| cut -d, -f1

def moveSpeakerStepMotor(speakerMotorIterationDirection,step,TEST_MODE):
    logger.info ('moveSpeakerStepMotor Step:'+str(step))
    logger.info ('moveSpeakerStepMotor Direction:'+str(speakerMotorIterationDirection))
    p=subprocess.Popen([SCRIPT_DIR+"/motor.iterate.sh", "T500", str(speakerMotorIterationDirection), str(step),TEST_MODE])
    p.wait()
    time.sleep(1)

def resetSpeakerStepMotor(TEST_MODE):
    logger.info ('reset speaker motor ...')
    if TEST_MODE != "TEST_MODE_NONE" :
       time.sleep(1)
       return    
    p=subprocess.Popen([SCRIPT_DIR+"/motor.reset.sh","T500"])
    p.wait()
    time.sleep(1)
    #subprocess.Popen(["ticcmd", "-d", TIC_T825, "--deenergize"])

def resetMicrophoneStepMotor(TEST_MODE):
    logger.info ('reset microphone motor ...')
    if TEST_MODE != "TEST_MODE_NONE" :
       time.sleep(1)
       return    
    p=subprocess.Popen([SCRIPT_DIR+"/motor.reset.sh","T825"])
    p.wait()
    time.sleep(1)

# microphoneIterationDirection is only +/- , justy on the same line, forward or backward.
def moveMicophoneStepMotor(step,TEST_MODE):
    logger.info ('moveMicophoneStepMotor Step:'+str(step))
    p=subprocess.Popen([SCRIPT_DIR+"/motor.iterate.sh", "T825", "1",str(step),TEST_MODE])
    p.wait()
    time.sleep(1)


def getMicrophoneMotorPosition(TEST_MODE): 
    p = subprocess.Popen([SCRIPT_DIR+"/motor_get_current_position.sh", "T825",TEST_MODE], stdout=subprocess.PIPE)
    p.wait()
    return p.stdout.read().strip().decode('ascii')


def getSpeakerMotorPosition(TEST_MODE): 
    p = subprocess.Popen([SCRIPT_DIR+"/motor_get_current_position.sh", "T500",TEST_MODE], stdout=subprocess.PIPE)
    p.wait()
    return p.stdout.read().strip().decode('ascii')

