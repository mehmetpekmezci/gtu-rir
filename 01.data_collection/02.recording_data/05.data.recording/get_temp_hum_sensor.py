#!/usr/bin/env python3
from header import *
from sensor import *



def main():
    tempHumMicrophone=getMicrophoneTemperatureHumidity()
    tempHumSpeaker=getSpeakerTemperatureHumidity()

    record_file = open("/tmp/tempHum.txt", "w")
    record_file.write("temperatureAtMicrohponeStand="+str(tempHumMicrophone[0])+"\n")
    record_file.write("humidityAtMicrohponeStand="+str(tempHumMicrophone[1])+"\n")
    record_file.write("temperatureAtMSpeakerStand="+str(tempHumSpeaker[0])+"\n")
    record_file.write("humidityAtSpeakerStand="+str(tempHumSpeaker[1])+"\n")
    record_file.write("tempHumTimestamp="+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S'))+"\n")
    record_file.close()

if __name__ == '__main__':
 main()





  
