from header import *

## https://pypi.org/project/lywsd03mmc/

## determine mac adresses
## sudo hcitool lescan

# LYWSD03MMC


lastReadMicrophoneTemperature=0
lastReadMicrophoneHumidity=0
lastReadSpeakerTemperature=0
lastReadSpeakerHumidity=0


def getMicrophoneTemperatureHumidity():

   global lastReadMicrophoneTemperature,lastReadMicrophoneHumidity
   returnValue=[]
   returnValue.append(0)
   returnValue.append(0)
   
   try:
     client = lywsd03mmc.Lywsd03mmcClient("A4:C1:38:19:CE:E7")
     data = client.data
     returnValue[0]=data.temperature
     returnValue[1]=data.humidity
     lastReadMicrophoneTemperature=data.temperature
     lastReadMicrophoneHumidity=data.humidity
     logger.info('-----Microphone-Temperature: ' + str(data.temperature))
     logger.info('-----Microphone-Humidity: ' + str(data.humidity))
     logger.info('-----Microphone-Battery: ' + str(data.battery))
     logger.info('-----Microphone-Display units: ' + client.units)
   except:
     logger.info('-----Could not connect to the Temperature/Humidity Sensor device of Microphone Stand ...')
     returnValue[0]=lastReadMicrophoneTemperature
     returnValue[1]=lastReadMicrophoneHumidity
     
   #finally:
   #  print("") 

   return returnValue
             
def getSpeakerTemperatureHumidity():

   global lastReadSpeakerTemperature,lastReadSpeakerHumidity
   returnValue=[]
   returnValue.append(0)
   returnValue.append(0)
   
   try:
     client = lywsd03mmc.Lywsd03mmcClient("A4:C1:38:C2:31:15")
     data = client.data
     returnValue[0]=data.temperature
     returnValue[1]=data.humidity
     lastReadSpeakerTemperature=data.temperature
     lastReadSpeakerHumidity=data.humidity
     logger.info('-----Speaker-Temperature: ' + str(data.temperature))
     logger.info('-----Speaker-Humidity: ' + str(data.humidity))
     logger.info('-----Speaker-Battery: ' + str(data.battery))
     logger.info('-----Speaker-Display units: ' + client.units)
   except:
     logger.info('-----Could not connect to the Temperature/Humidity Sensor device of Speaker Stand ...')
     returnValue[0]=lastReadSpeakerTemperature
     returnValue[1]=lastReadSpeakerHumidity
   #finally:
   #  print("") 
   return returnValue
             

