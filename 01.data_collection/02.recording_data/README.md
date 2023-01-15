# GTU-RIR : COLLECTING REAL RIR DATA / RECORDING DATA

How do we use the system to record sound data :
1. We go in a room and place the microphone and speaker stands in a position.
2. We first prepare the recording system using instructions in [01.setting_to_work](01.setting_to_work/README.md).  
2. Then we start recording data using the run.sh script as described in [02.data_recording](02.data_recording/README.md).  

The system collects 2400 data points ( microphone-speaker position based sound records) in 8 hours.

We restart the system when we change microphone/speaker stand positions.

IMPORTANT NOTE : We tried also a translating system ( not rotating but going straight). But the system was not stable because :
   1. It was nearly impossible maintain a straight line, which leads to deviations in coordinates of microphones and speakers.
   2. Moving whole microphone/speaker stand requires more power, this causes the step motor to stop at some point randomly.
   3. Moving the stands by using small wheels mounted at the bottom is very instable because of the floor properties change room-to-room.

 


