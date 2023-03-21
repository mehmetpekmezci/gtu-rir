# GTU-RIR : COLLECTING REAL RIR DATA / RECORDING DATA / DATA RECORDING SCRIPT

Execute the run.sh file. The recorded data will be in the $HOME/GTI-RIR-DATA directory.  

NOTE: Click on the <a href='https://github.com/mehmetpekmezci/gtu-rir/tree/master/01.data_collection/02.recording_data/02.data_recording'>02.data_recording link </a>  (or simple clone the repository) to see the scripts.

General program execution is :
1. Generate ESS signal 
2. For each microphone rotation step
   1. For each speaker rotation step
      1. For each speaker id (there are 4 speakers)
         1. Start 6 microphone recording parallely (thread)
         2. Transmit the ESS signal from determined speaker id.
         3. Stop 6 microphone recording parallely (thread.join)
         4. Save microphone and speaker position and directions to a txt file.  
   
**NOTE-1** : You may control the recording start time by changing START_HOUR variable value in the main.py file .  
         This is by default 0 (starts immediately, but it would be more convenient to start at night to not to disturb other people, i recommend starting at 21:00). 

**NOTE-2** : The run.sh script automatically checks 
   1. if all devices are functional, 
   2. if all configuration files are in place,
   3. if suspend system services are stopped (sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target)
   4. if the user is configured to be in /etc/sudoers
   5. if grub has "usbcore.autosuspend=-1" configuration
