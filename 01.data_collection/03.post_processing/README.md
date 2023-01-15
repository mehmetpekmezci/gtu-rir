# GTU-RIR : COLLECTING REAL RIR DATA / POST PROCESSING

After collecting data in $HOME/GTU-RIR-DATA, you may post process the data in another computer having the same dirctory structure.

To post process the data, just run the run.sh file in this directory.

This process is not an unattended process, after running run.sh your input will be expected to determine visually if there was a problem in the movement of the Arms of the microphone/speaker stands.  

The run.sh script flow is :

1. For each record step :
   1. Show a record step's webcam photo.
   2. Expects user input : "Is the speaker/microphone mechanism moved appropirately"
   3. Clean (move to trash) the data if the user enters "n" for the above question.
2. For all the sound files that are not moved to trash:
   1. Clean (move to trash) if SNR value is negative.
   2. Remove the noise from the sound record ( first 2 seconds of each record is only for environment noise records, by sampling frequencies , we remove the noise from whole record. )
   3. Allign transmitted sound data and recorded data.
3. For all records, regenerate csv files (ess_db.csv, song_db.csv)
   1. These CSV files will be used to build a pickle file in 04.data_statistics section.



You may run this script every time you collect and put more data in $HOME/GTU-RIR-DATA directory.  
This script will ommit the record that are already processed, and only focus on the newly added records.  




