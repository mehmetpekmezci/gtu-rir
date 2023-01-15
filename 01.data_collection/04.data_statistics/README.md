# GTU-RIR : COLLECTING REAL RIR DATA / DATA STATISTICS

Simply run "run.sh" script , the following actions will be done for all data :

1. Generate RIR.pickle.dat file that will contain 
   1. RIR sound wav array, 
   2. 42 other metadata fields written in the [RIRData.py](RIRData.py) file.
2. Generate HeatMaps for the positions of microphones and speakers in a room. 
   1. You may see example heatmaps generated for the GTU-RIR data collected during this project in this [directory](heatmaps)
   2. We produce different heatmap files for speaker and microphone distributions.
3. Generate Wave Plots to visualize the wave data. ( [Examples](waveplots))
   1. Every plot line contains combined 24 plots of waves ( 6 microphones x 4 speakers )