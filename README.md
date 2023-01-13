# GTU-RIR
This is a Room Impulse Response (RIR) data collection and generation project.
1. [Collecting real data](#01)
2. [Generating RIR data using MESH, positions and a GAN model](#02)

## 1. COLLECTING REAL RIR DATA <a name="01"></a>

We built a semi-automated RIR collection system to collect real "room impulse response" data in a room.
The system consists of:
1. Speakers (4), 
2. Microphones (6), 
3. Step motors (2) and their controllers, 
4. Microphone Stand (1 custom constructed),
5. Speaker Stand (1 custom constructed),
6. USB hubs (2) and cables,
7. Computer (1) . 
   
Speakers and microphones are mounted as shown below. 
You may find information [here][01.data_collection] about 
- How to construct this system
- How to start recording sounds in a room.
- How to clean records and extract RIRs from that records.
- How to visualize recorded data
  - Recording Position Heatmaps
  - RIR signal waves
- How to compare recorded RIRs (read data) with a pre-existing model's ([FAST-RIR][FAST-RIR]) generated RIR data .

![RIR Collection System Design][rir_measurement_setup]

---



## 2. GENERATING RIR DATA USING MESH, POSITIONS and a GAN MODEL <a name="02"></a>





[01.data_collection]: 01.data_collection/README.md
[rir_measurement_setup]: README.md.resources/rir-measurement-setup.png
[FAST-RIR]: https://github.com/anton-jeran/FAST-RIR