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

Following information is given in the [data collection directory][01.data_collection]
- How to construct this system
- How to start recording sounds in a room.
- How to clean records and extract RIRs from that records.
- How to visualize recorded data
  - Recording Position Heatmaps
  - RIR signal waves
- How to compare recorded RIRs (read data) with a pre-existing model's ([FAST-RIR][FAST-RIR]) generated RIR data .

Speakers and microphones are mounted as shown below. 

![RIR Collection System Design][rir_measurement_setup]  

<br>
<br>

---

<br>
<br>

## 2. GENERATING RIR DATA USING MESH, POSITIONS and a GAN MODEL <a name="02"></a>
We took [MESH2IR][MESH2IR] paper as a referrence point, we modified it to find our own model.  
Within this work , we tried to 
- Change models
- Change inputs/outputs
  
We tested the fidelity of generated RIRs using out collected real data (GTU-RIR) as shown below :

![mesh2ir_generate_and_test][mesh2ir_generate_and_test]

We obtained comparison results as follows :

![EXAMPLE_COMPARE_1][EXAMPLE_COMPARE_1]
![EXAMPLE_COMPARE_2][EXAMPLE_COMPARE_2]  

Details are found at [02.data_generation][02.data_generation] directory.




[01.data_collection]: 01.data_collection/README.md
[02.data_generation]: 02.data_generation/README.md
[rir_measurement_setup]: README.md.resources/rir-measurement-setup.png
[mesh2ir_generate_and_test]: README.md.resources/mesh2ir.generate.and.gtu-rir.test.small.png
[FAST-RIR]: https://github.com/anton-jeran/FAST-RIR
[MESH2IR]: https://github.com/anton-jeran/MESH2IR
[EXAMPLE_COMPARE_1]: README.md.resources/example.compare.1.small.png
[EXAMPLE_COMPARE_2]: README.md.resources/example.compare.2.small.png
