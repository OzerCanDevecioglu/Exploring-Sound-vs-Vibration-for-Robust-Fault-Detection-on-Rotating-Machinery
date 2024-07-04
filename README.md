# Exploring-Sound-vs-Vibration-for-Robust-Fault-Detection-on-Rotating-Machinery


# Project Description
Robust and real-time detection of faults on rotating machinery has become an ultimate objective for predictive maintenance in various industries. Vibration-based Deep Learning (DL) methodologies have become the de facto standard for bearing fault detection as they can produce state-of-the-art detection performances under certain conditions. Despite such particular focus on the vibration signal, the utilization of sound, on the other hand, has been neglected whilst only a few studies have been proposed during the last two decades, all of which were based on a conventional ML approach. One major reason is the lack of a benchmark dataset providing a large volume of both vibration and sound data over several working conditions for different machines and sensor locations. In this study, we address this need by presenting the new benchmark Qatar University Dual-Machine Bearing Fault Benchmark dataset (QU-DMBF), which encapsulates sound and vibration data from two different motors operating under 1080 working conditions overall. Then we draw the focus on the major limitations and drawbacks of vibration-based fault detection due to numerous installation and operational conditions. Finally, we propose the first DL approach for sound-based fault detection and perform comparative evaluations between the sound and vibration over the QU-DMBF dataset. A wide range of experimental results shows that the sound-based fault detection method is significantly more robust than its vibration-based counterpart, as it is entirely independent of the sensor location, cost-effective (requiring no sensor and sensor maintenance), and can achieve the same level of the best detection performance by its vibration-based counterpart. With this study, the QU-DMBF dataset, the optimized source codes in PyTorch, and comparative evaluations are now publicly shared.
[Paper Link](https://arxiv.org/abs/2312.10742)


## Qatar University Dual-Machine Bearing Fault Benchmark Dataset: QU-DMBF

![image](https://user-images.githubusercontent.com/98646583/207285515-23333c67-e1fe-41f3-a339-d39a3cfaeb68.png)

The benchmark dataset used in this study was generated by the researchers at Qatar University from two electric machines. Figure 3 shows the installation of two machines (A and B) along with the sensor locations.  For Machine-A, the setup includes a 3-phase AC motor (brand: VEMAT, Model: 3VTB-90LA(2P), Vicena-Italy), whose input frequency was controlled by a variable frequency drive. The motor used in this experiment supplied 2.2 kW (3 Hp) at a maximum speed of 2840 rpm. The vibration signals were acquired by using 5 high sensitivity, ceramic shear ICP accelerometers (from PCB Piezotronics, Model No. 352C33, 100 mV/g, NY-USA), which were fixed on the same mounting base that supported the load cell (from Omegadyne Inc, Model No. LCM204-50KN, Manchester-UK), and 4 other different location on machine and motor. The readings were controlled by two four-channel NI-9234 sound and vibration input modules at a sampling frequency of 4.096 kHz. It has a ~180 kg weight and the setup has 100x100x40 cm dimensions. Machine-A has the following variations on the working conditions:

-	19 different bearing configurations: 1 healthy, 18 fault cases: 9 with a defect on the outer ring, and 9 with a defect on the inner ring. The defect sizes are varying from 0.35mm to 2.35mm.
-	5 different accelerometer localization: 3 different positions and 2 different directions (radial and axial)
-	2 different load (force) levels: 0.12 kN and 0.20 kN. 
-	3 different speeds: 480, 680, and 1010 RPM. 


Machine B, on the other hand, was originally a Machinery Fault Simulator from SpectraQuest Inc, USA. The setup includes a DC motor of 0.37 kW (0.5 HP), 90 VDC, 5 A, with a maximum rotational speed of 2500 RPM. The original shaft of the machine was removed and replaced by a new and bigger one to accommodate the same bearings used on Machine-A. For the sake of resistance and stability, other supporting mechanical components were also redesigned. The machine has an approximate weight of 50kg and overall dimensions of 100x63x53 cm.   Machine-B has the following variations on the working conditions:

-	19 different bearing configurations: 1 healthy, 9 with a defect on the outer ring, and 9 with a defect on the inner ring. The defect sizes are varying from 0.35mm to 2.35mm.
-	6 different accelerometer positions.
-	A fixed load (force) of 0.15 kN. 
-	5 different speeds: 240, 360, 480, 700, and 1020 RPM
- Full QU-DMBF dataset with user manual can be downloaded from the given [link](https://drive.google.com/drive/folders/1glUH3mLPUowrwi-B0yrIWHWm_ZpNfnBN?usp=share_link)
## Run

#### Train
- Training/Validation/Test dataset for two motors can be downloaded from the given links [MachineA](https://drive.google.com/drive/folders/1y_mT_q-LMwr2xdc2VwU7pE1-79w5vyNA?usp=sharing) and [MachineB](https://drive.google.com/drive/folders/1KcxuGtECtzyvU75Wj4E7WDQh8pLsz3_o?usp=sharing)
- Use Matlab_MachineA and Matlab_MachineA speed data codes to extract sensor locationwise data. a variable in the codes denotes the sensor location. (You can use a=1 for the vibration sensor on the bearing) (Dataset details are given in [sheet](https://docs.google.com/spreadsheets/d/1k7BxKLgmbw2zBb67EOpm1O6m0VX3qjyc/edit?usp=sharing&ouid=115813379435147019116&rtpof=true&sd=true)])
- Use Matlab traintestsplit.m code for prepairing dataset.
- Start training 
```http
  python train.py
```


## Prerequisites
- Pyton 3
- Pytorch
- [FastONN](https://github.com/junaidmalik09/fastonn) 


  
## Results

![image](https://user-images.githubusercontent.com/98646583/207303899-b35256d6-24a3-4f41-9ad4-3c2b2520e0e7.png)

## Citation
If you find this project useful, we would be grateful if you cite this paper：

```http
---
```
