#!/usr/bin/env python3
from header import *
from microphone_numbers import * 
from Record import *
from ess import *
from transmitter import *
from receiver import *
from sensor import *
from webcam import *
from motor import *
from save import *
from scipy import io as scipyio

## channel0 == leftSignal
## channel1 == rightSignal

def main():


   ess_signal=generate_ess_signal()
   rightEssSignal=generate_right_signal(ess_signal)
   leftEssSignal=generate_left_signal(ess_signal)

   print("Speaker 0 :")
   transmitSignal(0,leftEssSignal,"TEST_MODE_NONE")
   input("press enter")
   print("Speaker 1 :")
   transmitSignal(0,rightEssSignal,"TEST_MODE_NONE")
   input("press enter")
   print("Speaker 2 :")
   transmitSignal(1,rightEssSignal,"TEST_MODE_NONE")
   input("press enter")
   print("Speaker 3 :")
   transmitSignal(1,leftEssSignal,"TEST_MODE_NONE")
   input("press enter")

if __name__ == '__main__':
 main()





  
