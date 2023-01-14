#!/usr/bin/env python3
from RIRHeader import *
from RIRLogger import *
from RIRData import *




def main():
  rirLogger=RIRLogger()
  rirData=RIRData(rirLogger.logger)
  rirData.visualizeAllTheData()

  
                   

if __name__ == '__main__':
 main()



