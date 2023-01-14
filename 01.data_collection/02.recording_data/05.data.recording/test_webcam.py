#!/usr/bin/env python3
from header import *
from Record import *
from ess import *
from transmitter import *
from receiver import *
from sensor import *
from webcam import *
from motor import *
from save import *




def main():
    photos=takePhoto("TEST_MODE_NONE")
    cv2.imwrite('test.jpg',photos[0])


if __name__ == '__main__':
 main()





  
