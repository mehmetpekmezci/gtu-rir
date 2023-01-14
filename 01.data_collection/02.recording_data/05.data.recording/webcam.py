#!/usr/bin/env python3
from header import *

def takePhoto(TEST_MODE):
     logger.info ('-----takePhoto ')
     photos=[]
    
     if TEST_MODE != "TEST_MODE_NONE" :
        photos.append(cv2.imread('test.jpg'))
        return photos

#     deviceVendorId="0c45"
#     deviceModelId="64ab"

     ##ls /dev/video*
     ##udevadm info -n video2 -q path     

     device="/dev/video0"    
     
#     while device == "" :
#      for i in range(4):
#         process=subprocess.Popen(["udevadm info  /dev/video"+str(i)+"| grep "+deviceModelId+" "],shell=True,stdout=subprocess.PIPE)
#         out,err=process.communicate()
#         model=out.strip().decode('ascii')
#         print(model)
#         process=subprocess.Popen(["udevadm info  /dev/video"+str(i)+"| grep "+deviceModelId+" | wc -l"],shell=True,stdout=subprocess.PIPE)
#         out,err=process.communicate()
#         isFoundModel=out.strip().decode('ascii')
#         print(isFoundModel)
#         if isFoundModel == "1" :
#            process=subprocess.Popen(["udevadm info  /dev/video"+str(i)+"| grep capture | wc -l"],shell=True,stdout=subprocess.PIPE)
#            out,err=process.communicate()
#            isCapture=out.strip().decode('ascii')
#            if isFoundModel == "1" :
#               device="/dev/video"+str(i)
#      if device == "" :
#         process=subprocess.Popen(["lsusb| grep "+deviceModelId+"|cut -d: -f1| awk '{print \"/dev/bus/usb/\"$2\"/\"$4}'"],shell=True,stdout=subprocess.PIPE)
#         out,err=process.communicate()
#         webcamUsbBusPath=out.strip().decode('ascii')
#         subprocess.Popen(["sudo", SCRIPT_DIR+"/usbreset", webcamUsbBusPath])
#         
         
     camera = cv2.VideoCapture(device)
     time.sleep(3)
     if(not camera.isOpened()) :
         camera = cv2.VideoCapture(device)
         time.sleep(3)
         if(not camera.isOpened()) :
           camera = cv2.VideoCapture(device)
           time.sleep(3)
         if(not camera.isOpened()) :
           camera = cv2.VideoCapture(device)
           time.sleep(3)
#         if(not camera.isOpened()) :
#           process=subprocess.Popen(["lsusb| grep "+deviceModelId+"|cut -d: -f1| awk '{print \"/dev/bus/usb/\"$2\"/\"$4}'"],shell=True,stdout=subprocess.PIPE)
#           out,err=process.communicate()
#           webcamUsbBusPath=out.strip().decode('ascii')
#           subprocess.Popen(["sudo", SCRIPT_DIR+"/usbreset", webcamUsbBusPath])
         
     if(camera.isOpened()) :
         return_value,image = camera.read()
         photos.append(image)

     camera.release()
     cv2.destroyAllWindows()
     return photos

       
