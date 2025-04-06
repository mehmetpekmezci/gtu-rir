#!/usr/bin/env python3
import importlib
math        = importlib.import_module("math")
logging     = importlib.import_module("logging")
urllib3     = importlib.import_module("urllib3")
tarfile     = importlib.import_module("tarfile")
csv         = importlib.import_module("csv")
glob        = importlib.import_module("glob")
sys         = importlib.import_module("sys")
os          = importlib.import_module("os")
argparse    = importlib.import_module("argparse")
np          = importlib.import_module("numpy")
librosa     = importlib.import_module("librosa")
pandas      = importlib.import_module("pandas")
time        = importlib.import_module("time")
random      = importlib.import_module("random")
datetime    = importlib.import_module("datetime")
#keras       = importlib.import_module("keras")
gc          = importlib.import_module("gc")
wave        = importlib.import_module("wave")
scipy       = importlib.import_module("scipy")
#cv2         = importlib.import_module("cv2") 
configparser= importlib.import_module("configparser")
subprocess  = importlib.import_module("subprocess")
pickle      = importlib.import_module("pickle")
matplotlib  = importlib.import_module("matplotlib")



gtu_rir_data_pickle_file=str(sys.argv[1]).strip()

list_of_room_ids=[ "room-207",  "room-208", "room-conferrence01", "room-sport01", "room-sport02", "room-z02", "room-z04" ,"room-z06" ,"room-z10", "room-z11" ,"room-z23" ]

def load_rir_data(roomId):
 rir_data=[]
 if os.path.exists(gtu_rir_data_pickle_file) :
         rir_data_file=open(gtu_rir_data_pickle_file+"."+roomId,'rb')
         rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
 else :
         print(f"{gtu_rir_data_pickle_file} not exists")
         exit(1)

 return rir_data



rir_data_field_numbers={"timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
                                "physicalSpeakerNo":6,"microphoneStandInitialCoordinateX":7,"microphoneStandInitialCoordinateY":8,"microphoneStandInitialCoordinateZ":9,"speakerStandInitialCoordinateX":10,
                                "speakerStandInitialCoordinateY":11,"speakerStandInitialCoordinateZ":12,"microphoneMotorPosition":13,"speakerMotorPosition":14,"temperatureAtMicrohponeStand":15,
                                "humidityAtMicrohponeStand":16,"temperatureAtMSpeakerStand":17,"humidityAtSpeakerStand":18,"tempHumTimestamp":19,"speakerRelativeCoordinateX":20,"speakerRelativeCoordinateY":21,
                                "speakerRelativeCoordinateZ":22,"microphoneStandAngle":23,"speakerStandAngle":24,"speakerAngleTheta":25,"speakerAnglePhi":26,"mic_RelativeCoordinateX":27,"mic_RelativeCoordinateY":28,
                                "mic_RelativeCoordinateZ":29,"mic_DirectionX":30,"mic_DirectionY":31,"mic_DirectionZ":32,"mic_Theta":33,"mic_Phi":34,"essFilePath":35,
                                "roomId":36,"configId":37,"micNo":38, ## THESE VALUES WILL BE PARSED FROM essFilePath
                                "roomWidth":39,"roomHeight":40,"roomDepth":41, ## THESE VALUES WILL BE RETREIVED FROM ROOM PREOPERTIES                              
                                "rt60":42, ## RT60 will be calculated                              
                                "rirData":43 ## will be loaded from wav file   
                              }


for selectedRoomId in list_of_room_ids:
    print(f"loading {selectedRoomId}")
    room_data=load_rir_data(selectedRoomId)

    print(f"Analyzing {selectedRoomId}")

    for dataline1 in room_data:
       speakerMotorIterationNo_1=dataline1[rir_data_field_numbers['speakerMotorIterationNo']]
       microphoneMotorIterationNo_1=dataline1[rir_data_field_numbers['microphoneMotorIterationNo']]
       physicalSpeakerNo_1=dataline1[rir_data_field_numbers['physicalSpeakerNo']]
       micNo_1=dataline1[rir_data_field_numbers['micNo']]

       microphoneCoordinatesX_1=float(dataline1[rir_data_field_numbers['microphoneStandInitialCoordinateX']]) +float(dataline1[rir_data_field_numbers['mic_RelativeCoordinateX']])  
       microphoneCoordinatesY_1=float(dataline1[rir_data_field_numbers['microphoneStandInitialCoordinateY']]) +float(dataline1[rir_data_field_numbers['mic_RelativeCoordinateY']])  
       microphoneCoordinatesZ_1=float(dataline1[rir_data_field_numbers['mic_RelativeCoordinateZ']])  

       speakerCoordinatesX_1=float(dataline1[rir_data_field_numbers['speakerStandInitialCoordinateX']]) +float(dataline1[rir_data_field_numbers['speakerRelativeCoordinateX']])  
       speakerCoordinatesY_1=float(dataline1[rir_data_field_numbers['speakerStandInitialCoordinateY']]) +float(dataline1[rir_data_field_numbers['speakerRelativeCoordinateY']])  
       speakerCoordinatesZ_1=float(dataline1[rir_data_field_numbers['speakerRelativeCoordinateZ']])  

       for dataline2 in room_data:
           speakerMotorIterationNo_2=dataline2[rir_data_field_numbers['speakerMotorIterationNo']]
           microphoneMotorIterationNo_2=dataline2[rir_data_field_numbers['microphoneMotorIterationNo']]
           physicalSpeakerNo_2=dataline2[rir_data_field_numbers['physicalSpeakerNo']]
           micNo_2=dataline2[rir_data_field_numbers['micNo']]

           microphoneCoordinatesX_2=float(dataline2[rir_data_field_numbers['microphoneStandInitialCoordinateX']]) +float(dataline2[rir_data_field_numbers['mic_RelativeCoordinateX']])  
           microphoneCoordinatesY_2=float(dataline2[rir_data_field_numbers['microphoneStandInitialCoordinateY']]) +float(dataline2[rir_data_field_numbers['mic_RelativeCoordinateY']])  
           microphoneCoordinatesZ_2=float(dataline2[rir_data_field_numbers['mic_RelativeCoordinateZ']])  

           speakerCoordinatesX_2=float(dataline2[rir_data_field_numbers['speakerStandInitialCoordinateX']]) +float(dataline2[rir_data_field_numbers['speakerRelativeCoordinateX']])  
           speakerCoordinatesY_2=float(dataline2[rir_data_field_numbers['speakerStandInitialCoordinateY']]) +float(dataline2[rir_data_field_numbers['speakerRelativeCoordinateY']])  
           speakerCoordinatesZ_2=float(dataline2[rir_data_field_numbers['speakerRelativeCoordinateZ']])  

           if micNo_1!=micNo_2:
              if  ( 
                      abs(microphoneCoordinatesX_1-microphoneCoordinatesX_2)<4 and  ## 5 == 5 CM
                      abs(microphoneCoordinatesY_1-microphoneCoordinatesY_2)<4 and
                      abs(microphoneCoordinatesZ_1-microphoneCoordinatesZ_2)<4 and
                      abs(speakerCoordinatesX_1-speakerCoordinatesX_2)<4 and
                      abs(speakerCoordinatesY_1-speakerCoordinatesY_2)<4 and
                      abs(speakerCoordinatesZ_1-speakerCoordinatesZ_2)<4 
                  ):
                  print (f"(spkItrNo-micItrNo-SpkNo-micNo)  {speakerMotorIterationNo_1}-{microphoneMotorIterationNo_1}-{physicalSpeakerNo_1}-{micNo_1}  and {speakerMotorIterationNo_2}-{microphoneMotorIterationNo_2}-{physicalSpeakerNo_2}-{micNo_2} are close : ")
                  print (f"mic_XYZ_Diff={abs(microphoneCoordinatesX_1-microphoneCoordinatesX_2)}-{abs(microphoneCoordinatesY_1-microphoneCoordinatesY_2)}-{abs(microphoneCoordinatesZ_1-microphoneCoordinatesZ_2)}  spk_XYZ_Diff={abs(speakerCoordinatesX_1-speakerCoordinatesX_2)}-{abs(speakerCoordinatesY_1-speakerCoordinatesY_2)}-{abs(speakerCoordinatesZ_1-speakerCoordinatesZ_2)}")






'''

def plotWav(real_data,generated_data,MSE,SSIM,glitch_points,MFCC_MSE,MFCC_SSIM,MFCC_CROSS_ENTROPY,title,show=False,saveToPath=None):
     pt1=time.time()
     plt.clf()

     #plt.subplot(1,1,1)
     minValue=np.min(real_data)
     minValue2=np.min(generated_data)
     if minValue2 < minValue:
        minValue=minValue2

     pt2=time.time()
     #plt.text(2600, minValue+abs(minValue)/11, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}\nGLITCH={int(len(glitch_points))}\nMFCC_MSE={float(MFCC_MSE):.4f}\nMFCC_SSIM={float(MFCC_SSIM):.4f}\nMFCC_CROSS_ENTROPY={float(MFCC_CROSS_ENTROPY):.4f}", style='italic',
     plt.text(2600, minValue+abs(minValue)/11, f"MSE={float(MSE):.4f}\nSSIM={float(SSIM):.4f}\nGLITCH={int(len(glitch_points))}", style='italic',
        bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10})


     #plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
     #plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
     #plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
     #    fontsize=20)

     pt3=time.time()
     #plt.plot(real_data,color='r', label='real_data')
     plt.plot(real_data,color='#101010', label='real_data')
     pt4=time.time()
     plt.plot(generated_data,color='#909090', label='generated_data')
     plt.title(title)
     pt5=time.time()
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend(loc = "upper right")
     pt6=time.time()

     x=glitch_points
     y=generated_data[x]
     plt.scatter(x,y,color="black")

     if show :
        plt.show()
     if saveToPath is not None :
        plt.savefig(saveToPath)

     #plt.close()

def allignVertically(self,generated_data,real_data):
#         return generated_data,real_data

         generated_data_max=np.max(np.abs(generated_data))
         real_data_max=np.max(np.abs(real_data))
         generated_data=generated_data/generated_data_max
         real_data=real_data/real_data_max
         return generated_data,real_data


def allignHorizontally(self,generated_data,real_data):
         t1=time.time() 
         max_point_index_within_first_1000_points_real_data=self.getLocalArgMax(1000,real_data) #np.argmax(real_data[0:1000])
         max_point_index_within_first_1000_points_generated_data=self.getLocalArgMax(1000,generated_data)#np.argmax(generated_data[0:1000])

         #print("max_point_index_within_first_1000_points_real_data"+str(max_point_index_within_first_1000_points_real_data))
         #print("max_point_index_within_first_1000_points_generated_data"+str(max_point_index_within_first_1000_points_generated_data))

         diff=int(abs(max_point_index_within_first_1000_points_real_data-max_point_index_within_first_1000_points_generated_data)/2)

         if diff > 0 :
           if    max_point_index_within_first_1000_points_real_data > max_point_index_within_first_1000_points_generated_data :
                 new_generated_data=np.zeros(generated_data.shape)
                 new_generated_data[diff:]=generated_data[:-diff]
                 generated_data=new_generated_data

                 new_real_data=np.zeros(real_data.shape)
                 new_real_data[:-diff]=real_data[diff:]
                 real_data=new_real_data
           else :
                 new_generated_data=np.zeros(generated_data.shape)
                 #new_generated_data[diff:]=generated_data[:-diff]
                 new_generated_data[:-diff]=generated_data[diff:]
                 generated_data=new_generated_data

                 new_real_data=np.zeros(real_data.shape)
                 #new_real_data[:-diff]=real_data[diff:]
                 new_real_data[diff:]=real_data[:-diff]
                 real_data=new_real_data

         
         #print("0.MAX ALLIGNMENT : np.argmax(real_data[0:1000]):"+str(self.getLocalArgMax(1000,real_data)))
         #print("0.MAX ALLIGNMENT : np.argmax(generated_data[0:1000]):"+str(self.getLocalArgMax(1000,generated_data)))

         localArgMaxReal=self.getLocalArgMax(1000,real_data)
         localArgMaxGenerted=self.getLocalArgMax(1000,generated_data)
         diff=1
         if localArgMaxReal ==localArgMaxGenerted+diff :
                  new_generated_data=np.zeros(generated_data.shape)
                  new_generated_data[diff:]=generated_data[:-diff]
                  generated_data=new_generated_data

         elif  localArgMaxGenerted == localArgMaxReal+diff:
                  new_generated_data=np.zeros(generated_data.shape)
                  #new_generated_data[diff:]=generated_data[:-diff]
                  new_generated_data[:-diff]=generated_data[diff:]
                  generated_data=new_generated_data

                  
         t2=time.time() 
         print(f"DeltaT.allignHorizontally={t2-t1}")
         print("1.MAX ALLIGNMENT : np.argmax(real_data[0:1000]):"+str(self.getLocalArgMax(1000,real_data)))
         print("1.MAX ALLIGNMENT : np.argmax(generated_data[0:1000]):"+str(self.getLocalArgMax(1000,generated_data)))

         #real_data1=librosa.resample(self.rir_data[i][-1], orig_sr=44100, target_sr=sr)
         #real_data1=real_data1[:generated_data.shape[0]]
         #print("1.MAX ALLIGNMENT : np.argmax(real_data1[0:1000]):"+str(self.getLocalArgMax(1000,real_data1)))
         # test edildi problem yok :)
         return generated_data,real_data
 
 
 
         
def diffBetweenGeneratedAndRealRIRData(self):

     t0=time.time() 
     print("diffBetweenGeneratedAndRealRIRData is started at : "+str(t0))

     if  os.path.exists( self.report_dir+"/."+self.selected_room_id+".wavesAndSpectrogramsGenerated") :
        print("wavesAndSpectrograms already generated for "+self.selected_room_id)
        return
        
     sr=16000
     ## STRUCTURAL SIMILARITY  librosa.segment.cross_similarity
     ## https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
     ## https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html
     ## https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
     ## https://github.com/VainF/pytorch-msssim
     ## https://github.com/francois-rozet/piqa
     
     
     #https://www.tensorflow.org/api_docs/python/tf/image/ssim
     
     
     
     #https://www.kaggle.com/code/msripooja/steps-to-convert-audio-clip-to-spectrogram
     #https://www.frank-zalkow.de/en/create-audio-spectrograms-with-python.html  ## this is with STFT
     #https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3
     #https://analyticsindiamag.com/hands-on-guide-to-librosa-for-handling-audio-files/
     #https://dsp.stackexchange.com/questions/72027/python-audio-analysis-which-spectrogram-should-i-use-and-why
     
    
     #MSE
     #SSIM
     #MFCC-MSE
     #MFCC-SSIM
     #MFCC-CROSS_ENTROPY
     # (MFCC-MSE + MFCC-SSIM) -- no need.

     print("len(self.rir_data):{len(self.rir_data)}")

     for i in range(len(self.rir_data)):
       t1=time.time() 
       dataline=self.rir_data[i] 
       essFilePath=str(dataline[int(self.rir_data_field_numbers['essFilePath'])])
       roomId=dataline[int(self.rir_data_field_numbers['roomId'])] 
       if roomId != self.selected_room_id :
             continue

       print(f"roomId={roomId}, self.selected_room_id={self.selected_room_id}")

       configId=dataline[int(self.rir_data_field_numbers['configId'])] 
       roomWorkDir=self.report_dir+"/"+roomId+"/"+configId
       rt60=str(self.rir_data[i][int(self.rir_data_field_numbers['rt60'])])

       speakerIterationNo=int(dataline[int(self.rir_data_field_numbers['speakerMotorIterationNo'])])
       microphoneIterationNo=int(dataline[int(self.rir_data_field_numbers['microphoneMotorIterationNo'])])
       physicalSpeakerNo=int(dataline[int(self.rir_data_field_numbers['physicalSpeakerNo'])]) 
       micNo=dataline[int(self.rir_data_field_numbers['micNo'])] 

       record_name = f"SPEAKER_ITERATION-{speakerIterationNo}-MICROPHONE_ITERATION-{microphoneIterationNo}-PHYSICAL_SPEAKER_NO-{physicalSpeakerNo}-MICROPHONE_NO-{micNo}"
         
       wave_name=record_name+".wav"
          
       #print(roomWorkDir+"/"+wave_name+" filename="+essFilePath+"  rir_data rt60 : "+rt60)
        
       try:
         
         generated_data,rate=librosa.load(roomWorkDir+"/"+wave_name,sr=sr,mono=True)
         generated_data=generated_data[0:3500]  
         real_data=librosa.resample(self.rir_data[i][-1], orig_sr=44100, target_sr=sr) 
         real_data=real_data[:generated_data.shape[0]]
        
         print("-----------------")
         print("Non aligned data :")
         print(f"np.max(generated_data)={np.max(generated_data)}")
         print(f"np.max(real_data)={np.max(real_data)}")
         print(f"np.min(generated_data)={np.min(generated_data)}")
         print(f"np.min(real_data)={np.min(real_data)}")
         print("-----------------")
         
         generated_data,real_data=self.allignHorizontally(generated_data,real_data)         
         
         ######### BEGIN : YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)
         generated_data=generated_data-np.sum(generated_data)/generated_data.shape[0]
         ## bu sekilde ortalamasi 0'a denk gelecek
         ######### END: YATAY ESITLEME (dikey esitleme zaten maksimum noktalarini esitliyerek yapilmisti)

         generated_data,real_data=self.allignVertically(generated_data,real_data)         
         
         generated_spectrogram=self.getSpectrogram(generated_data)
         
         real_spectrogram=self.getSpectrogram(real_data)

         t1=time.time() 
         MSE=np.square(np.subtract(real_data,generated_data)).mean()
         t2=time.time() 
         print(f"DeltaT.MSE={t2-t1}")
         
         t1=time.time() 
         MFCC_CROSS_ENTROPY=TF.cross_entropy(torch.from_numpy(real_data), torch.from_numpy(generated_data)).item()
         #MFCC_CROSS_ENTROPY=-np.sum(real_data * np.log(generated_data))
         #MFCC_CROSS_ENTROPY=-np.sum(real_spectrogram * np.log(generated_spectrogram))
         print(f"MFCC_CROSS_ENTROPY={MFCC_CROSS_ENTROPY}")
         t2=time.time() 
         print(f"DeltaT.MFCC_CROSS_ENTROPY={t2-t1}")
         
         t1=time.time() 
         MFCC_MSE=np.square(np.subtract(real_spectrogram,generated_spectrogram)).mean()
         t2=time.time() 
         print(f"DeltaT.MFCC_MSE={t2-t1}")
         
         t1=time.time() 
         generated_spectrogram=np.reshape(generated_spectrogram,(generated_spectrogram.shape[0],generated_spectrogram.shape[1],1))
         real_spectrogram=np.reshape(real_spectrogram,(real_spectrogram.shape[0],real_spectrogram.shape[1],1))

         generated_spectrogram=np.reshape(generated_spectrogram,(1,1,generated_spectrogram.shape[0],generated_spectrogram.shape[1]))
         real_spectrogram=np.reshape(real_spectrogram,(1,1,real_spectrogram.shape[0],real_spectrogram.shape[1]))
         print(f"np.max(generated_spectrogram)={np.max(generated_spectrogram)}")
         print(f"np.max(real_spectrogram)={np.max(real_spectrogram)}")
         MFCC_SSIM=ssim( torch.Tensor(generated_spectrogram), torch.Tensor(real_spectrogram), data_range=255, size_average=False).item()
         #SSIM=tf.image.ssim(generated_spectrogram_tensor, real_spectrogram_tensor, max_val=max_val_tensor, filter_size=4,filter_sigma=1.5, k1=0.01, k2=0.03).numpy()
         t2=time.time() 
         print(f"DeltaT.MFCC_SSIM={t2-t1}")
         
         t1=time.time() 
         generated_data_tiled=np.tile(generated_data, (2, 1)) ## duplicate 1d data to 2d
         real_data_tiled=np.tile(real_data, (2, 1)) ## duplicate 1d data to 2d

         generated_data_tiled=np.reshape(generated_data_tiled,(1,1,generated_data_tiled.shape[0],generated_data_tiled.shape[1]))
         real_data_tiled=np.reshape(real_data_tiled,(1,1,real_data_tiled.shape[0],real_data_tiled.shape[1]))

         generated_data_tensor=torch.from_numpy(generated_data_tiled)
         real_data_tensor=torch.from_numpy(real_data_tiled)

         print(f"np.max(generated_data)={np.max(generated_data)}")
         print(f"np.max(real_data)={np.max(real_data)}")

         #print(f"generated_data.shape={generated_data.shape}")
         #print(f"real_data.shape={real_data.shape}")

         #/usr/local/lib/python3.8/dist-packages/pytorch_msssim/ssim.py

         # data_range  = np.max(real_data)-np.min(real_data) --> bunu bi 2 olarak set ediyoruz.
         #SSIM=ssim(generated_data_tensor,real_data_tensor, data_range=2.0,size_average=True).item()
         SSIM=ssim(generated_data_tensor,real_data_tensor,data_range=4.0,size_average=True).item()
         print(f"SSIM={SSIM}")
         #SSIM=ssim(generated_data_tensor,real_data_tensor, data_range=255, size_average=True).item()
         #SSIM=tf.image.ssim(generated_spectrogram_tensor, real_spectrogram_tensor, max_val=max_val_tensor, filter_size=4,filter_sigma=1.5, k1=0.01, k2=0.03).numpy()
         t2=time.time() 
         print(f"DeltaT.SSIM={t2-t1}")
         
         glitch_points=self.getGlitchPoints(generated_data,real_data)

         #crossCorrelation=self.getCrossCorrelation(generated_data,real_data)

         #title=record_name
         #title=f"RT60-{float(labels_embeddings_batch[9]):.2f}-MX-{float(labels_embeddings_batch[0]):.2f}-MY-{float(labels_embeddings_batch[1]):.2f}-MZ-{float(labels_embeddings_batch[2]):.2f}-SX-{float(labels_embeddings_batch[3]):.2f}-SY-{float(labels_embeddings_batch[4]):.2f}-SZ-{float(labels_embeddings_batch[5]):.2f}"
         title=""

         ## plot only 1 of 10 samples.
         if True or i%10 == 0 :
            self.plotWav(real_data,generated_data,MSE,SSIM,glitch_points,MFCC_MSE,MFCC_SSIM,MFCC_CROSS_ENTROPY,title,saveToPath=roomWorkDir+"/"+record_name+".wave.png")
         
         t1=time.time() 
         f = open(roomWorkDir+"/MSE.db.txt", "a")
         f.write(record_name+"="+str(MSE)+"\n")
         f.close()
         f = open(roomWorkDir+"/SSIM.db.txt", "a")
         f.write(record_name+"="+str(SSIM)+"\n")
         f.close()
         f = open(roomWorkDir+"/MFCC_MSE.db.txt", "a")
         f.write(record_name+"="+str(MFCC_MSE)+"\n")
         f.close()
         f = open(roomWorkDir+"/MFCC_SSIM.db.txt", "a")
         f.write(record_name+"="+str(MFCC_SSIM)+"\n")
         f.close()
         f = open(roomWorkDir+"/MFCC_CROSS_ENTROPY.db.txt", "a")
         f.write(record_name+"="+str(MFCC_CROSS_ENTROPY)+"\n")
         f.close()
         f = open(roomWorkDir+"/GLITCH_COUNT.db.txt", "a")
         f.write(record_name+"="+str(len(glitch_points))+"\n")
         f.close()
         t2=time.time() 
         print(f"DeltaT.SAVE_ALL_TO_FILE=={t2-t1}")
       except:
           print("Exception: roomId="+roomId+", record_name="+record_name)
           traceback.print_exc()

     open( self.report_dir+"/."+self.selected_room_id+".wavesAndSpectrogramsGenerated", 'a').close()   
 
 def getGlitchPoints(self,generated,real):
     t1=time.time()
     INSENSITIVITY=3
     glitchThreshold=np.std(np.abs(real))*INSENSITIVITY
     #glitchThreshold=np.max(real)*1/2
     glitchPoints=[]
     #checkNextN=int(self.reduced_sampling_rate/50)
     #for i in range(len(generated)-checkNextN):
         #if  self.isBiggerThanNextN(i,checkNextN,glitchThreshold,generated,real):
     for i in range(len(generated)):
         if  abs(abs(generated[i])-abs(real[i]) )> glitchThreshold :
             glitchPoints.append(i)
     t2=time.time()
     print(f"DeltaT.getGlitchPoints={t2-t1}")
     return glitchPoints

# def isBiggerThanNextN(self,checkIndex,N,threshold,generated,real):
#     isBigger=True
#     for i in range(N):
#         if abs(generated[checkIndex]-real[checkIndex+i]) < threshold:
#             isBigger=False
#     return isBigger

 def getLocalArgMax(self,limit,data):
     maximum_value=np.max(data[:limit])*4/5 # 20% error threshold for max
     return np.argmax(data[:limit]>=maximum_value)

        
 def localMaxDiffSum(self,signal1,signal2,numberOfChunks=64):
     maxDiffSum=0
     chunkSize=int(self.data_length/numberOfChunks)
     for i in range(numberOfChunks):
         #max1=np.max(np.abs(signal1[i:i+chunkSize]))
         #max2=np.max(np.abs(signal2[i:i+chunkSize]))
         max1=np.mean(np.abs(signal1[i:i+chunkSize]))
         max2=np.mean(np.abs(signal2[i:i+chunkSize]))
         maxDiff=abs(max2-max1)
         maxDiffSum=maxDiffSum+maxDiff
     return maxDiffSum/numberOfChunks
'''
