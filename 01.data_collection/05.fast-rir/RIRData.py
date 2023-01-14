#!/usr/bin/env python3


import gc
from RIRHeader import *
import scipy.io.wavfile
from scipy import signal
from scipy import stats
import librosa.display
import librosa
import matplotlib.pyplot as plt


from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)


np.set_printoptions(threshold=sys.maxsize)


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchaudio
import torch


class RIRData :
 def __init__(self, logger,fast_rir_dir):
 
   self.logger  = logger
   self.script_dir=os.path.dirname(os.path.realpath(__file__))
   self.sampling_rate=44100 # 44100 sample points per second
   self.reduced_sampling_rate=16000
   self.rir_seconds=2
   self.track_length=self.rir_seconds*self.sampling_rate 
   self.final_sound_data_length=int(self.track_length/self.rir_seconds)
   self.number_of_microphones=6
   self.number_of_speakers=4
   self.data_dir=self.script_dir+'/../../../data/single-speaker/'
   self.fast_rir_eval_yml='/var/tmp/fast_rir_eval.yml'
   self.fast_rir_dir=fast_rir_dir
   self.fast_rir_training_data_dir=fast_rir_dir+'/data/Medium_Room'
   self.roomProperties={}
   self.rooms_and_configs={}
   self.data_length=4096
   
   self.SPECTROGRAM_DIM=11

          
   self.rir_data_file_path=self.data_dir+"/RIR.pickle.dat"
   self.fast_rir_data_input_file_path=self.data_dir+"/RIR.fast_rir_inputs.pickle"
   self.fast_rir_data_input_ess_matching_file_path=self.data_dir+"/RIR.fast_rir_inputs_ESS_matching.pickle"
   
   self.rir_data=[]  ##  "RIR.dat" --> list of list [34]

   if  os.path.exists( self.rir_data_file_path) :
         rir_data_file=open(self.rir_data_file_path,'rb')
         self.rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
   else :
         exit(1)
                  
   self.rir_data_field_numbers={"timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
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
                             
   ## essFilePath =   <room_id> / <config_id> / <spkstep-SPKSTEPNO-micstep-MICSTEPNO-spkno-SPKNO> / receivedEssSignal-MICNO.wav
   
   self.transmittedEssWav=None
                                 
   # micNo
   #
   #    5          1
   #    |          |
   # 4-------||-------0
   #    |    ||    |
   #    6    ||    2                    


   # physicalSpeakerNo
   #              
   # 3---2---||
   #         || \    
   #         ||   1                      
   #         ||     \                      
   #         ||      0                    

  






  

             
 def generateFastRIRInputs(self):


          if  os.path.exists( self.fast_rir_data_input_file_path) :
              os.remove(self.fast_rir_data_input_file_path) 
          if  os.path.exists( self.fast_rir_data_input_ess_matching_file_path) :
              os.remove(self.fast_rir_data_input_ess_matching_file_path) 

          max_dimension = 5

          self.fastRirInputData=[]

          for dataline in self.rir_data:
          
              CENT=100 ## M / CM 
          
              roomDepth=float(dataline[int(self.rir_data_field_numbers['roomDepth'])])/CENT # CM to M
              roomWidth=float(dataline[int(self.rir_data_field_numbers['roomWidth'])])/CENT # CM to M
              roomHeight=int(dataline[int(self.rir_data_field_numbers['roomHeight'])])/CENT # CM to M
                  
              microphoneCoordinatesX=float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateX'])])/CENT +float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateX'])])/CENT # CM to M 
              microphoneCoordinatesY=float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateY'])])/CENT +float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateY'])])/CENT # CM to M
              microphoneCoordinatesZ=float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateZ'])])/CENT

              speakerCoordinatesX=float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateX'])])/CENT +float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateX'])])/CENT # CM to M
              speakerCoordinatesY=float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateY'])])/CENT +float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateY'])])/CENT # CM to M
              speakerCoordinatesZ=float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateZ'])])/CENT

              rt60=float(dataline[int(self.rir_data_field_numbers['rt60'])])

              fastRirDataline=[microphoneCoordinatesX,microphoneCoordinatesY,microphoneCoordinatesZ,speakerCoordinatesX,speakerCoordinatesY,speakerCoordinatesZ,roomDepth,roomWidth,roomHeight,rt60]

              
              #print(fastRirDataline)
               
              fastRirDataline =np.divide(fastRirDataline,max_dimension)-1
              
              #fastRirDataline=fastRirDataline/2
             
              
              #fastRirDatalineAsStr=s = "_".join([str(i) for i in fastRirDataline])
              essFilePath=dataline[int(self.rir_data_field_numbers['essFilePath'])]

              if  True :
             
#              if  6 <= roomDepth and roomDepth <= 12 and \
#                  8 <= roomWidth and roomWidth <= 12 and \
#                  2.5 <= roomHeight and roomHeight <= 3.5 and \
#                  1 <= microphoneCoordinatesX and microphoneCoordinatesX <= roomDepth  and \
#                  1 <= microphoneCoordinatesY and microphoneCoordinatesY <= roomWidth  and \
#                  1 <= speakerCoordinatesX and speakerCoordinatesX <= roomDepth and \
#                  1 <= speakerCoordinatesY and speakerCoordinatesY <= roomWidth and \
#                  0.2 <= rt60 and rt60 <= 2:
                  
#              if  6 <= roomDepth and roomDepth <= 10.5 and \
#                  6 <= roomWidth and roomWidth <= 10.5 and \
#                  2.5 <= roomHeight and roomHeight <= 3.5 and \
#                  6 <= microphoneCoordinatesX and microphoneCoordinatesX <= 10.5 and \
#                  2 <= microphoneCoordinatesY and microphoneCoordinatesY <= 8 and \
#                  8 <= speakerCoordinatesX and speakerCoordinatesX <= 10 and \
#                  6 <= speakerCoordinatesY and speakerCoordinatesY <= 9 and \
#                  0.5 <= rt60 and rt60 <= 1.0 :

                 self.fastRirInputData.append(fastRirDataline)

              


                 #if 2 > rt60 and rt60 > 1.5 :
                 #   print(fastRirDataline)
                 #   print(essFilePath)


          
          #for i in range(len(self.fastRirInputData)):
          #   print(f"self.fastRirInputData[{i}]={(self.fastRirInputData[i]+1)*5}")
              
          print("len(self.fastRirDataline)="+str(len(self.fastRirInputData)))
          #exit(0)
          
          with open(self.fast_rir_data_input_file_path, 'wb') as f:
              pickle.dump(self.fastRirInputData, f, protocol=2)     
              f.close()               

          #exit(0)                        
              #if not os.path.exists(self.data_dir+"/"+essFilePath+".ir.wav"):
              # if not os.path.exists(self.data_dir+"/"+essFilePath+".ir.wav.bz2") :
              #    self.logger.info (self.data_dir+"/"+essFilePath+" file does not exist (neither *.bz2 file exists) ...")
              #    continue   


 def triggerFastRIRToGenerateRIRsAndT60s(self):
 
    if  os.path.exists( self.fast_rir_dir+"/code_new/Generated_RIRs/.triggerFastRIRToGenerateRIRsAndT60s_OK") :
        print("triggerFastRIRToGenerateRIRsAndT60s already generated")
        return
         
    if  os.path.exists( self.fast_rir_eval_yml) :
        os.remove(self.fast_rir_eval_yml) 

    record_file = open(self.fast_rir_eval_yml, "w")
    record_file.write("CONFIG_NAME: 'eval'\n")
    record_file.write("DATASET_NAME: 'RIR'\n")
    record_file.write("EMBEDDING_TYPE: 'cnn-rnn'\n")
    record_file.write("GPU_ID: '0,1'\n")
#    record_file.write("NET_G: '../output/RIR_stageI_2022_10_06_10_47_19/Model/netG_epoch_1400.pth'\n")
#    record_file.write("NET_G: '../generate/netG_epoch_77.pure_ssim.pth'\n")
#    record_file.write("NET_G: '../generate/pure_ssim_netG_epoch_303.pth'\n")
    record_file.write("NET_G: '../generate/netG_epoch_82.fastrir.mse.mfcc.pth'\n")
    
#    record_file.write("NET_G: '../generate/netG_epoch_150.ssim_only.pth'\n")
    #record_file.write("NET_G: '../generate/netG_epoch_140_05mse_05ssim.pth'\n")
    #record_file.write("NET_G: '../generate/netG_epoch_200_02mse_08ssim.pth'\n")
#    record_file.write("NET_G: '../generate/netG_epoch_276_02mse_08ssim.pth'\n")
#    record_file.write("NET_G: '../generate/netG_epoch_242.pth_default'\n")
    record_file.write("DATA_DIR: '"+self.fast_rir_training_data_dir+"'\n")
    record_file.write("EVAL_DIR: '"+self.fast_rir_data_input_file_path+"'\n")
    record_file.write("WORKERS: 4\n")
    record_file.write("RIRSIZE: 4096\n")
    record_file.write("STAGE: 1\n")
    record_file.write("TRAIN:\n")
    record_file.write("    FLAG: False\n")
    record_file.write("    BATCH_SIZE: 16\n")
    record_file.write("GAN:\n")
    record_file.write("    CONDITION_DIM: 10\n")
    record_file.write("    DF_DIM: 96\n")
    #record_file.write("    DF_DIM: 16\n")
    record_file.write("    GF_DIM: 256\n")
    #record_file.write("    GF_DIM: 64\n")
    record_file.write("TEXT:\n")
    record_file.write("    DIMENSION: 10\n")
    record_file.close()

    p=subprocess.Popen(["python3 main.py --cfg "+self.fast_rir_eval_yml+" --gpu 0"],cwd=self.fast_rir_dir+"/code_new",shell=True)
    p.wait()
    
    open(self.fast_rir_dir+"/code_new/Generated_RIRs/.triggerFastRIRToGenerateRIRsAndT60s_OK", 'a').close()

 def plotWav(self,real_data,generated_data,MSE,SSIM,LMDS,title,show=False,saveToPath=None):
     #plt.clf()
     
     plt.subplot(1,1,1)
     minValue=np.min(real_data)
     minValue2=np.min(generated_data)
     if minValue2 < minValue:
        minValue=minValue2
        
     plt.text(2800, minValue+0.1, 'MSE='+str(MSE)+"\nSSIM="+str(SSIM)+"\nLMDS="+str(LMDS), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        
        
     #plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
     #plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
     #plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
     #    fontsize=20)   
        
     plt.plot(real_data,color='r', label='real_data')
     plt.plot(generated_data,color='b', label='generated_data')
     plt.title(title)
     plt.xlabel('Time')
     plt.ylabel('Amlpitude')
     plt.legend()
     if show :
        plt.show()
     if saveToPath is not None :
        plt.savefig(saveToPath)
        
     plt.close()    
         
         
 def plotSpectrogram(self,title,power_to_db,sr,show=False,saveToPath=None):
 
 
     
     
     ###plt.figure(figsize=(8, 7))
     fig, ax = plt.subplots()
     #fig.set_figwidth(self.SPECTROGRAM_DIM)
     #fig.set_figheight(self.SPECTROGRAM_DIM)
     #img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length,ax=ax)
     img=librosa.display.specshow(power_to_db, sr=sr, x_axis='time',  cmap='magma',  ax=ax)
     
     
     fig.colorbar(img, ax=ax,label='dB')
     #plt.title('Mel-Spectrogram (dB)'+title, fontdict=dict(size=18))
     plt.title('MFCC '+title, fontdict=dict(size=18))
     plt.xlabel('', fontdict=dict(size=15))
     plt.ylabel('', fontdict=dict(size=15))
     #plt.axis('off')
     
     
     ###if show :
     ###   plt.show()
     ###if saveToPath is not None :
     
     ###plt.savefig(saveToPath, bbox_inches='tight', pad_inches=0)
     plt.savefig(saveToPath)
        
     plt.close()    
         
 def getSpectrogram(self,data,title=None,saveToPath=None):
         sample_rate = 16000 ; frame_length=1024 ; frame_step=256 ; fft_length=2048 ;  fmax=8000 ; fmin=0;num_mfccs=40

         do_also_librosa_for_comparison=False
         
         if do_also_librosa_for_comparison:
           mfcc_librosa = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=num_mfccs)
           if saveToPath is not None :
            self.plotSpectrogram(title+"_librosa",mfcc_librosa,sample_rate,saveToPath=saveToPath+".librosa.png")

         #self.mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},).to("cuda")
         mfcc_transform_fn=torchaudio.transforms.MFCC(sample_rate=sample_rate,n_mfcc=num_mfccs,melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": num_mfccs, "center": False},)
         
         
         #data=np.reshape(data,(1,1,data.shape[0]))
         
         
         #print(data.shape)

         
         mfccs= mfcc_transform_fn( torch.Tensor(data) ).numpy()
         if saveToPath is not None :
            self.plotSpectrogram(title,mfccs,sample_rate,saveToPath=saveToPath)
         return mfccs
 



 
 
 
 
 
         
 def diffBetweenGeneratedAndRealRIRData(self):


     workDir=self.fast_rir_dir+"/code_new/Generated_RIRs_report/"
     
     if  os.path.exists( workDir+"/.wavesAndSpectrogramsGenerated") :
        print("wavesAndSpectrograms already generated")
        return
        
     sr=16000
     n_fft=2048
     hop_length=512
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
     
     
     for i in range(len(self.fastRirInputData)):
         
         essFilePath=str(self.rir_data[i][int(self.rir_data_field_numbers['essFilePath'])])
         roomId=str(self.rir_data[i][int(self.rir_data_field_numbers['roomId'])])
         roomWorkDir=workDir+"/"+roomId
         if not os.path.exists(roomWorkDir):
            os.makedirs(roomWorkDir) 
         rt60=str(self.rir_data[i][int(self.rir_data_field_numbers['rt60'])])
         
         
         
         labels_embeddings_batch=(self.fastRirInputData[i]+1)*5
         record_name = f"RIR-DEPTH-{labels_embeddings_batch[6]}-WIDTH-{labels_embeddings_batch[7]}-HEIGHT-{labels_embeddings_batch[8]}-RT60-{labels_embeddings_batch[9]}-MX-{labels_embeddings_batch[0]}-MY-{labels_embeddings_batch[1]}-MZ-{labels_embeddings_batch[2]}-SX-{labels_embeddings_batch[3]}-SY-{labels_embeddings_batch[4]}-SZ-{labels_embeddings_batch[5]}-{i}"
         wave_name=record_name+".wav"
          
 
                
         print(roomWorkDir+"/"+wave_name+" filename="+essFilePath+"  rir_data rt60 : "+rt60)
         

         generated_data,rate=librosa.load(self.fast_rir_dir+"/code_new/Generated_RIRs/"+wave_name,sr=sr,mono=True)
         
         real_data=librosa.resample(self.rir_data[i][-1], orig_sr=44100, target_sr=sr) 
         real_data=real_data[:generated_data.shape[0]]
         
         
         ### ALLIGN START
         max_point_index_within_first_1000_points_real_data=np.argmax(real_data[0:1000])
         max_point_index_within_first_1000_points_generated_data=np.argmax(generated_data[0:1000])
         
         print("max_point_index_within_first_1000_points_real_data"+str(max_point_index_within_first_1000_points_real_data))
         print("max_point_index_within_first_1000_points_generated_data"+str(max_point_index_within_first_1000_points_generated_data))

         diff=int(abs(max_point_index_within_first_1000_points_real_data-max_point_index_within_first_1000_points_generated_data)/2)
         if diff > 0 :      
           new_generated_data=np.zeros(generated_data.shape)
           new_generated_data[diff:]=generated_data[:-diff]
           generated_data=new_generated_data
            
           new_real_data=np.zeros(real_data.shape)
           new_real_data[:-diff]=real_data[diff:]
           real_data=new_real_data

         print("np.argmax(real_data[0:1000])"+str(np.argmax(real_data[0:1000])))
         print("np.argmax(generated_data[0:1000])"+str(np.argmax(generated_data[0:1000])))
         
         
         generated_data_sum=np.sum(np.abs(generated_data))
         real_data_sum=np.sum(np.abs(real_data))
         ratio_sum=real_data_sum/generated_data_sum
         
         #generated_data_max=np.max(generated_data)
         #real_data_max=np.max(real_data)
         #ratio_max=real_data_max/generated_data_max
         
         generated_data=generated_data*ratio_sum
         
         #print("ratio_max="+str(ratio_max)+"  ratio_sum="+str(ratio_sum))
         
         #print("np.max(real_data[0:1000])"+str(np.max(real_data[0:1000])))
         #print("np.max(generated_data[0:1000])"+str(np.max(generated_data[0:1000])))


         ### ALLIGN END
         
         
         
         generated_spectrogram=self.getSpectrogram(generated_data,title="Generated "+record_name+" "+essFilePath,saveToPath=roomWorkDir+"/"+record_name+"-generated.spectrogram.png")
         generated_spectrogram=np.reshape(generated_spectrogram,(generated_spectrogram.shape[0],generated_spectrogram.shape[1],1))
         
         real_spectrogram=self.getSpectrogram(real_data,title="Real "+record_name+" "+essFilePath,saveToPath=roomWorkDir+"/"+record_name+"-real.spectrogram.png")
         real_spectrogram=np.reshape(real_spectrogram,(real_spectrogram.shape[0],real_spectrogram.shape[1],1))
        
         MSE=np.square(np.subtract(real_data,generated_data)).mean()
         
         
         #generated_spectrogram_tensor=tf.convert_to_tensor(generated_spectrogram)
         
         #real_spectrogram_tensor=tf.convert_to_tensor(real_spectrogram)
         
         #max_val_tensor=tf.convert_to_tensor(np.max(real_spectrogram))

         generated_spectrogram=np.reshape(generated_spectrogram,(1,1,generated_spectrogram.shape[0],generated_spectrogram.shape[1]))
         real_spectrogram=np.reshape(real_spectrogram,(1,1,real_spectrogram.shape[0],real_spectrogram.shape[1]))
         
         SSIM=ssim( torch.Tensor(generated_spectrogram), torch.Tensor(real_spectrogram), data_range=255, size_average=True).item()
         
         #SSIM=tf.image.ssim(generated_spectrogram_tensor, real_spectrogram_tensor, max_val=max_val_tensor, filter_size=4,filter_sigma=1.5, k1=0.01, k2=0.03).numpy()
 
         
         LMDS=self.localMaxDiffSum(real_data,generated_data)
         
         
         #title=record_name
         title=f"RT60-{float(labels_embeddings_batch[9]):.2f}-MX-{float(labels_embeddings_batch[0]):.2f}-MY-{float(labels_embeddings_batch[1]):.2f}-MZ-{float(labels_embeddings_batch[2]):.2f}-SX-{float(labels_embeddings_batch[3]):.2f}-SY-{float(labels_embeddings_batch[4]):.2f}-SZ-{float(labels_embeddings_batch[5]):.2f}"
         
         self.plotWav(real_data,generated_data,MSE,SSIM,LMDS,title,saveToPath=roomWorkDir+"/"+record_name+".wave.png")
         
         f = open(roomWorkDir+"/MSE.db.txt", "a")
         f.write(record_name+"="+str(MSE)+"\n")
         f.close()
         f = open(roomWorkDir+"/LMDS.db.txt", "a")
         f.write(record_name+"="+str(LMDS)+"\n")
         f.close()
         f = open(roomWorkDir+"/SSIM.db.txt", "a")
         f.write(record_name+"="+str(SSIM)+"\n")
         f.close()
         
     open( workDir+"/.wavesAndSpectrogramsGenerated", 'a').close()   
 
        
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

------------------TF SSIM------------------------------

# Read images (of size 255 x 255) from file.
    im1 = tf.image.decode_image(tf.io.read_file('path/to/im1.png'))
    im2 = tf.image.decode_image(tf.io.read_file('path/to/im2.png'))
    tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
    tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
    # Add an outer batch for each image.
    im1 = tf.expand_dims(im1, axis=0)
    im2 = tf.expand_dims(im2, axis=0)
    # Compute SSIM over tf.uint8 Tensors.
    ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

    # Compute SSIM over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    # ssim1 and ssim2 both have type tf.float32 and are almost equal.



    
---------------------------  Frank Zalkow WAV to SPECTROGRAM ---------------------------------

""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                      samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap,
               interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in
                       ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

plotstft("my_audio_file.wav")


---------------------------------------- LIBROSA WAV to SPECTROGRAM ------------------



X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()



------------------------------------------------------- How to save a Librosa spectrogram plot as a specific sized image?   -------------------------

import numpy as np
import matplotlib.pyplot as plt
import librosa.display

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

hl = 512 # number of samples per time-step in spectrogram
hi = 128 # Height of image
wi = 384 # Width of image

# Loading demo track
y, sr = librosa.load(librosa.ex('trumpet'))
window = y[0:wi*hl]

S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=hi, fmax=8000,
hop_length=hl)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)

plt.savefig("out.png")
plt.show()

-------------------------------------------------------  

mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, 
 n_fft=n_fft)
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
plt.figure(figsize=(8, 7))
librosa.display.specshow(power_to_db, sr=sr, x_axis=’time’, y_axis=’mel’, cmap=’magma’, 
 hop_length=hop_length)
plt.colorbar(label=’dB’)
plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.show()


----------------------------------------------------------

x, sr = librosa.load('audio/00020_2003_person1.wav', sr=None)

window_size = 1024
hop_length = 512 
n_mels = 128
time_steps = 384 

window = np.hanning(window_size)
stft= librosa.core.spectrum.stft(x, n_fft = window_size, hop_length = hop_length, window=window)
out = 2 * np.abs(stft) / np.sum(window)

plt.figure(figsize=(12, 4))
ax = plt.axes()
plt.set_cmap('hot')
librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), y_axis='log', x_axis='time',sr=sr)
plt.savefig('spectrogramA.png', bbox_inches='tight', transparent=True, pad_inches=0.0 )

--------------------------------------------------------
https://github.com/Tony607/tf_audio_signal/blob/master/tf_audio_signal.ipynb
def get_mfccs(audio_file=None ,signals=None, 
sample_rate = 44100, num_mfccs = 13, 
frame_length=1024, frame_step=512, 
fft_length=1024, fmax=8000, fmin=80):
    """Compute the MFCCs for audio file
    
    Keyword Arguments:
        audio_file {str} -- audio wav file path (default: {None})
        signals {tensor} -- input signals as tensor or np.array in float32 type (default: {None})
        sample_rate {int} -- sampling rate (default: {44100})
        num_mfccs {int} -- number of mfccs to keep (default: {13})
        frame_length {int} -- frame length to compute STFT (default: {1024})
        frame_step {int} -- frame step to compute STFT (default: {512})
        fft_length {int} -- FFT length to compute STFT (default: {1024})
        fmax {int} -- Top edge of the highest frequency band (default: {8000})
        fmin {int} -- Lower bound on the frequencies to be included in the mel spectrum (default: {80})
    
    Returns:
        Tensor -- mfccs as tf.Tensor
    """

    
    if signals is None and audio_file is not None:
      audio_binary = tf.read_file(audio_file)
      # tf.contrib.ffmpeg not supported on Windows, refer to issue
      # https://github.com/tensorflow/tensorflow/issues/8271
      waveform = tf.contrib.ffmpeg.decode_audio(audio_binary, 
          file_format='wav', samples_per_second=sample_rate, channel_count=1)
      signals = tf.reshape(waveform, [1, -1])
    
    # Step 1 : signals->stfts
    # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1 = 513.
    stfts = tf.contrib.signal.stft(signals, frame_length=frame_length, frame_step=frame_step,
                                   fft_length=fft_length)
    # Step2 : stfts->magnitude_spectrograms
    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [batch_size, ?, 513].
    magnitude_spectrograms = tf.abs(stfts)

    # Step 3 : magnitude_spectrograms->mel_spectrograms
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    num_mel_bins = 64

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, fmin,
        fmax)

    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    # Step 4 : mel_spectrograms->log_mel_spectrograms
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    # Step 5 : log_mel_spectrograms->mfccs
    # Keep the first `num_mfccs` MFCCs.
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfccs]
    
    return mfccs
'''





         




