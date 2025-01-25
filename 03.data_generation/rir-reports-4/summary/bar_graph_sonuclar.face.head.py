import numpy as np 
import matplotlib.pyplot as plt 
import csv

GTURIR_LABELS=[]
GTURIR_VALUES_MSE=[]
GTURIR_VALUES_SSIM=[]
GTURIR_VALUES_GLITCH_COUNT=[]

BUT_LABELS=[]
BUT_BARS_GLITCH_COUNT=[]
BUT_BARS_MSE=[]
BUT_BARS_SSIM=[]


with open('data_face_head.txt', newline='\n') as f:
    data = csv.reader(f,delimiter=';')
    for row in data:
       DATASET=row[0]
       GENRE='FACES='+str(row[1])+' - HEADS='+str(row[2])
       MSE=float(row[3])
       ONE_MINUS_SSIM=float(row[4])
       GLITCH_COUNT=float(row[5])
              
       if DATASET=="GTU-RIR":
          GTURIR_LABELS.append(GENRE)
          GTURIR_VALUES_MSE.append(MSE)
          GTURIR_VALUES_SSIM.append(ONE_MINUS_SSIM)
          GTURIR_VALUES_GLITCH_COUNT.append(GLITCH_COUNT)
       else:  
          BUT_LABELS.append(GENRE)
          BUT_BARS_MSE.append(MSE)
          BUT_BARS_SSIM.append(ONE_MINUS_SSIM)    
          BUT_BARS_GLITCH_COUNT.append(GLITCH_COUNT)
          

        
        #print(row)




fig, ax = plt.subplots()


bar_colors = ['#FF0000','#FF5050','#FF9090','#00FF00','#50FF50','#90FF90','#0000FF','#5050FF','#9090FF']

ax.bar(GTURIR_LABELS, GTURIR_VALUES_MSE, label=GTURIR_LABELS, color=bar_colors)
ax.set_ylabel('MSE of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('GTU-RIR MSE of Generated RIRs per Mesh Face and Attention Head')
ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.gturir.mse.png')


plt.clf()
fig, ax = plt.subplots()
ax.bar(GTURIR_LABELS, GTURIR_VALUES_SSIM, label=GTURIR_LABELS, color=bar_colors)
ax.set_ylabel('1-SSIM of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('GTU-RIR 1-SSIM of Generated RIRs per Mesh Face and Attention Head')
ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.gturir.ssim.png')


plt.clf()
fig, ax = plt.subplots()
ax.bar(GTURIR_LABELS, GTURIR_VALUES_GLITCH_COUNT, label=GTURIR_LABELS, color=bar_colors)
ax.set_ylabel('Glitch Count of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('GTU-RIR Glitch Count of Generated RIRs per Mesh Face and Attention Head')
ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.gturir.glitch_count.png')


ax.bar(BUT_LABELS, BUT_BARS_MSE, label=GTURIR_LABELS, color=bar_colors)
ax.set_ylabel('MSE of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('BUT ReverbDB MSE of Generated RIRs per Mesh Face and Attention Head')
ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.butreverbdb.mse.png')


plt.clf()
fig, ax = plt.subplots()
ax.bar(BUT_LABELS, BUT_BARS_SSIM, label=GTURIR_LABELS, color=bar_colors)
ax.set_ylabel('1-SSIM of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('BUT ReverbDB 1-SSIM of Generated RIRs per Mesh Face and Attention Head')
ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.butreverbdb.ssim.png')


plt.clf()
fig, ax = plt.subplots()
ax.bar(BUT_LABELS, BUT_BARS_GLITCH_COUNT, label=GTURIR_LABELS, color=bar_colors)
ax.set_ylabel('Glitch Count of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('BUT ReverbDB Glitch Count of Generated RIRs per Mesh Face and Attention Head')
ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.butreverbdb.glitch_count.png')


plt.close()


