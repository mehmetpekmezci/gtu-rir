import numpy as np 
import matplotlib.pyplot as plt 
import csv

DATA={}
DATA["GTURIR"]={}
DATA["GTURIR"]["LABELS"]=[]
DATA["GTURIR"]["NODE"]=[]
DATA["GTURIR"]["HEAD"]=[]
DATA["GTURIR"]["MSE"]=[]
DATA["GTURIR"]["SSIM"]=[]
DATA["GTURIR"]["GLITCH"]=[]
DATA["GTURIR"]["MESHMSE"]=[]

DATA["BUTReverbDB"]={}
DATA["BUTReverbDB"]["LABELS"]=[]
DATA["BUTReverbDB"]["NODE"]=[]
DATA["BUTReverbDB"]["HEAD"]=[]
DATA["BUTReverbDB"]["MSE"]=[]
DATA["BUTReverbDB"]["SSIM"]=[]
DATA["BUTReverbDB"]["GLITCH"]=[]
DATA["BUTReverbDB"]["MESHMSE"]=[]



node_list=[]
head_list=[]

file_data_=[]

with open('summary_table.csv', newline='\n') as f:
    file_data = csv.reader(f,delimiter=';')
    for row in file_data:
       if row[0] == "DATASET" :
           continue
           # this is the header row :)
       file_data_.append(row)    
       DATASET=row[0]
       NODE=row[1]
       HEAD=row[2]
       node_list.append(NODE)
       head_list.append(HEAD)

sorted_node_list=sorted(map(int, set(node_list)))    
sorted_head_list=sorted(map(int, set(head_list)))    



for node in sorted_node_list:
   for head in sorted_head_list:
     for row in file_data_:
       if str(row[1]) == str(node) and str(row[2]) == str(head):   
        DATASET=row[0]
        NODE=row[1]
        HEAD=row[2]
        GENRE=str(NODE)+'-'+str(HEAD)
        MSE=float(row[3])
        ONE_MINUS_SSIM=1-float(row[4])
        GLITCH_COUNT=float(row[5])
        MESHMSE=float(row[6])       
        DATA[DATASET]["LABELS"].append(GENRE)
        DATA[DATASET]["NODE"].append(NODE)
        DATA[DATASET]["HEAD"].append(HEAD)
        DATA[DATASET]["MSE"].append(MSE)
        DATA[DATASET]["SSIM"].append(ONE_MINUS_SSIM)
        DATA[DATASET]["GLITCH"].append(GLITCH_COUNT)
        DATA[DATASET]["MESHMSE"].append(MESHMSE)
       




bar_colors_base = ['#FF0000','#0000FF','#00FF00','#FF5050','#5050FF','#50FF50','#FFA0A0','#A0A0FF','#A0FFA0']
bar_colors=[]
for i in range(len(sorted_node_list)):
   for head in sorted_head_list:
     if str(sorted_node_list[i])+"-"+str(head) in DATA["GTURIR"]["LABELS"]:
        bar_colors.append(bar_colors_base[i])
  
  
plt.clf()   
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["GTURIR"]["LABELS"], DATA["GTURIR"]["MSE"], label=DATA["GTURIR"]["LABELS"], color=bar_colors)
ax.set_ylabel('MSE of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('GTU-RIR MSE of Generated RIRs per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.gturir.mse.png')


plt.clf()
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["GTURIR"]["LABELS"], DATA["GTURIR"]["SSIM"], label=DATA["GTURIR"]["LABELS"], color=bar_colors)
ax.set_ylabel('1-SSIM of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('GTU-RIR 1-SSIM of Generated RIRs per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.gturir.ssim.png')


plt.clf()
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["GTURIR"]["LABELS"], DATA["GTURIR"]["GLITCH"], label=DATA["GTURIR"]["LABELS"], color=bar_colors)
ax.set_ylabel('Glitch Count of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('GTU-RIR Glitch Count of Generated RIRs per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.gturir.glitch_count.png')

plt.clf()
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["GTURIR"]["LABELS"], DATA["GTURIR"]["MESHMSE"], label=DATA["GTURIR"]["LABELS"], color=bar_colors)
ax.set_ylabel('MSE of MESH ')
ax.set_xlabel('')
ax.set_title('GTU-RIR MESH MSE per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.gturir.meshmse.png')

plt.clf()
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["BUTReverbDB"]["LABELS"], DATA["BUTReverbDB"]["MSE"], label=DATA["GTURIR"]["LABELS"], color=bar_colors)
ax.set_ylabel('MSE of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('BUT ReverbDB MSE of Generated RIRs per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.butreverbdb.mse.png')


plt.clf()
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["BUTReverbDB"]["LABELS"], DATA["BUTReverbDB"]["SSIM"], label=DATA["GTURIR"]["LABELS"], color=bar_colors)
ax.set_ylabel('1-SSIM of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('BUT ReverbDB 1-SSIM of Generated RIRs per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.butreverbdb.ssim.png')


plt.clf()
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["BUTReverbDB"]["LABELS"], DATA["BUTReverbDB"]["GLITCH"], label=DATA["GTURIR"]["LABELS"], color=bar_colors)
ax.set_ylabel('Glitch Count of Generated RIRs ')
ax.set_xlabel('')
ax.set_title('BUT ReverbDB Glitch Count of Generated RIRs per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.butreverbdb.glitch_count.png')

plt.clf()
fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(DATA["BUTReverbDB"]["LABELS"], DATA["BUTReverbDB"]["MESHMSE"], label=DATA["BUTReverbDB"]["LABELS"], color=bar_colors)
ax.set_ylabel('MSE of MESH ')
ax.set_xlabel('')
ax.set_title('BUT ReverbDB MESH MSE per Mesh Face and Attention Head')
#ax.legend(title='Number of Mesh Faces and Attention Heads')
plt.savefig('face.head.butreverbdb.meshmse.png')

plt.close()


