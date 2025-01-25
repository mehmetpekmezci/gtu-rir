import numpy as np 
import matplotlib.pyplot as plt 
import csv


WALL_AREA=0
CONFIGS={}
BARS_GLITCH_COUNT={}
BARS_MSE={}
BARS_SSIM={}
BARS_MESHES_MSE={}
object_counts=['53', '29', '18', '9', '0']

with open('data.txt', newline='\n') as f:
    data = csv.reader(f,delimiter=';')
    for row in data:
        if row[4]=="8" or row[4]=="21" :
           continue

        else:   
         WALL_AREA=row[0]
         if not row[1] in CONFIGS:
            CONFIGS[row[1]]={}
            CONFIGS[row[1]]['AREA']=row[2]
            
         #GENRE='FACES='+str(row[3])+' - HEADS='+str(row[4])
         GENRE='FACES='+str(row[3])
         if not GENRE in BARS_MSE:
            
            print(GENRE+'\n')           
            BARS_MSE[GENRE]=[]
            BARS_SSIM[GENRE]=[]
            BARS_GLITCH_COUNT[GENRE]=[]
         
         
         BARS_MSE[GENRE].append(float(row[5]))
         BARS_SSIM[GENRE].append(1-float(row[6]))
         BARS_GLITCH_COUNT[GENRE].append(float(row[7]))
        
        #print(row)

with open('mse.meshes.txt', newline='\n') as f:
    mse_meshes = csv.reader(f,delimiter=';')
    for row in mse_meshes:
        if row[2]=="8" or row[2]=="21" :
           continue

        else:   
         GENRE='FACES='+str(row[1])
         #GENRE='FACES='+str(row[1])+' - HEADS='+str(row[2])
         if not GENRE in BARS_MESHES_MSE:
            BARS_MESHES_MSE[GENRE]=[]
         BARS_MESHES_MSE[GENRE].append(float(row[3]))   
        #print(row)

barWidth = 0.1


plt.clf()

#fig = plt.subplots(figsize =(12, 8)) 
fig, ax = plt.subplots(layout='constrained')

#print(BARS_MSE)
#print("################")
#print(list(BARS_MSE.keys()))
#print(list(BARS_MSE.keys())[0])
#print(BARS_MSE[list(BARS_MSE.keys())[0]])
#print(len(BARS_MSE[list(BARS_MSE.keys())[0]]))

x = np.arange(len(BARS_MSE[list(BARS_MSE.keys())[0]])) 
#print(x)
#X={}
#previous_x=None
#for GENRE in BARS_MSE:
#    if previous_x is None:
#     X[GENRE]=np.arange(len(BARS_MSE[GENRE]))
#    else:
#     X[GENRE]=np.arange(len(BARS_MSE[GENRE])) 
##     X[GENRE]=[x + barWidth for x in previous_x] 
#
#    previous_x=X[GENRE]

#colors=['#FF0000','#FF5050','#FFAAAA','#00FF00','#50FF50','#AAFFAA','#0000FF','#5050FF','#AAAAFF']
#colors=['#000000','#202020','#404040','#606060','#808080','#A0A0A0','#C0C0C0','#E0E0E0','#F0F0F0']
colors=['#000000','#606060','#C0C0C0']

multiplier = 0
for attribute, measurement in BARS_MSE.items():
#    print(attribute)
#    print(measurement)
    offset = barWidth * multiplier
    rects = ax.bar(x + offset, measurement, color =colors[multiplier],width=barWidth, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    #break
    
#for GENRE in BARS_MSE:
#    plt.bar(BARS_MSE[GENRE], X[GENRE], color =colors[i], width = barWidth, edgecolor ='grey', label=GENRE)
#    i=i+1 

plt.xlabel('Number of Objects in Room', fontweight ='bold', fontsize = 12) 
plt.ylabel('MSE of Generated-Real RIR', fontweight ='bold', fontsize = 12) 
#plt.xticks([r + barWidth for r in range(5)], 
#        ['53', '29', '18', '9', '0'])

ax.set_xticks(x + barWidth, object_counts)
#ax.set_xticks(x + barWidth, ['53 (32 m^2)', '18 (13 m^2)', '0 (0 m^2)'])
ax.set_ylim(0, 0.04)
plt.legend()
#plt.show() 
ax.set_title('MSE of Generated RIRs per Room Object Configurations')

plt.savefig('mse.png')


plt.clf()
fig, ax = plt.subplots(layout='constrained')
multiplier = 0
for attribute, measurement in BARS_SSIM.items():
#    print(attribute)
#    print(measurement)
    offset = barWidth * multiplier
    rects = ax.bar(x + offset, measurement, color =colors[multiplier],width=barWidth, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    #break
plt.ylabel('1 - SSIM of Generated-Real RIR', fontweight ='bold', fontsize = 12) 
ax.set_ylim(0, 1)
ax.set_xticks(x + barWidth, object_counts)
plt.legend()
plt.savefig('ssim.png')


plt.clf()
fig, ax = plt.subplots(layout='constrained')
multiplier = 0
for attribute, measurement in BARS_GLITCH_COUNT.items():
#    print(attribute)
#    print(measurement)
    offset = barWidth * multiplier
    rects = ax.bar(x + offset, measurement, color =colors[multiplier],width=barWidth, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    #break
plt.ylabel('Glitch Count of Generated-Real RIR', fontweight ='bold', fontsize = 12) 
ax.set_ylim(0, 150)
ax.set_xticks(x + barWidth,object_counts)
plt.legend()
plt.savefig('glitch_count.png')


plt.clf()
fig, ax = plt.subplots(layout='constrained')
multiplier = 0
for attribute, measurement in BARS_MESHES_MSE.items():
#    print(attribute)
#    print(measurement)
    offset = barWidth * multiplier
    rects = ax.bar(x + offset, measurement, color =colors[multiplier],width=barWidth, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    #break
plt.ylabel('MSE of Generated-Real Mesh', fontweight ='bold', fontsize = 12) 
ax.set_ylim(0,0.00015)
ax.set_xticks(x + barWidth, object_counts)
plt.legend()
plt.savefig('mse_mesh.png')



plt.close()


