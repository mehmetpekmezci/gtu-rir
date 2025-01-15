#!/usr/bin/env python3
from PIL import Image,ImageDraw,ImageFont
import sys



SUMMARY_DIR=sys.argv[1].strip()
SUB_TYPE=sys.argv[2].strip()
DATASET=sys.argv[3].strip()

MESHTAE_REAL=Image.open(sys.argv[4].strip())
MESH2IR_REAL=Image.open(sys.argv[5].strip())
MESH2IR_MESHTAE=Image.open(sys.argv[6].strip())



images=[
         [MESHTAE_REAL,MESH2IR_REAL,MESH2IR_MESHTAE],
      ]

titles=[
         ["(a) MESH-TAE / REAL ","(b) MESH2IR / REAL " ,  "(c) MESH-TAE / MESH2IR " ],
      ]


width, height = MESHTAE_REAL.size
total_width = 3*width+90
#max_height = 2*height+240
max_height = height+120
fontsize=20
new_im = Image.new('RGB', (total_width, max_height))
ImageDraw.Draw(new_im).rectangle([(0,0),(total_width,max_height)],fill="white")

for i in range(len(images)) :
    for j in range(len(images[0])):
      if images[i][j] is not None :
        new_im.paste(images[i][j], (10+j*width,10+i*(30+height)))

for i in range(len(images)) :
    for j in range(len(images[0])):
      if images[i][j] is not None :
        ImageDraw.Draw(new_im).text((200+j*width,10+(i+1)*(30+height)),titles[i][j],fill="black",font=ImageFont.truetype("FreeSerifBold.ttf", fontsize))

          
new_im.save(SUMMARY_DIR+'/coherence_plot_group.'+DATASET+'.'+SUB_TYPE+'.png')
         

