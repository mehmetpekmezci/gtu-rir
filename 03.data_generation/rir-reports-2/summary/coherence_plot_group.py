#!/usr/bin/env python3
from PIL import Image,ImageDraw,ImageFont
import sys



SUMMARY_DIR=sys.argv[1].strip()
SUB_TYPE=sys.argv[2].strip()

MESHTAE_GTURIR=Image.open(sys.argv[3].strip())
MESHTAE_BUT=Image.open(sys.argv[4].strip())
MESH2IR_GTURIR=Image.open(sys.argv[5].strip())
MESH2IR_BUT=Image.open(sys.argv[6].strip())



images=[
         [MESHTAE_GTURIR,MESHTAE_BUT],
         [MESH2IR_GTURIR,MESH2IR_BUT],
      ]

titles=[
         ["(a.1) MESH-TAE / GTU-RIR ","(a.2) MESH-TAE / BUT ReverbDB"],
         ["(b.1) MESH2IR / GTU-RIR ","(b.2) MESH2IR / BUT RererbDB"],
      ]


width, height = MESHTAE_GTURIR.size
total_width = 2*width+60
max_height = 2*height+240
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

          
new_im.save(SUMMARY_DIR+'/coherence_plot_group.'+SUB_TYPE+'.png')
         

