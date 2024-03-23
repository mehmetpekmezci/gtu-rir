#!/usr/bin/env python3
from PIL import Image,ImageDraw,ImageFont
import sys



#FASTRIR-MSE          --  FASTRIR-SSIM        --   FASTRIR-SSIM_PLUS_MSE
#FASTRIR-MFCC-MSE     --  FASTRIR-MFCC-SSIM   --   FASTRIR-MFCC-SSIM-AND-MSE-WEIGHTED
#
#MESH2IR-MSE          --  MESH2IR-SSIM        --   MESH2IR-SSIM_PLUS_MSE
#MESH2IR-MFCC-MSE     --  MESH2IR-MFCC-SSIM   --   MESH2IR-MFCC-SSIM-AND-MSE-WEIGHTED
#


SUMMARY_DIR=sys.argv[1].strip()
SUB_TYPE=sys.argv[2].strip()

FASTRIR_MSE=Image.open(sys.argv[3].strip())
FASTRIR_SSIM=Image.open(sys.argv[4].strip())
FASTRIR_SSIM_PLUS_MSE=Image.open(sys.argv[5].strip())
FASTRIR_MFCC_MSE=Image.open(sys.argv[6].strip())
FASTRIR_MFCC_SSIM=Image.open(sys.argv[7].strip())
FASTRIR_MFCC_SSIM_AND_MSE_WEIGHTED=Image.open(sys.argv[8].strip())
#FASTRIR_EDGE_WEIGHT_MSE=Image.open(sys.argv[9].strip())
#FASTRIR_EDGE_WEIGHT_SSIM=Image.open(sys.argv[10].strip())
#FASTRIR_EDGE_WEIGHT_SSIM_PLUS_MSE=Image.open(sys.argv[11].strip())

MESH2IR_MSE=Image.open(sys.argv[9].strip())
MESH2IR_SSIM=Image.open(sys.argv[10].strip())
MESH2IR_SSIM_PLUS_MSE=Image.open(sys.argv[11].strip())
MESH2IR_MFCC_MSE=Image.open(sys.argv[12].strip())
MESH2IR_MFCC_SSIM=Image.open(sys.argv[13].strip())
MESH2IR_MFCC_SSIM_AND_MSE_WEIGHTED=Image.open(sys.argv[14].strip())
#MESH2IR_EDGE_WEIGHT_MSE=Image.open(sys.argv[18].strip())
#MESH2IR_EDGE_WEIGHT_SSIM=Image.open(sys.argv[19].strip())
#MESH2IR_EDGE_WEIGHT_SSIM_PLUS_MSE=Image.open(sys.argv[20].strip())


images=[
         [FASTRIR_MSE,FASTRIR_SSIM,FASTRIR_SSIM_PLUS_MSE],
#         [FASTRIR_MFCC_MSE,FASTRIR_MFCC_SSIM,FASTRIR_MFCC_SSIM_AND_MSE_WEIGHTED],
#         [None,None,None],
         [MESH2IR_MSE,MESH2IR_SSIM,MESH2IR_SSIM_PLUS_MSE],
#         [MESH2IR_MFCC_MSE,MESH2IR_MFCC_SSIM,MESH2IR_MFCC_SSIM_AND_MSE_WEIGHTED],
#         [MESH2IR_EDGE_WEIGHT_MSE,MESH2IR_EDGE_WEIGHT_SSIM,MESH2IR_EDGE_WEIGHT_SSIM_PLUS_MSE]
      ]

titles=[
         ["(a.1) FASTRIR-MSE","(a.2) FASTRIR-SSIM","(a.3) FASTRIR-SSIM_PLUS_MSE"],
#         ["(b.1) FASTRIR-MFCC-MSE","(b.2) FASTRIR-MFCC-SSIM","(b.3) FASTRIR-MFCC-SSIM_PLUS_MSE "],
#         [None,None,None],
#         ["(c.1) MESH2IR-MSE","(c.2) MESH2IR-SSIM","(c.3) MESH2IR-SSIM_PLUS_MSE"],
         ["(b.1) MESH2IR-MSE","(b.2) MESH2IR-SSIM","(b.3) MESH2IR-SSIM_PLUS_MSE"],
#         ["(d.1) MESH2IR-MFCC-MSE","(d.2) MESH2IR-MFCC-SSIM","(d.3) MESH2IR-MFCC-SSIM_PLUS_MSE "],
#         ["(e.1) MESH2IR-EDGE_WEIGHT-MSE","(e.2) MESH2IR-EDGE_WEIGHT-SSIM","(e.3) MESH2IR-EDGE_WEIGHT-SSIM_PLUS_MSE"]
      ]


width, height = FASTRIR_MSE.size
total_width = 3*width+60
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
         

