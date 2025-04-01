#!/usr/bin/env python3
from PIL import Image,ImageDraw,ImageFont
import sys
def generateReport(reportName,fastrirMSEImagePath,fastrirSSIMImagePath,mesh2irMSEImagePath,mesh2irSSIMImagePath):
          images = [Image.open(x) for x in [fastrirMSEImagePath,fastrirSSIMImagePath,mesh2irMSEImagePath,mesh2irSSIMImagePath]]
          width, height = images[0].size
          total_width = 2*width
          max_height = 2*height+20
          fontsize=40
          new_im = Image.new('RGB', (total_width, max_height))
          ImageDraw.Draw(new_im).rectangle([(0,0),(total_width,max_height)],fill="white")
          new_im.paste(images[0], (10,10))
          new_im.paste(images[1], (width+10,10))
          new_im.paste(images[2], (10,40+height))
          new_im.paste(images[3], (width+20,40+height))

          ImageDraw.Draw(new_im).rectangle([(0,30),(total_width,60)],fill="white")
          ImageDraw.Draw(new_im).rectangle([(0,20+height),(total_width,90+height)],fill="white")
          ImageDraw.Draw(new_im).text((200,10),"FASTRIR-MSE",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))
          ImageDraw.Draw(new_im).text((width+200,10),"FASTRIR-SSIM",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))
          ImageDraw.Draw(new_im).text((200,40+height),"MESH2IR-MSE",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))
          ImageDraw.Draw(new_im).text((width+200,40+height),"MESH2IR-SSIM",fill="black",font=ImageFont.truetype("arial.ttf", fontsize))
          
          new_im.save(reportName+'.png')

if __name__ == '__main__':
 generateReport(sys.argv[1].strip(),sys.argv[2].strip(),sys.argv[3].strip(),sys.argv[4].strip(),sys.argv[5].strip())

         

