import requests
from huggingface_hub import configure_http_backend


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

#from diffusers.models import AutoencoderKL
#from diffusers import StableDiffusionPipeline

#model = "stabilityai/your-stable-diffusion-model"
#vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
#pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)


from diffusers.models import AutoencoderKL
###vae = AutoencoderKL.from_pretrained("gabehubner/vae-256px-32z")
model = AutoencoderKL.from_pretrained("zelaki/eq-vae")
model.eval()
model.cuda()

#latents = model.encode(images).sample().to(torch.float32)

#print(latents)

#print(model)

import numpy as np
from PIL import Image
import torch

# Defined an array 3 by 3 with 200 as all the pixel values
obj_array = np.full((256,256), 250).reshape(256,256,1).repeat(3,axis=2)
print(obj_array.shape)
new_imageB = Image.fromarray(np.uint8(obj_array), mode="RGB")
new_imageB.save('deneme1.png')

imtensor=torch.tensor(np.array(new_imageB).transpose(2, 0, 1), dtype=torch.float32).reshape(1,3,256,256).cuda()

#obj_array = np.full((360,180), 250)
#obj_array = np.full((128,128), 250)

#obj_array=obj_array.reshape(1,360,180).repeat(3,axis=0)
#obj_array=obj_array.reshape(1,128,128).repeat(3,axis=0)

#print(obj_array.shape)

#print(obj_array)
# Create an image from the array

#imtensor=torch.from_numpy(np.array([new_imgaeB]))
#imtensor=torch.from_numpy(np.array([obj_array])).cuda()
#latents=model.encode(imtensor.to(torch.float32)).sample().to(torch.float32)
#latents=model.encode(imtensor.to(torch.float32))[0].sample()
print(imtensor.shape)
#latents=model.encode(imtensor.to(torch.float32)).latent_dist.sample()

with torch.no_grad():
  latents=model.encode(imtensor).latent_dist.sample()
  latents_flat=latents.flatten()
  print(latents_flat.shape)
  decoded_images_tensor = model.decode(latents).sample
  decoded_images = torch.clamp(127.5 * decoded_images_tensor + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
  d=decoded_images[0]
  print(d.shape)
  new_imageB = Image.fromarray(np.uint8(d), mode="RGB")
  new_imageB.save('deneme2.png')



#import numpy as np
#from PIL import Image
#im=np.asarray(Image.open(r"20220703_130731.jpg"))
#im2 = Image.fromarray(np.uint8(im))
#im2.save('deneme.png')

