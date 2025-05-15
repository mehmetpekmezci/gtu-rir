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
#model = AutoencoderKL.from_pretrained("zelaki/eq-vae")
#model.eval()


#latents = model.encode(images).sample().to(torch.float32)

#print(latents)

#print(model)

print("start")
import numpy as np
from PIL import Image

# Defined an array 3 by 3 with 200 as all the pixel values
obj_array = np.full((100, 800, 6), 250)

print(obj_array)
# Create an image from the array
new_imageB = Image.fromarray(np.uint8(obj_array), mode="RGB")

# Show the image generated using .fromarray()
new_imageB.save('deneme.png')



#import numpy as np
#from PIL import Image
#im=np.asarray(Image.open(r"20220703_130731.jpg"))
#im2 = Image.fromarray(np.uint8(im))
#im2.save('deneme.png')

