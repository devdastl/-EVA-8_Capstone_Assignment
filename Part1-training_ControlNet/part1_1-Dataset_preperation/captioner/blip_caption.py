from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from captioner.models.blip import blip_decoder


class BlipCaption():
    def __init__(self, device = 'cuda', 
                 model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth',
                 image_size = 384):
        self.device = device
        self.model_url = model_url
        self.model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base').eval()
        self.model = self.model.to(self.device)
        self.image_size = image_size

    def __image_transform(self, image):
        raw_image = Image.open(image).convert('RGB')
        w,h = raw_image.size
        transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(self.device)   
        return image
    
    def generate_caption(self, image_path):
        image = self.__image_transform(image_path)
        with torch.no_grad():
            caption = self.model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            return caption[0]