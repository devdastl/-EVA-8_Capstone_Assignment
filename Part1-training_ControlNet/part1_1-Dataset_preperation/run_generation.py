import os
import sys
import json
from pathlib import Path
import shutil
import cv2

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector

#custom class to encapsulate BLIP inference for caption generation
from captioner.blip_caption import BlipCaption


def get_images(path, image_path='images'):

    path = os.path.abspath(path)
    image_path = os.path.join(path, image_path)
    images = os.listdir(image_path)
    images = [os.path.join(image_path, image) for image in images]
    return images


def generate_dir(path, data_folder='dataset'):
    path = os.path.abspath(path)
    data_path = os.path.join(path, data_folder)

    source = (os.path.join(data_path, 'source'))
    target = (os.path.join(data_path, 'target'))
    prompt = (os.path.join(data_path, 'prompt.json'))

    Path(source).mkdir(parents=True, exist_ok=True)
    Path(target).mkdir(parents=True, exist_ok=True)

    return source, target, prompt


def get_canny(image, source_image_path):
    res = 512
    l = 100
    h = 200

    try:
        image = cv2.imread(image)
        image = resize_image(HWC3(image), res)
        model_canny = CannyDetector()
        result = model_canny(image, l, h)

    except:
        raise Exception(f"cannot convert {image} into edge through canny detector: check if file exists")

    cv2.imwrite(source_image_path, result)


def prompt_json(caption, source, target, json_file):
    json_dict = {'source':str(source), 'target':str(target), 'prompt':str(caption)}

    with open(json_file, 'a') as f:
        json.dump(json_dict, f)
        f.write('\n')
    f.close()

path = './'
list_images = get_images(path, image_path='Images')
source, target, json_file = generate_dir(path)

#generate instance, load model
caption_generator = BlipCaption()

for image in list_images:
    print(f"processing image {image}")
    #set source image target image and copy
    target_image = os.path.join(target, os.path.basename(image))
    source_image = os.path.join(source, os.path.basename(image))
    shutil.copyfile(image, target_image)

    #generate canny images
    get_canny(target_image, source_image_path=source_image)

    #generate caption from target image
    caption = caption_generator.generate_caption(target_image)

    #write to prompt.json
    prompt_json(caption, source_image, target_image, json_file)


  