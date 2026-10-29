from constants import batch_size, custom_format, device
from PIL import Image,ImageChops
from PIL.ImageTransform import AffineTransform
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import random

def make_image_noisy(image,noise_level=40):
    """This function adds some noise to the image and rotate it randomly and translate it randomly
    Args:
        image (PIL.Image): The image to add noise to
        noise_level (float): The level of noise to add
    Returns:
        torch.Tensor: The noisy image
    """
    width, height = image.size
    angle = random.uniform(-180, 180)
    x_shift = random.uniform(-5, 5)
    y_shift = random.uniform(-5, 5)
    noise_texture = Image.effect_noise(image.size,noise_level)
    image = image.transform(image.size, AffineTransform([1, 0, x_shift, 0, 1, y_shift]))
    image = image.rotate(angle,resample=Image.Resampling.BICUBIC)
    image = image.resize((width, height),resample=Image.Resampling.BICUBIC) 
    image = ImageChops.multiply(image,noise_texture)
    return image

def load_data(files,noisy=False,shape=(28*28,)):
  loaded_data = []
  for file in tqdm(files,desc="Loading data",bar_format=custom_format):
    with Image.open(file) as img:
      label = file.stem.split("_")[1]
      label_tensor = torch.tensor([1 if i == int(label) else 0 for i in range(10)])
      prop = random.random()
      if prop < 0.3 and noisy:
        img_noisy = make_image_noisy(img)
        loaded_data.append({'image':torch.tensor(np.array(img_noisy).reshape(shape)),'label':label_tensor.float()})
      loaded_data.append({'image':torch.tensor(np.array(img).reshape(shape)),'label':label_tensor.float()})
  return loaded_data


def get_batches(data , normalize = True):
  suffle_idx = torch.randperm(len(data))
  # suffle_idx = range(batch_size)
  suffle_batch = [suffle_idx[i:i+batch_size] for i in range(0,len(suffle_idx),batch_size)]
  batch_dict= []
  image_list = []
  label_list = []
  
  def normalize_fn(item):
    if normalize:
      return item.float()/255
    
    return item.float()
  for bth in suffle_batch:
    if len(bth) < batch_size:
      continue
    items = [data[i] for i in bth]
    image_list.append(torch.stack([normalize_fn(item['image']) for item in items]))
    label_list.append(torch.stack([item['label'] for item in items]))
    batch_dict.append(items)
    
  image_tensor = torch.stack(image_list).to(device=device)
  label_tensor = torch.stack(label_list).to(device=device)
  return image_tensor,label_tensor,batch_dict


def get_date_suffix():
  return datetime.now().strftime("%Y%m%d_%H%M%S")