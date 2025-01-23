import os
import pickle as pkl
from PIL import Image
from utils import display, log_entry
import json
import numpy as np

class ExtractedObject():
  def __init__(self, log_file_path):
    self.log_file_path = log_file_path
    self.mask = None
    self.mask_with_pixels = None
    self.id = None
    self.class_label = None
    self.box = None
    self.box_in_pixels = None
    self.save_location = None
    self.is_setup = False
    self.file_location = None

  def setup(self, mask, mask_with_pixels, id, class_label, box, box_in_pixels, save_location):
    if not self.is_setup:
      self.mask = mask
      self.mask_with_pixels = mask_with_pixels
      self.id = id
      self.class_label = class_label
      self.box = box
      self.box_in_pixels = box_in_pixels
      self.save_location = save_location
      location_folder = os.path.join(self.save_location, f'extracted_objects')
      if not os.path.exists(location_folder):
        os.makedirs(location_folder)
      self.file_location = os.path.join(location_folder, f'{self.class_label}_{self.id}.pkl')
      self.obj_id = f'{self.class_label}_{self.id}'
      self.is_setup = True
    else:
      print("Setup failed: Object already setup")
  
  def log_object(self):
    entry = {
      f"{self.obj_id}": {
        "class_label": self.class_label,
        "og_image_id": self.id,
        "file_location": self.file_location
      }
    }

    log_entry(self.log_file_path, entry, id=self.obj_id)
    

  def save_object(self):
    with open(self.file_location, 'wb') as f:
      pkl.dump(self, f)
    self.log_object()

  def load_object(self, location):
    with open(location, 'rb') as f:
      obj = pkl.load(f)
    self.__dict__ = obj.__dict__.copy()
    self.is_setup = True

  def display_extracted_object(self):
    image = Image.fromarray(self.mask_with_pixels)
    # display image in window
    display(image)

  def save_mask_with_pixels_as_jpg(self):
    location_folder = os.path.join(self.save_location, f'pixel_masks')
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    location = os.path.join(location_folder, f'pixel_mask_{self.class_label}_{self.id}.jpg')
    image = Image.fromarray(self.mask_with_pixels)
    image.save(location)

  def save_mask(self):
    location_folder = os.path.join(self.save_location, f'masks')
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    location = os.path.join(location_folder, f'mask_{self.class_label}_{self.id}.jpg')
    image = Image.fromarray(self.mask)
    image.save(location)


  def scale_object(self, new_width, new_height):
    self.mask = Image.fromarray(self.mask)
    self.mask = self.mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
    self.mask = np.array(self.mask)

    self.mask_with_pixels = Image.fromarray(self.mask_with_pixels)
    self.mask_with_pixels = self.mask_with_pixels.resize((new_width, new_height), Image.Resampling.LANCZOS)
    self.mask_with_pixels = np.array(self.mask_with_pixels)

    print(f"Object scaled to: {new_width}x{new_height}")