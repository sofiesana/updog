import os
import pickle as pkl
from PIL import Image
from utils import display

class ExtractedObject():
  def __init__(self):
    self.mask = None
    self.mask_with_pixels = None
    self.id = None
    self.class_label = None
    self.box = None
    self.box_in_pixels = None
    self.save_location = None
    self.is_setup = False

  def setup(self, mask, mask_with_pixels, id, class_label, box, box_in_pixels, save_location):
    if not self.is_setup:
      self.mask = mask
      self.mask_with_pixels = mask_with_pixels
      self.id = id
      self.class_label = class_label
      self.box = box
      self.box_in_pixels = box_in_pixels
      self.save_location = save_location
      self.is_setup = True
    else:
      print("Setup failed: Object already setup")

  def save_object(self):
    location_folder = os.path.join(self.save_location, f'extracted_objects')
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    location = os.path.join(location_folder, f'{self.class_label}_{self.id}.pkl')
    with open(location, 'wb') as f:
      pkl.dump(self, f)

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
