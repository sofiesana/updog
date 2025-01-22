# imports
import os
from PIL import Image
from utils import display
from ObjectTransplanter import ObjectTransplanter

class ImageWithTransplantedObjects():
  def __init__(self, image, og_id, save_location):
    if save_location is None:
      self.save_location = ''
    else:
      self.save_location = save_location
    self.og_image = image
    self.modified_image = self.og_image
    self.og_id = og_id
    self.transplant_dict = {}

  def add_transplanted_object(self, obj, location):
    self.transplant_dict[location] = (obj)
    transplanter = ObjectTransplanter()
    transplanter.transplant_object(self.modified_image, obj, location)
    self.modified_image = transplanter.get_transplanted_image()

  def save_transplanted_image(self):
    location_folder = os.path.join(self.save_location, f'transplanted_images')
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    location = os.path.join(location_folder, f'transplanted_image_{self.og_id}.jpg')
    image = Image.fromarray(self.modified_image)
    image.save(location)

  def display_transplanted_image(self):
    image = Image.fromarray(self.modified_image)
    display(image)