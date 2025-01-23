# imports
import os
from PIL import Image
from utils import display
from ObjectTransplanter import ObjectTransplanter
import numpy as np
from utils import log_transplantation
import pickle as pkl

class ImageWithTransplantedObjects():
  def __init__(self, sample, save_location, log_file="transplantation_log.json"):
    self.log_file = log_file
    original_image_path = sample.filepath
    if save_location is None:
      self.save_location = ''
    else:
      self.save_location = save_location
    self.og_image = Image.open(original_image_path)
    self.modified_image = self.og_image
    self.og_id = sample.id
    self.transplant_dict = {}
    self.og_sample = sample
    self.modified_sample = sample.copy()

    location_folder = os.path.join(self.save_location, f'transplanted_images')
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    self.image_save_location = os.path.join(location_folder, f'transplanted_image_{self.og_id}.jpg')

    location_folder = os.path.join(self.save_location, 'transplanted_samples', )
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    self.modified_sample_path = os.path.join(location_folder, f"transplanted_{self.og_id}.pkl")

  def add_transplanted_object(self, obj, location):
    self.transplant_dict[location] = (obj)
    transplanter = ObjectTransplanter()
    transplanter.transplant_object(self.modified_image, obj, location)
    self.modified_image = transplanter.get_transplanted_image()
    self.update_modified_sample_with_transplant(obj, location)
    log_transplantation(self.log_file, self.og_sample.id, obj.class_label, obj.id, location)

  def save_transplanted_image(self):
    image = Image.fromarray(self.modified_image)
    image.save(self.image_save_location)

    with open(self.modified_sample_path, 'wb') as f:
        pkl.dump(self.modified_sample, f)

  def display_transplanted_image(self):
    image = Image.fromarray(self.modified_image)
    display(image)
  
  def update_modified_sample_with_transplant(self, obj, location):
        print("updating modified sample")
        # note, this does not actually update the sample image, just the bounding boxes...

        # Calculate the new bounding box and segmentation map
        new_bbox = [location[0], location[1], obj.box[2], obj.box[3]]
        new_segmentation = obj.mask

        # Add the new detection to the sample
        new_detection = {
            "label": obj.class_label,
            "bounding_box": new_bbox,
            "mask": new_segmentation
        }
        self.modified_sample["ground_truth"].detections.append(new_detection)
        # need to see if this will work:
        self.modified_image["filepath"] = self.image_save_location
        
        print("original image:")
        print(self.og_sample)
        print("modified image:")
        print(self.modified_sample)