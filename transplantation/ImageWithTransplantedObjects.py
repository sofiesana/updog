# imports
import os
from PIL import Image
from .utils import display, log_entry, get_next_id
from .ObjectTransplanter import ObjectTransplanter
import numpy as np
import pickle as pkl
import json
import fiftyone as fo
import copy
import cv2

class ImageWithTransplantedObjects():
  def __init__(self, sample, save_location, dataset_name, filename_appendix=None):
    self.log_file = os.path.join(save_location, dataset_name + "_" + "transplantation_log.json")
    original_image_path = sample.filepath
    self.og_image = Image.open(original_image_path)
    self.modified_image = self.og_image
    self.og_id = sample.id
    self.og_sample = sample
    self.transplanted_image_id = f"{self.og_id}"
    self.dataset_name = dataset_name
    self.transplantations = {}
    self.transplantation_counter = 0
    self.abort = False

    if filename_appendix is not None:
      self.filename_appendix = '_' + filename_appendix
    else:
      self.filename_appendix = ''

    self.save_location = save_location
    self.image_location_folder = self.make_folder('transplanted_images')
    self.sample_location_folder = self.make_folder('transplanted_samples')

    self.modified_sample = None

    self.dataset = self.setup_dataset()

    self.current_object = None

    self.filename = None

  def make_save_paths(self):
    self.filename = f'{self.transplanted_image_id}_{self.current_object.obj_id}{self.filename_appendix}'
    self.image_save_location = os.path.join(self.image_location_folder, f'transplanted_image_{self.filename}.jpg')
    self.modified_sample_path = os.path.join(self.sample_location_folder, f"transplanted_{self.filename}.json")

    # Check if the image save location already exists
    if os.path.exists(self.image_save_location):
        print(f"The image save location already exists. ({self.image_save_location})")
        self.abort = True

    # Check if the modified sample path already exists
    if os.path.exists(self.modified_sample_path):
        print(f"The modified sample path already exists. ({self.modified_sample_path})")
        self.abort = True
    

  def make_folder(self, folder_name):
    location_folder = os.path.join(self.save_location, folder_name)
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    return location_folder

  def setup_dataset(self):
    if self.dataset_name not in fo.list_datasets():
      dataset = fo.Dataset(name=self.dataset_name)
    else:
      dataset = fo.load_dataset(self.dataset_name)
    dataset.persistent = True
    return dataset
  
  def setup_modified_sample(self):
    self.modified_sample["new_id"] = self.transplanted_image_id
    # self.modified_sample["ground_truth"] = self.og_sample["ground_truth"]
    self.modified_sample["ground_truth"] = copy.deepcopy(self.og_sample["ground_truth"])
    self.modified_sample.id = self.transplanted_image_id
    self.modified_sample.metadata = self.og_sample.metadata

  def add_transplanted_object(self, obj, location):
    self.current_object = obj
    self.make_save_paths()
    if self.abort:
      print("Aborting transplantation.")
      return
    self.modified_sample = fo.Sample(filepath=self.image_save_location)
    self.setup_modified_sample()
    self.dataset.add_sample(self.modified_sample)
    self.transplantation_counter += 1
    self.transplantations[self.transplantation_counter] = {"object_id": obj.id, "obj_file_location": obj.file_location, "location": location}
    transplanter = ObjectTransplanter()
    transplanter.transplant_object(self.modified_image, obj, location)
    self.modified_image = transplanter.get_transplanted_image()
    self.update_modified_sample_with_transplant(obj, location)  

  def save_transplanted_image(self):
    if self.abort:
      print("transplantated image not saved as transplantation was aborted")
      return
    self.log_modified_image()
    self.save_image()

  def log_modified_image(self):
    entry = {
       f"{self.filename}": {
        "original_image_id": self.og_id,
        "transplanted_object_id": self.current_object.id,
        "image_file_location": self.image_save_location,
        "sample_save_location": self.modified_sample_path,
        "transplantations": self.transplantations
       }
    }
    log_entry(self.log_file, entry, self.og_id)

  def save_image(self):
    image = Image.fromarray(self.modified_image)
    image.save(self.image_save_location)
    
    self.modified_sample["original_image_id"] = self.og_id
    self.modified_sample["origina_image_path"] = self.og_sample.filepath
    self.modified_sample.save()

    json_sample = self.modified_sample.to_dict(include_private=True)
    with open(self.modified_sample_path, 'w') as f:
      json.dump(json_sample, f)

  def display_transplanted_image(self):
    image = Image.fromarray(self.modified_image)
    display(image)
  
  def update_modified_sample_with_transplant(self, obj, location):
        # print("updating modified sample")
        # print("OG Image Size: ", self.og_image.size)
        # print("OG Img dimensions: ", obj.og_img_dimensions)
        # print("Box: ", obj.box) 
        # print("Location: ", location)

        new_width = obj.box[2] * (obj.og_img_dimensions[1]/self.og_image.size[0])
        new_height = obj.box[3] * (obj.og_img_dimensions[0]/self.og_image.size[1])
        new_bbox = [location[0]/self.og_image.size[0], location[1]/self.og_image.size[1], new_width, new_height]
        new_segmentation = obj.mask
        # print(new_bbox, new_segmentation)
        new_detection = {
            "label": obj.class_label,
            "bounding_box": new_bbox,
            "mask": new_segmentation
        }
        new_detection = fo.Detection(
                    label=obj.class_label,
                    bounding_box=new_bbox,
                    mask=new_segmentation
                )
        self.modified_sample["ground_truth"].detections.append(new_detection)

        # print(self.modified_sample["ground_truth"].detections)
        # self.modified_sample.save()
        
