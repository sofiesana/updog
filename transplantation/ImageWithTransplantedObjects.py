# imports
import os
from PIL import Image
from utils import display
from ObjectTransplanter import ObjectTransplanter
import numpy as np
from utils import log_entry, get_next_id
import pickle as pkl
import json
import fiftyone as fo

class ImageWithTransplantedObjects():
  def __init__(self, sample, save_location, dataset_name):
    self.log_file = dataset_name + "_" + "transplantation_log.json"
    original_image_path = sample.filepath
    if save_location is None:
      self.save_location = ''
    else:
      self.save_location = os.path.join(save_location, dataset_name)
    self.og_image = Image.open(original_image_path)
    self.modified_image = self.og_image
    self.og_id = sample.id
    self.og_sample = sample
    self.transplanted_image_id = f"{self.og_id}_{get_next_id('transplantation_ids.json')}"
    self.dataset_name = dataset_name
    self.transplantations = {}
    self.transplantation_counter = 0

    location_folder = os.path.join(self.save_location, f'transplanted_images')
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    self.image_save_location = os.path.join(location_folder, f'transplanted_image_{self.transplanted_image_id}.jpg')

    location_folder = os.path.join(self.save_location, 'transplanted_samples', )
    if not os.path.exists(location_folder):
      os.makedirs(location_folder)
    self.modified_sample_path = os.path.join(location_folder, f"transplanted_{self.transplanted_image_id}.json")

    self.modified_sample = fo.Sample(filepath=self.image_save_location)
    self.setup_modified_sample()

    self.dataset = self.setup_dataset()

  def setup_dataset(self):
    if self.dataset_name not in fo.list_datasets():
      dataset = fo.Dataset(name=self.dataset_name)
    else:
      dataset = fo.load_dataset(self.dataset_name)
    dataset.persistent = True
    return dataset
  
  def setup_modified_sample(self):
    self.modified_sample["new_id"] = self.transplanted_image_id
    self.modified_sample["ground_truth"] = self.og_sample["ground_truth"]
    self.modified_sample.id = self.transplanted_image_id
    self.modified_sample.metadata = self.og_sample.metadata

  def add_transplanted_object(self, obj, location):
    self.transplantation_counter += 1
    self.transplantations[self.transplantation_counter] = {"object_id": obj.id, "obj_file_location": obj.file_location, "location": location}
    transplanter = ObjectTransplanter()
    transplanter.transplant_object(self.modified_image, obj, location)
    self.modified_image = transplanter.get_transplanted_image()
    self.update_modified_sample_with_transplant(obj, location)

  def transplant_with_sliding_window(self, obj, stride):
    generated_images = []
    image_width, image_height = self.modified_image.size
    print("Image size: ", image_height, image_width)
    obj_width, obj_height = obj.mask.shape[1], obj.mask.shape[0]

    for y in range(0, image_height - obj_height + 1, stride):
      for x in range (0, image_width - obj_width + 1, stride):
          print(f"Placing object at ({x}, {y})")

          overlap_exceeded = False
          for detection in self.og_sample["ground_truth"].detections:
            other_mask = detection.mask
            other_bbox = detection.bounding_box

            if obj.check_for_overlap(image_width, image_height, other_mask, other_bbox, x, y):
              overlap_exceeded = True
              print("Skipping transplant due to overlap")
              break

          print()
          
          if overlap_exceeded == True:
            continue

          # for saving the images, both as a png and as a pkl in unique folders
          unique_save_location = os.path.join(
              'transplantation/outputs/transplants_with_stride',
              f'{self.transplanted_image_id}_x{x}_y{y}'
          )

          transplanted_images_folder = os.path.join(unique_save_location, 'transplanted_images')
          transplanted_samples_folder = os.path.join(unique_save_location, 'transplanted_samples')
          os.makedirs(transplanted_images_folder, exist_ok=True)
          os.makedirs(transplanted_samples_folder, exist_ok=True)

          new_transplanted_image = ImageWithTransplantedObjects(
            sample = self.og_sample,
            save_location = unique_save_location,
            dataset_name = self.dataset_name
          )

          new_transplanted_image.add_transplanted_object(obj, (x,y))
          new_transplanted_image.save_transplanted_image()
          generated_images.append(new_transplanted_image)

    return generated_images
  

  def save_transplanted_image(self):
    self.log_modified_image()
    self.save_image()

  def log_modified_image(self):
    entry = {
       f"{self.transplanted_image_id}": {
        "original_image_id": self.og_id,
        "image_file_location": self.image_save_location,
        "sample_save_location": self.modified_sample_path,
        "transplantations": self.transplantations
       }
    }
    log_entry(self.log_file, entry, self.og_id)

  def save_image(self):
    image = Image.fromarray(self.modified_image)
    image.save(self.image_save_location)
    
    self.dataset.add_sample(self.modified_sample)
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
        print("updating modified sample")
        new_bbox = [location[0], location[1], obj.box[2], obj.box[3]]
        new_segmentation = obj.mask
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
        
