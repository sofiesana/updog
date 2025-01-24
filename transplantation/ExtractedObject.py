import os
import pickle as pkl
from PIL import Image
from utils import display, log_entry
import json
import numpy as np
import matplotlib.pyplot as plt

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


  def set_bb_to_origin(self, obj_x, obj_y):
     # Get the bounding box of self.mask and reset it to start from (0,0)
    obj_min_x, obj_max_x = obj_x.min(), obj_x.max()
    obj_min_y, obj_max_y = obj_y.min(), obj_y.max()

    # Calculate the offset to reset the mask to (0, 0)
    offset_x = obj_min_x
    offset_y = obj_min_y

    # "Reset" the mask by subtracting the offsets from the bounding box
    obj_min_x -= offset_x
    obj_max_x -= offset_x
    obj_min_y -= offset_y
    obj_max_y -= offset_y

    return obj_max_x, obj_min_x, obj_max_y, obj_min_y
  

  def check_for_overlap(self, image_width, image_height, other_mask, other_bbox, x, y, threshold=95):
    # Convert bounding boxes from [0, 1] to pixel coordinates
    obj_bbox = self.box
    obj_y, obj_x = self.mask.nonzero()

    # other_bbox: [xmin, ymin, xmax, ymax] in [0, 1] range
    other_y, other_x = other_mask.nonzero()

    # Convert the bounding boxes from relative (0, 1) to absolute coordinates
    obj_bbox_pixels = [int(obj_bbox[0] * image_width), int(obj_bbox[1] * image_height),
                       int(obj_bbox[2] * image_width), int(obj_bbox[3] * image_height)]
    other_bbox_pixels = [int(other_bbox[0] * image_width), int(other_bbox[1] * image_height),
                         int(other_bbox[2] * image_width), int(other_bbox[3] * image_height)]

    # Adjust the object mask position based on the x, y location provided
    print("Coords: ", obj_bbox_pixels[0] + x, obj_bbox_pixels[1] + y, obj_bbox_pixels[2] + x, obj_bbox_pixels[3] + y)
    obj_bbox_pixels = [obj_bbox_pixels[0] + x, obj_bbox_pixels[1] + y, obj_bbox_pixels[2] + x, obj_bbox_pixels[3] + y]

    # Create a mask of the object in the absolute image space
    obj_mask_in_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Adjust for the object's position and place the mask in the image
    obj_y_adjusted = obj_y + obj_bbox_pixels[1] + y
    obj_x_adjusted = obj_x + obj_bbox_pixels[0] + x
    valid_obj_y = (obj_y_adjusted >= 0) & (obj_y_adjusted < image_height)
    valid_obj_x = (obj_x_adjusted >= 0) & (obj_x_adjusted < image_width)
    valid_obj_mask = valid_obj_y & valid_obj_x
    obj_mask_in_image[obj_y_adjusted[valid_obj_mask], obj_x_adjusted[valid_obj_mask]] = 1

    # Check for overlap with the other mask, apply bounding box limits to the other mask
    other_mask_in_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Adjust for the other mask's position and place the mask in the image
    other_y_adjusted = other_y + other_bbox_pixels[1] + y
    other_x_adjusted = other_x + other_bbox_pixels[0] + x
    valid_other_y = (other_y_adjusted >= 0) & (other_y_adjusted < image_height)
    valid_other_x = (other_x_adjusted >= 0) & (other_x_adjusted < image_width)
    valid_other_mask = valid_other_y & valid_other_x
    other_mask_in_image[other_y_adjusted[valid_other_mask], other_x_adjusted[valid_other_mask]] = 1

    # Crop both masks based on the bounding box of the other mask to ensure correct positioning
    obj_mask_cropped = obj_mask_in_image[other_bbox_pixels[1]:other_bbox_pixels[3], other_bbox_pixels[0]:other_bbox_pixels[2]]
    other_mask_cropped = other_mask_in_image[other_bbox_pixels[1]:other_bbox_pixels[3], other_bbox_pixels[0]:other_bbox_pixels[2]]

    # Calculate the intersection (only where both masks are 1)
    intersection = np.logical_and(obj_mask_cropped, other_mask_cropped)

    # Calculate percentage of the other mask overlapped by the object mask
    overlap = np.sum(intersection) / np.sum(other_mask_cropped) * 100

    # Check if the overlap is above the threshold
    if overlap >= threshold:
        print(f"Overlap detected: {overlap}%")
    else:
        print(f"No significant overlap: {overlap}%")

    return overlap


