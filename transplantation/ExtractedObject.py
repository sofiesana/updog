import os
import pickle as pkl
from PIL import Image
from .utils import display, log_entry, get_next_id
import json
import numpy as np
import matplotlib.pyplot as plt

class ExtractedObject():
  def __init__(self, log_file_path=None):
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
    self.obj_id = None
    self.og_img_dimensions = None
    self.mask_to_og_ratio = None

  def setup(self, image, mask, mask_with_pixels, id, obj_id, class_label, box, box_in_pixels, save_location):
    if not self.is_setup:
      # print(f"Setting up object with ID: {obj_id}")
      self.og_img_dimensions = image.shape
      obj_area = mask.shape[0] * mask.shape[1]
      img_area = np.prod(image.shape)
      self.mask_to_og_ratio = obj_area / img_area
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
      
      self.obj_id = obj_id
      self.file_location = os.path.join(location_folder, f'{self.obj_id}.pkl')
      
      #check if file_location already exists
      if os.path.exists(self.file_location):
        print(f"Setup failed: Object with ID {obj_id} already exists at {self.file_location}")
        self.is_setup = False
      else:
        # print(f"Object setup successful for ID: {obj_id}")
        self.is_setup = True
    else:
      print(f"Setup failed: Object with ID {obj_id} already setup")
  
  def log_object(self):
    if self.log_file_path is None:
      print("No log file path provided")
      return
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
  
  def plot_masks(self, image_mask_obj, image_mask_other):
    # plot the obj mask in red and the other mask in blue on the same plot

    mask_plot = np.zeros((image_mask_obj.shape[0], image_mask_obj.shape[1], 3))
    mask_plot[:, :, 0] = image_mask_obj
    mask_plot[:, :, 2] = image_mask_other

    plt.imshow(mask_plot)
    plt.title("Red: Object mask, Blue: Other mask")
    plt.show()
  

  def check_for_overlap(self, image_width, image_height, other_mask, other_bbox, x, y, threshold=95):
    """ Check if the object overlaps with another object in the image. 
    Both masks and their corresponding bounding boxes are provided. These
    are both from the same image so they are in different coordinates. """

    # get bounding box of object and mask
    obj_mask = self.mask
    obj_bbox = self.box

    # Convert the bounding boxes from relative (0, 1) to absolute coordinates
    obj_bbox_pixels = [int(obj_bbox[0] * image_width), int(obj_bbox[1] * image_height),
                       int(obj_bbox[2] * image_width), int(obj_bbox[3] * image_height)]
    other_bbox_pixels = [int(other_bbox[0] * image_width), int(other_bbox[1] * image_height),
                         int(other_bbox[2] * image_width), int(other_bbox[3] * image_height)]
    
    # print("Obj pixels: ", obj_bbox_pixels, "Other pixels: ", other_bbox_pixels)

    # create a blank image with the same size as the image
    blank_image_obj = np.zeros((image_height, image_width))
    blank_image_other = np.zeros((image_height, image_width))

    # iterate through the obj mask and set the corresponding pixels in the blank image
    for i in range(obj_mask.shape[0]):
        for j in range(obj_mask.shape[1]):
            if obj_mask[i, j] == 1:
                image_x = obj_bbox_pixels[0] + j
                image_y = obj_bbox_pixels[1] + i
                image_x_plus_offset = x + j # x is the offset since we are sliding the object
                image_y_plus_offset = y + i # y is the offset since we are sliding the object
                blank_image_obj[image_y_plus_offset, image_x_plus_offset] = 1

    # iterate through the other mask and set the corresponding pixels in the blank image
    for i in range(other_mask.shape[0]):
        for j in range(other_mask.shape[1]):
            if other_mask[i, j] == 1:
                image_x = other_bbox_pixels[0] + j
                image_y = other_bbox_pixels[1] + i
                blank_image_other[image_y, image_x] = 1

    # calculate the overlap between the two masks
    overlap = np.logical_and(blank_image_obj, blank_image_other)

    # calculate the percentage of overlap over the other object
    overlap_percentage = np.sum(overlap) / np.sum(blank_image_other) * 100

    # print(f"Overlap percentage: {overlap_percentage}")

    # self.plot_masks(blank_image_obj, blank_image_other) # uncomment to plot the masks for debugging


    # if the overlap is greater than the threshold, return True
    if overlap_percentage > threshold:
        # print(f"Overlapping")
        return True
    else:
        # print(f"Not overlapping")
        return False


