# add imports
import numpy as np
from PIL import Image
from utils import display
from ExtractedObject import ExtractedObject

class MaskExtractor():
  def __init__(self, box, mask, pixels, id, obj_id, class_label, save_location, object_log_file):
    self.object_log_file = object_log_file
    self.box = box
    self.mask = mask
    self.pixels = pixels
    self.masked_pixels = np.copy(pixels)
    self.id = id
    self.obj_id = obj_id
    self.class_label = class_label
    self.save_location = save_location

    # Getting all the dimensions
    self.top_left_x, self.top_left_y, self.width, self.height = box
    self.box_in_pixels = self.get_box_in_pixels()
    self.x_min, self.x_max, self.y_min, self.y_max = self.get_x_y_min_max()

    self.mask_with_pixels = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

  def get_box_in_pixels(self):
    pix = [self.top_left_x*self.pixels.shape[1], self.top_left_y*self.pixels.shape[0], self.width*self.pixels.shape[1], self.height*self.pixels.shape[0]]
    pix = [int(x) for x in pix]
    return pix

  def get_x_y_min_max(self):
    x_min = self.box_in_pixels[0]
    x_max = self.box_in_pixels[0] + self.box_in_pixels[2]
    y_min = self.box_in_pixels[1]
    y_max = self.box_in_pixels[1] + self.box_in_pixels[3]

    return x_min, x_max, y_min, y_max

  def pixel_in_range(self, x, y, image):
    if x < image.shape[1] and y < image.shape[0]:
      return True
    return False

  def pixel_in_box(self, x, y):
    if x > self.x_min or x < self.x_max or y > self.y_min or y < self.y_max:
        if self.pixel_in_range(x, y, self.pixels):
          return True
    return False

  def get_new_x_y(self, x, y):
    x_coord = x - self.x_min
    y_coord = y - self.y_min
    return x_coord, y_coord

  def pixel_is_not_masked(self, x, y):
    x_coord, y_coord = self.get_new_x_y(x, y)
    if 0 <= x_coord < self.mask.shape[1] and 0 <= y_coord < self.mask.shape[0]:
        return self.mask[y_coord, x_coord]
    else:
        return False

  def filter_pixels(self, x, y):
    if not self.pixel_in_box(x, y):
      self.masked_pixels[y, x] = (0, 0, 0)
    else:
      if self.pixel_is_not_masked(x, y):
        x_coord, y_coord = self.get_new_x_y(x, y)
        self.mask_with_pixels[y_coord, x_coord] = self.masked_pixels[y, x]
      elif self.pixel_in_range(x, y, self.masked_pixels):
        self.masked_pixels[y, x] = (0, 0, 0)

  def run_extractor(self, display=False):
    for y in range(self.pixels.shape[0]):
      for x in range(self.pixels.shape[1]):
        self.filter_pixels(x, y)
    if display:
      self.display_image(self.mask_with_pixels)
    self.save_extracted_object()

  def get_mask_with_pixels(self):
    return self.mask_with_pixels

  def get_masked_pixels(self):
    return self.masked_pixels

  def display_image(self, image):
    image = Image.fromarray(image)
    display(image)
    pass

  def save_extracted_object(self):
    obj = ExtractedObject(log_file_path=self.object_log_file)
    obj.setup(self.pixels, self.mask, self.mask_with_pixels, self.id, self.obj_id, self.class_label, self.box, self.box_in_pixels, self.save_location)
    if obj.is_setup:
      obj.save_object()
      obj.save_mask_with_pixels_as_jpg()
      obj.save_mask()