# imports
import numpy as np

class ObjectTransplanter():
  def __init__(self):
    self.image = None
    self.mask_with_pixels = None
    self.mask = None
    self.transplant_location = None
    self.image_with_transplant = None

  def setup_transplant(self, image, obj, transplant_location):
    self.image = np.array(image)
    self.mask_with_pixels = obj.mask_with_pixels
    self.mask = obj.mask
    self.transplant_location = transplant_location

  def transplant_object(self, image, obj, transplant_location):
    self.setup_transplant(image, obj, transplant_location)
    self.run_transplant()

  def is_not_masked(self, x, y):
    return self.mask[y,x]

  def is_not_out_of_bounds(self, x, y):
    return x < self.image.shape[1] and y < self.image.shape[0]

  def is_cut_off(self, loc_x, loc_y):
    if (loc_x + self.mask.shape[1] > self.image.shape[1]) or (loc_y + self.mask.shape[0] > self.image.shape[0]):
      print("Warning: The transplanted image is cut off by the edge of the image")
      return True
    return False

  def run_transplant(self, allow_cut_off=True):
    image_with_transplant = None
    loc_x, loc_y = self.transplant_location
    is_cut = self.is_cut_off(loc_x, loc_y) # save this somehow if important

    if is_cut == True and allow_cut_off == False:
      print("Warning: The transplanted image is cut off by the edge of the image, transplant aborted. Returns None.")
    else:
      image_with_transplant = self.image.copy()
      for y in range(self.mask.shape[0]):
        for x in range(self.mask.shape[1]):
          if self.is_not_masked(x, y) and self.is_not_out_of_bounds(loc_x + x, loc_y + y):
            image_with_transplant[loc_y + y, loc_x + x] = self.mask_with_pixels[y, x]

    self.image_with_transplant = image_with_transplant

  def get_transplanted_image(self):
    return self.image_with_transplant