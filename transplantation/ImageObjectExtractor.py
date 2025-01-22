# imports
import numpy as np
from PIL import Image
from IPython.display import display
from MaskExtractor import MaskExtractor


class ImageObjectExtractor():
  def __init__(self, sample, save_location):
    self.sample = sample
    self.id = sample.id
    if save_location is None:
      self.save_location = ''
    else:
      self.save_location = save_location
    self.img = self.get_image(sample)
    self.all_segmentations = sample.ground_truth.detections
    self.mask = None
    self.box = None
    self.mask_extractor = None

  def get_image(self, sample):
    original_image_path = self.sample.filepath
    img = Image.open(original_image_path)
    return img

  def reset_variables(self):
    self.mask = None
    self.box = None
    self.mask_extractor = None

  def get_mask_and_box(self, segmentation):
    self.mask = segmentation.mask
    self.box = segmentation.bounding_box

  def extract_objects(self, class_label=None):
    label_found = False
    for segmentation in self.all_segmentations:
      if class_label is None or segmentation.label == class_label:
        label = segmentation.label
        label_found = True
        self.get_mask_and_box(segmentation)

        self.mask_extractor = MaskExtractor(self.box, self.mask, np.array(self.img), self.id, label, self.save_location)
        self.mask_extractor.run_extractor()

        self.reset_variables()
    if not label_found:
      print("Class not found in sample")
