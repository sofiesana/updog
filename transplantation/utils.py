# imports
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

def get_next_id(tracker_file):
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            data = json.load(f)
            last_id = data.get('last_id', 0)
    else:
        last_id = 0

    next_id = last_id + 1

    with open(tracker_file, 'w') as f:
        json.dump({'last_id': next_id}, f)

    return next_id

def get_image(sample):
  original_image_path = sample.filepath
  img = Image.open(original_image_path)
  return img

def display_image(self, image):
    image = Image.fromarray(image)
    display(image)
    pass

# def display(image):
#     image.show()

def display(image):
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show(block=True)  # This will block execution until the window is closed

def log_entry(log_file, log_entry, id):
    if os.path.exists(log_file):
      with open(log_file, 'r') as f:
        log_data = json.load(f)
    else:
      log_data = []

    if is_not_yet_logged(log_file, id):
      log_data.append(log_entry)
      with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    else:
      print("Object already logged")

def is_not_yet_logged(log_file_path, id):
    if os.path.exists(log_file_path):
      with open(log_file_path, 'r') as f:
        log_data = json.load(f)
      for entry in log_data:
        if id in entry:
          return False
    return True

def load_sample_from_json(filepath):
    with open(filepath, 'r') as f:
        sample_dict = json.load(f)
    return fo.Sample.from_dict(sample_dict)