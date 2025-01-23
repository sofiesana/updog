# imports
from PIL import Image
import matplotlib.pyplot as plt
import json
import os

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

def log_transplantation(log_file, new_image_id, obj_class, obj_image_id, location):
    log_entry = {
        "new_image_id": new_image_id,
        "object_class": obj_class,
        "object_image_id": obj_image_id,
        "location": location
    }

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []

    log_data.append(log_entry)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)