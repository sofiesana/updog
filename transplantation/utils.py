# imports
from PIL import Image
import matplotlib.pyplot as plt

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