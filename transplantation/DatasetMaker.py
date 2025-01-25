import os
from .ImageWithTransplantedObjects import ImageWithTransplantedObjects
from .ImageObjectExtractor import ImageObjectExtractor
import json
from .utils import get_id_at_index
import fiftyone as fo
from .ExtractedObject import ExtractedObject
from PIL import Image

class DatasetMaker():
    def __init__(self, stride_size, save_folder, og_dataset_name, new_dataset_name, allow_overlap=False):
        self.stride_size = stride_size
        self.og_dataset_name = og_dataset_name
        self.og_dataset = fo.load_dataset(og_dataset_name)
        self.new_dataset_name = new_dataset_name
        self.dataset = None
        self.check_dataset_availability()
        self.dataset.persistent = True
        self.allow_overlap = allow_overlap
        self.save_location = save_folder
        self.save_folder = os.path.join(save_folder, og_dataset_name)
        self.current_og_sample = None
        self.dataset_save_location = os.path.join(self.save_folder, new_dataset_name)
        
    
    def check_dataset_availability(self):
        print(fo.list_datasets())
        if self.new_dataset_name in fo.list_datasets():
            choice = input(f"Dataset {self.new_dataset_name} already exists. Do you want to delete it or add to it? (delete/add): ").strip().lower()
            if choice == 'delete':
                print(f"Deleting dataset {self.new_dataset_name}. Creating a new dataset for {self.new_dataset_name}.")
                fo.delete_dataset(self.new_dataset_name)
                self.dataset = fo.Dataset(name=self.new_dataset_name)
            elif choice == 'add':
                print(f"Adding to the existing dataset {self.new_dataset_name}.")
                self.dataset = fo.load_dataset(self.new_dataset_name)
            else:
                print("Invalid choice. Please enter 'delete' or 'add'.")
        else:
            print(f"Dataset {self.new_dataset_name} does not exist. Creating a new dataset")
            self.dataset = fo.Dataset(name=self.new_dataset_name)

    def extract_all_objects(self):
        print("extracting")
        for sample in self.og_dataset:
            print("test")
            obj_extract = ImageObjectExtractor(sample, save_location=self.save_location, filter=True, filter_type="min", og_dataset_name=self.og_dataset_name)
            obj_extract.extract_objects()

    def run_dataset_maker(self):
        self.extract_all_objects()

        with open(os.path.join(self.save_folder, f'{self.og_dataset_name}_extracted_objects_log.json')) as f:
            objects_log = json.load(f)

        for sample in self.og_dataset:
            self.current_og_sample = sample
            for i in range(len(objects_log)):
                id = get_id_at_index(objects_log, i)
                og_image_id = objects_log[i][id]['og_image_id']

                if og_image_id == sample.id:
                    # don't want to transplant an object from the same image
                    continue

                obj = ExtractedObject()
                obj.load_object(os.path.join(self.save_folder,f'extracted_objects/{id}.pkl'))
                self.transplant_with_sliding_window(obj)


    def transplant_with_sliding_window(self, obj):
        image_width, image_height = Image.open(self.current_og_sample.filepath).size
        print("Image size: ", image_height, image_width)
        obj_width, obj_height = obj.mask.shape[1], obj.mask.shape[0]

        for y in range(0, image_height - obj_height + 1, self.stride_size):
            for x in range (0, image_width - obj_width + 1, self.stride_size):
                print(f"Placing object at ({x}, {y})")

                if not self.allow_overlap:
                    overlap_exceeded = False
                    for detection in self.current_og_sample["ground_truth"].detections:
                        other_mask = detection.mask
                        other_bbox = detection.bounding_box

                        if obj.check_for_overlap(image_width, image_height, other_mask, other_bbox, x, y):
                            overlap_exceeded = True
                            print("Skipping transplant due to overlap")
                            break

                    if overlap_exceeded == True:
                        continue

                # # for saving the images, both as a png and as a pkl in unique folders
                # unique_save_location = os.path.join(
                #     'transplantation/outputs/transplants_with_stride',
                #     f'{self.transplanted_image_id}_x{x}_y{y}'
                # )

                # transplanted_images_folder = os.path.join(unique_save_location, 'transplanted_images')
                # transplanted_samples_folder = os.path.join(unique_save_location, 'transplanted_samples')
                # os.makedirs(transplanted_images_folder, exist_ok=True)
                # os.makedirs(transplanted_samples_folder, exist_ok=True)

                new_transplanted_image = ImageWithTransplantedObjects(
                    sample=self.current_og_sample, #new_sample,
                    save_location=self.dataset_save_location,
                    dataset_name=self.new_dataset_name,
                    filename_appendix=f'_x{x}_y{y}'
                )

                new_transplanted_image.add_transplanted_object(obj, (x,y))
                new_transplanted_image.save_transplanted_image()
                # generated_images.append(new_transplanted_image)

        # return generated_images