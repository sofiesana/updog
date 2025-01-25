import os
from .ImageWithTransplantedObjects import ImageWithTransplantedObjects
from .ImageObjectExtractor import ImageObjectExtractor
import json
from .utils import get_id_at_index
import fiftyone as fo
from .ExtractedObject import ExtractedObject
from PIL import Image
import shutil
from tqdm import tqdm

class DatasetMaker():
    def __init__(self, stride_size, save_folder, og_dataset_name, new_dataset_name, allow_overlap=True, overlap_threshold=100, auto_add = False):
        self.stride_size = stride_size
        self.auto_add = auto_add
        self.og_dataset_name = og_dataset_name
        self.og_dataset = fo.load_dataset(og_dataset_name)
        self.new_dataset_name = new_dataset_name
        self.allow_overlap = allow_overlap
        self.save_location = save_folder
        self.save_folder = os.path.join(save_folder, og_dataset_name)
        self.current_og_sample = None
        self.dataset = None
        self.dataset_save_location = os.path.join(self.save_folder, new_dataset_name)
        self.check_dataset_availability()
        self.dataset.persistent = True
        self.overlap_threshold = overlap_threshold
    
    def check_dataset_availability(self):
        print(fo.list_datasets())
        if self.new_dataset_name in fo.list_datasets():
            if not self.auto_add:
                choice = input(f"Dataset {self.new_dataset_name} already exists. Do you want to delete it or add to it? (delete/add): ").strip().lower()
                if choice == 'delete':
                    print(f"Deleting dataset {self.new_dataset_name}. Creating a new dataset for {self.new_dataset_name}.")
                    fo.delete_dataset(self.new_dataset_name)
                    self.dataset = fo.Dataset(name=self.new_dataset_name)
                    if os.path.exists(self.dataset_save_location):
                        shutil.rmtree(self.dataset_save_location)
                        print(f"Folder '{self.dataset_save_location}' has been deleted.")
                elif choice == 'add':
                    print(f"Adding to the existing dataset {self.new_dataset_name}.")
                    self.dataset = fo.load_dataset(self.new_dataset_name)
                else:
                    print("Invalid choice. Please enter 'delete' or 'add'.")
            else: 
                print(f"Adding to the existing dataset {self.new_dataset_name}.")
                self.dataset = fo.load_dataset(self.new_dataset_name)
        else:
            print(f"Dataset {self.new_dataset_name} does not exist. Creating a new dataset")
            self.dataset = fo.Dataset(name=self.new_dataset_name)
    
    def print_no_of_available_objects(self):
        with open(os.path.join(self.save_folder, f'{self.og_dataset_name}_extracted_objects_log.json')) as f:
            objects_log = json.load(f)
        print(f'{len(objects_log)} objects are extracted and ready to transplant')
    
    def print_no_of_images_created(self):
        path = os.path.join(self.dataset_save_location, f'{self.new_dataset_name}_transplantation_log.json')
        if os.path.exists(path):
            with open(os.path.join(self.dataset_save_location, f'{self.new_dataset_name}_transplantation_log.json')) as f:
                img_log = json.load(f)
            print(f'{len(img_log)} images have been created through transplantation')
        else:
            print('0 images have been created through transplantation')

    def run_dataset_maker(self):
        path_to_objects = os.path.join(self.save_folder, f'{self.og_dataset_name}_extracted_objects_log.json')
        if not os.path.exists(path_to_objects):
            print("ERROR: No objects available to transplant. Please run the DatasetObjectExtractor first.")
            return
        
        with open(os.path.join(self.save_folder, f'{self.og_dataset_name}_extracted_objects_log.json')) as f:
            objects_log = json.load(f)

        self.print_no_of_available_objects()

        for sample in tqdm(self.og_dataset, desc="Processing samples"):
            self.current_og_sample = sample
            # print(f"TRANSPLANTING INTO IMAGE {sample.id}")
            for i in tqdm(range(len(objects_log)), desc="Transplanting objects"):
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
        obj_width, obj_height = obj.mask.shape[1], obj.mask.shape[0]

        for y in range(0, image_height - obj_height + 1, self.stride_size):
            for x in range (0, image_width - obj_width + 1, self.stride_size):
                # print(f"Placing object at ({x}, {y})")

                if not self.allow_overlap:
                    overlap_exceeded = False
                    for detection in self.current_og_sample["ground_truth"].detections:
                        other_mask = detection.mask
                        other_bbox = detection.bounding_box

                        if obj.check_for_overlap(image_width, image_height, other_mask, other_bbox, x, y, threshold=self.overlap_threshold):
                            overlap_exceeded = True
                            # print("Skipping transplant due to overlap")
                            break

                    if overlap_exceeded == True:
                        continue

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