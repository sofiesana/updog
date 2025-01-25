from .ImageObjectExtractor import ImageObjectExtractor
import json
import os

class DatasetObjectExtractor():
    def __init__(self, dataset, save_location, filter, filter_type, og_dataset_name):
        self.dataset = dataset
        self.save_location = save_location
        self.filter = filter
        self.filter_type = filter_type
        self.og_dataset_name = og_dataset_name

    def extract_all_objects(self):
        print("Extracting Objects")
        for sample in self.dataset:
            obj_extract = ImageObjectExtractor(sample, save_location=self.save_location, filter=True, filter_type="min", og_dataset_name=self.og_dataset_name)
            obj_extract.extract_objects()
    
    def print_no_of_available_objects(self):
        with open(os.path.join(self.save_location, self.og_dataset_name, f'{self.og_dataset_name}_extracted_objects_log.json')) as f:
            objects_log = json.load(f)
        print(f'{len(objects_log)} objects are extracted and ready to transplant')
