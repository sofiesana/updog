# imports
from PIL import Image
import fiftyone.zoo as foz
import fiftyone as fo
from ImageObjectExtractor import ImageObjectExtractor
from utils import get_image
from ImageWithTransplantedObjects import ImageWithTransplantedObjects
from ExtractedObject import ExtractedObject
import os
import pickle 


def main():
    dataset_name = "testing"

    # delete "testing" dataset
    if "testing" in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        
    save_location = 'transplantation/outputs'
    print("importing dataset")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["segmentations"],
        max_samples = 10,
        classes= 'elephant'
    )
    dataset.persistent = True

    for sample in dataset:

        obj_extract = ImageObjectExtractor(sample, save_location=save_location, object_log_file='transplantation/outputs/extracted_objects_log.json')
        print("extracting objects")
        obj_extract.extract_objects()
    
    for sample in dataset:

        # loop through the files in the extracted_objects folder
        for file in os.listdir(os.path.join(save_location, 'extracted_objects')):
            obj = ExtractedObject(log_file_path='transplantation/outputs/extracted_objects_log.json')
            obj.load_object(os.path.join(save_location, 'extracted_objects', file))
        
            new_image = ImageWithTransplantedObjects(sample=sample, save_location=save_location, dataset_name=dataset_name)
            new_image.add_transplanted_object(obj, (50,50))
            # new_image.display_transplanted_image()
            new_image.save_transplanted_image()

    view_dataset(dataset_name)

def view_dataset(dataset_name):
    print(fo.list_datasets())
    test_dataset = fo.load_dataset(dataset_name)
    print(len(test_dataset))
    print("Loaded dataset samples:", test_dataset)
    session = fo.launch_app(test_dataset)
    session.wait()

if __name__ == "__main__":
    # First make sure the outputs folder is empty
    # run main() only once. If you want to run it again, you have to delete the folder:
    # transplantations/outputs
    # otherwise it becomes a mess.
    # once you have run it once, you can run view_dataset() to view the dataset (don't run main again)
    main()
    view_dataset()