# imports
from PIL import Image
import fiftyone.zoo as foz
import fiftyone as fo
from ImageObjectExtractor import ImageObjectExtractor
from utils import get_image
from ImageWithTransplantedObjects import ImageWithTransplantedObjects
from ExtractedObject import ExtractedObject
import os


def main():
    save_location = 'transplantation/outputs'
    print("importing dataset")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["segmentations"],
        max_samples = 20,
        classes= 'elephant'
    )
    dataset.persistent = True

    sample = dataset.first()
    # obj_extract = ImageObjectExtractor(sample, save_location=save_location, object_log_file='transplantation/outputs/extracted_objects_log.json')
    # print("extracting objects")
    # obj_extract.extract_objects()

    new_image = ImageWithTransplantedObjects(sample=sample, save_location=save_location, dataset_name="testing")
    obj = ExtractedObject(log_file_path='transplantation/outputs/extracted_objects_log.json')
    obj.load_object(os.path.join(save_location, f'extracted_objects/elephant_{sample.id}.pkl'))
    # obj.display_extracted_object()
    new_image.add_transplanted_object(obj, (0,0))
    # new_image.add_transplanted_object(obj, (200, 200))
    new_image.display_transplanted_image()
    new_image.save_transplanted_image()

if __name__ == "__main__":
    main()