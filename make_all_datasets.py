from evaluating import evaluation
from transplantation.DatasetMaker import DatasetMaker
import fiftyone.zoo as foz
import fiftyone as fo
import os
import shutil
from transplantation.DatasetObjectExtractor import DatasetObjectExtractor

if __name__ == '__main__':
    
    ### STEP 1: DOWNLOADING THE IMAGES FROM FIFTYONE AND MAKING THE DATASET (only needs to be done once)
    num_images = 2
    classes = None
    # make a new dataset depending on the classes needed
    if classes is not None:
        classes_for_name = classes
    else:
        classes_for_name = ''

    og_dataset_name = 'coco-2017-validation-' + classes_for_name + str(num_images)

    if og_dataset_name not in fo.list_datasets():
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["segmentations"],
            max_samples = num_images,
            classes = classes,
            dataset_name = og_dataset_name,
            shuffle=True
        )
        dataset.persistent = True
    else:
        dataset = fo.load_dataset(og_dataset_name)

    ### STEP 2: EXTRACT THE OBJECTS FROM THE DATASET

    # This will extract all the objects and make them ready for the transplantation step
    save_folder = 'transplantation/outputs'
    extractor = DatasetObjectExtractor(dataset, save_folder, filter=True, filter_type='min', og_dataset_name=og_dataset_name)
    extractor.extract_all_objects()
    extractor.print_no_of_available_objects()

    ### STEP 3: GENERATE THE TRANSPLANTED IMAGES FOR THE GIVEN PARAMETERS

    # define parameters:
    stride_size = 100
    ovelap_threshold = 20

    # generated dataset name:
    new_dataset_name = 'transdata_' + str(ovelap_threshold) + '_' + classes_for_name+ 'n' +str(num_images)
    print(new_dataset_name)
    # run maker
    dm = DatasetMaker(stride_size, save_folder, og_dataset_name, new_dataset_name, allow_overlap=False, overlap_threshold=ovelap_threshold, auto_add=True, )
    dm.run_dataset_maker(skip_extraction=True)
    dm.extract_dataset()

    ### STEP 4: EVALUATE MODEL PERFORMANCE ON TRANSPLANTED DATASET
    # matching_threshold = 0.99
    # evaluate_datasets(og_dataset_name, new_dataset_name, matching_threshold)
    # view_dataset()