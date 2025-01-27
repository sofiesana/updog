from evaluating import evaluation
from transplantation.DatasetMaker import DatasetMaker
import fiftyone.zoo as foz
import fiftyone as fo
import os
import shutil
from transplantation.DatasetObjectExtractor import DatasetObjectExtractor
from transplantation.utils import import_dataset, view_dataset, delete_previous_coco_load

import time



if __name__ == '__main__':
    ### STEP 0: CLEARING PREVIOUS DOWNLOADED IMAGES (if you haven't run this before)
    # delete_previous_coco_load() # I comment this out cuz I got the whole dataset already

    start_time = time.time()
    ### STEP 1: DOWNLOADING THE IMAGES FROM FIFTYONE AND MAKING THE DATASET (only needs to be done once)
    num_images = 25
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
            shuffle=False
        )
        dataset.persistent = True
    else:
        dataset = fo.load_dataset(og_dataset_name)

    # view_dataset(dataset)
    # view_dataset(dataset)

    ### STEP 2: EXTRACT THE OBJECTS FROM THE DATASET

    # This will extract all the objects and make them ready for the transplantation step
    save_folder = 'transplantation/outputs'
    extractor = DatasetObjectExtractor(dataset, save_folder, filter=True, filter_type='min', og_dataset_name=og_dataset_name)
    extractor.extract_all_objects()
    extractor.print_no_of_available_objects()

    # # ### STEP 3: GENERATE THE TRANSPLANTED IMAGES FOR THE GIVEN PARAMETERS
    stride_size = 100
    
    ## Dataset 1 (20% overlap allowed): 
    # ovelap_threshold = 20
    # # generated dataset name:
    # new_dataset_name = 'transdata_' + str(ovelap_threshold) + '_' + classes_for_name+ 'n' +str(num_images)
    # print(new_dataset_name)
    # # run maker
    # dm = DatasetMaker(stride_size, save_folder, og_dataset_name, new_dataset_name, allow_overlap=False, overlap_threshold=ovelap_threshold, auto_add=False)
    # dm.run_dataset_maker()
    # dm.print_no_of_images_created()

    
    # ## Dataset 2 (no overlap allowed): 
    # # generated dataset name:
    # ovelap_threshold = 0
    # new_dataset_name = 'transdata_' + str(ovelap_threshold) + '_' + classes_for_name+ 'n' +str(num_images)
    # print(new_dataset_name)
    # # run maker
    # dm = DatasetMaker(stride_size, save_folder, og_dataset_name, new_dataset_name, overlap_threshold=ovelap_threshold, allow_overlap=False, auto_add=False)
    # dm.run_dataset_maker()
    # dm.print_no_of_images_created()

    ## Dataset 3 (all overlap allowed): 
    ovelap_threshold = 100
    # generated dataset name:
    new_dataset_name = 'transdata_' + str(ovelap_threshold) + '_' + classes_for_name+ 'n' +str(num_images)
    print(new_dataset_name)
    # run maker
    dm = DatasetMaker(stride_size, save_folder, og_dataset_name, new_dataset_name, allow_overlap=True, overlap_threshold=ovelap_threshold, auto_add=False)
    dm.run_dataset_maker()
    dm.print_no_of_images_created()

    # view_dataset(dataset)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    

    