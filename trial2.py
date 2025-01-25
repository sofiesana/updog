from evaluating import evaluation
from transplantation.DatasetMaker import DatasetMaker
import fiftyone.zoo as foz
import fiftyone as fo
import os
import shutil

def view_dataset():
    ds = fo.load_dataset('test1')
    session = fo.launch_app(ds)
    session.wait()

def evaluate_datasets(og_dataset_name, new_dataset_name, threshold):
    ds_og = fo.load_dataset(og_dataset_name)
    ds_trans = fo.load_dataset(new_dataset_name)
    model_name = 'rtdetr-l-coco-torch'

    og_conf, og_f1, trans_conf, trans_f1, avg_matching_score, affected_matching_score, percentage_affected = evaluation.evaluate_datasets(ds_og, ds_trans, model_name, max_images=None, show_images=False, affected_threshold=threshold)
    
    print("OG Conf: ", round(og_conf, 2), " - Transplanted Conf: ", round(trans_conf, 2))
    print("OG F1: ", round(og_f1, 2), " - Transplanted F1: ", round(trans_f1, 2))
    print("Average BBox Matching Score: ", round(avg_matching_score, 2))
    print("Average Affected Score: ", round(affected_matching_score, 2))
    print("Percentage of affected images:", percentage_affected(100))

def delete_previous_coco_load():
     # Define the path to the FiftyOne dataset directory
    fiftyone_datasets_dir = os.path.expanduser("~/fiftyone")  # Adjust this path if necessary

    # Define the path to the COCO dataset directory
    coco_dataset_dir = os.path.join(fiftyone_datasets_dir, "coco-2017")  # Update with the correct directory name

    # Check if the directory exists and delete it
    if os.path.exists(coco_dataset_dir):
        shutil.rmtree(coco_dataset_dir)
        print(f"Deleted the dataset directory: {coco_dataset_dir}")
    else:
        print(f"Dataset directory not found: {coco_dataset_dir}")

    for dataset in fo.list_datasets():
        fo.delete_dataset(dataset)

if __name__ == '__main__':
    # delete_previous_coco_load()
    skip_extraction = True
    num_images = 25
    stride_size = 200
    save_folder = 'transplantation/outputs'
    classes = None
    # make a new dataset depending on the classes needed
    if classes is not None:
        classes_for_name = classes
    else:
        classes_for_name = ''

    og_dataset_name = 'coco-2017-validation-' + classes_for_name + str(25)

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

    allowed_overlap = False
    ovelap_threshold = 95
    new_dataset_name = 'test1_' + str(ovelap_threshold) + '_' + classes_for_name+ 'n' +str(num_images)
    matching_threshold = 0.99
    print(new_dataset_name)
    dm = DatasetMaker(stride_size, save_folder, og_dataset_name, new_dataset_name, allowed_overlap, overlap_threshold=ovelap_threshold, auto_add=True, )
    # dm.extract_all_objects()
    dm.print_no_of_available_objects()
    dm.run_dataset_maker(skip_extraction=True)
    # evaluate_datasets(og_dataset_name, new_dataset_name, matching_threshold)
    # view_dataset()
