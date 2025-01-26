from evaluating import evaluation
from DatasetMaker import DatasetMaker


import fiftyone as fo

def main():
    stride_size = 200
    save_folder = 'transplantation/outputs'
    og_dataset_name = 'coco-2017-validation-2'
    new_dataset_name = 'test1'
    allowed_overlap = False
    dm = DatasetMaker(stride_size, save_folder, og_dataset_name, new_dataset_name, allowed_overlap)
    dm.run_dataset_maker()

def view_dataset():
    ds = fo.load_dataset('test1')
    session = fo.launch_app(ds)
    session.wait()

def evaluate_datasets():
    ds_og = fo.load_dataset('coco-2017-validation-2')
    ds_trans = fo.load_dataset('test1')

    og_metrics, trans_metrics, avg_matching_score = evaluation.evaluate_datasets(ds_og, ds_trans, 'rtdetr-l-coco-torch')
    print("OG Metrics: ", og_metrics)
    print("Transplant Metrics: ", trans_metrics)
    print("Average BBox Matching Score: ", avg_matching_score)

if __name__ == '__main__':
    main()
    view_dataset()
    evaluate_datasets()