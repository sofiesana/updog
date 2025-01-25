from evaluating import evaluation
from transplantation.DatasetMaker import DatasetMaker


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
    model_name = 'rtdetr-l-coco-torch'

    og_conf, og_f1, trans_conf, trans_f1, avg_matching_score = evaluation.evaluate_datasets(ds_og, ds_trans, model_name, max_images=None, show_images=False)
    
    print("OG Conf: ", round(og_conf, 2), " - Transplanted Conf: ", round(trans_conf, 2))
    print("OG F1: ", round(og_f1, 2), " - Transplanted F1: ", round(trans_f1, 2))
    print("Average BBox Matching Score: ", round(avg_matching_score, 2))

if __name__ == '__main__':
    main()
    evaluate_datasets()
    view_dataset()
