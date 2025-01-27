import fiftyone as fo
from evaluating import evaluation   
from evaluating.Evaluator import Evaluator 

### STEP 4: EVALUATE MODEL PERFORMANCE ON TRANSPLANTED DATASET
    # matching_threshold = 0.99
    # evaluate_datasets(og_dataset_name, new_dataset_name, matching_threshold)
    # view_dataset()

def evaluate_datasets(ds_og_name, ds_trans_name, model_name):
    ds_og = fo.load_dataset(ds_og_name)
    ds_trans = fo.load_dataset(ds_trans_name)
    threshold = 0.3

    ev = Evaluator(ds_og, ds_trans, model_name)
    og_conf, og_f1, trans_conf, trans_f1, avg_matching_score, trans_obj_classification_acc, trans_obj_iou, affected_ms = ev.evaluate_datasets()
    
    print("OG Conf: ", round(og_conf, 2), " - Transplanted Conf: ", round(trans_conf, 2))
    print("OG F1: ", round(og_f1, 2), " - Transplanted F1: ", round(trans_f1, 2))
    print("Mean BBox Matching Score: ", round(avg_matching_score, 2))
    print("Transplanted Object Classification Accuracy: ", trans_obj_classification_acc)
    print("Mean Transplanted Object IoU: ", trans_obj_iou)
    print("Average Affected Score: ", round(affected_ms, 2))
    print(ev.affected_perc)

if __name__ == '__main__':
    # ds_og = 'coco-2017-validation-25'
    # ds_trans = 'transdata_0_n25'

    print(fo.list_datasets())

    ds_og = 'coco-2017-validation-25'
    # ds_trans = 'transdata_20_n2'
    ds_trans = 'transdata_100_n25'

    models_to_test = ['rtdetr-l-coco-torch', 'yolov8m-world-torch']

    for model in models_to_test:
        evaluate_datasets(ds_og, ds_trans, model)
    