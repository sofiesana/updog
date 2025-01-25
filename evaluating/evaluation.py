import numpy as np
import fiftyone.zoo as foz
import fiftyone as fo
from .utils import *

def get_model(model_name):
    return foz.load_zoo_model(model_name)

def get_metrics(predictions, verbose=False):
    results = predictions.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")
    
    mean_conf_og = get_mean_conf(results) # Get confidences
    f1_score_og = get_f1_score(results) # Get F1 scores

    detection_bboxs = get_bboxs(predictions) # Get bounding boxes
    detection_classes = get_pred_classes(predictions) # Get list of each detections class

    if verbose:
        print("Mean Confidence: ", mean_conf_og, " - F1 Score: ", f1_score_og)


    return mean_conf_og, f1_score_og, detection_bboxs, detection_classes

def evaluate_datasets(og_dataset, transplanted_dataset, model_name, max_images=100, show_images=False):
    model = get_model(model_name)
    stfu()

    og_and_trans_metrics = []
    matching_scores = []

    for idx, sample in enumerate(og_dataset):
        # Get model predictions on base image #
        if show_images:
            show_data_image(sample)

        predictions_og = og_dataset.match({"filepath": sample.filepath})
        predictions_og.apply_model(model, label_field="predictions")

        # Get metrics on base image #
        mean_conf_og, f1_score_og, og_detection_bboxs, og_detection_classes = get_metrics(predictions_og)

        trans_metrics = []
        match_found = False
        for trans_idx, trans_sample in enumerate(transplanted_dataset):
            if trans_sample.original_image_id == sample.id:
                match_found = True
                if show_images:
                    show_data_image(trans_sample)

                # Get model predictions on new (elephant) image #
                predictions_trans = transplanted_dataset.match({"filepath": trans_sample.filepath})
                predictions_trans.apply_model(model, label_field="predictions")

                # Get metrics on new (elephant) image #
                mean_conf_trans, f1_score_trans, trans_detection_bboxs, trans_detection_classes = get_metrics(predictions_trans)
                trans_metrics.append((mean_conf_trans, f1_score_trans))

                # Get bbox matching score #
                bbox_matching_score = get_bbox_matching_score(og_detection_bboxs, trans_detection_bboxs, og_detection_classes, trans_detection_classes)
                matching_scores.append(bbox_matching_score)
                # print("   BBox Matching Score: ", bbox_matching_score)

        if match_found:
            # Get the mean transplant metrics for all transplants of that image
            mean_trans_conf = np.mean([m[0] for m in trans_metrics])
            mean_trans_f1 = np.mean([m[1] for m in trans_metrics])

            og_and_trans_metrics.append((mean_conf_og, f1_score_og, mean_trans_conf, mean_trans_f1))

        if idx == max_images:
            break


    overall_mean_og_conf = np.mean([m[0] for m in og_and_trans_metrics])
    overall_mean_og_f1 = np.mean([m[1] for m in og_and_trans_metrics])

    overall_mean_trans_conf = np.mean([m[2] for m in og_and_trans_metrics])
    overall_mean_trans_f1 = np.mean([m[3] for m in og_and_trans_metrics])

    overall_mean_matching_score = np.mean(matching_scores)

    return overall_mean_og_conf, overall_mean_og_f1, overall_mean_trans_conf, overall_mean_trans_f1, overall_mean_matching_score
    
