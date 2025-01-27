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

def get_transplanted_object_metrics(object_id, predictions):
    results = predictions.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")
    results_ids = results["ytrue_ids"]
    matching_id_index = results_ids.index(object_id)
    object_conf = results["confidences"][matching_id_index]
    object_label = results["ytrue"][matching_id_index]
    object_pred_class = results["ypred"][matching_id_index]
    object_iou = results["ious"][matching_id_index]

    if object_label == object_pred_class:
        correctly_classified = 1
    else:
        correctly_classified = 0

    return correctly_classified, object_iou

def evaluate_datasets(og_dataset, transplanted_dataset, model_name, max_images=100, show_images=False, affected_threshold=0.3):
    model = get_model(model_name)
    stfu()

    og_and_trans_metrics = []
    matching_scores = []
    transplanted_object_metrics = [] # List of tuples (correctly_classified, object_iou)
    affected_scores = []
    affected_ToF = []

    for idx, sample in enumerate(og_dataset):
        # Get model predictions on base image #
        if show_images:
            show_data_image(sample)

        # getting predictions form the model on the original image
        predictions_og = og_dataset.match({"filepath": sample.filepath})
        predictions_og.apply_model(model, label_field="predictions")
        print(predictions_og)

        # Get metrics on base image #
        mean_conf_og, f1_score_og, og_detection_bboxs, og_detection_classes = get_metrics(predictions_og)

        results = predictions_og.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")

        trans_metrics = []
        match_found = False
        for trans_idx, trans_sample in enumerate(transplanted_dataset):
            
            # finding the transplanted images that correspond to the original image
            if trans_sample.original_image_id == sample.id:
                print("Evaluating image:", trans_sample.filename)

                match_found = True
                last_detection_id = trans_sample.ground_truth.detections[-1].id
                if show_images:
                    show_data_image(trans_sample)

                # Get model predictions on new (elephant) image #
                predictions_trans = transplanted_dataset.match({"filepath": trans_sample.filepath})
                predictions_trans.apply_model(model, label_field="predictions")

                # Get metrics on new (elephant) image #
                mean_conf_trans, f1_score_trans, trans_detection_bboxs, trans_detection_classes = get_metrics(predictions_trans)
                trans_metrics.append((mean_conf_trans, f1_score_trans))

                # Get transplanted object metrics #
                trans_obj_classified_correctly, trans_obj_iou = get_transplanted_object_metrics(last_detection_id, predictions_trans)
                transplanted_object_metrics.append((trans_obj_classified_correctly, trans_obj_iou))

                # Get bbox matching score #
                bbox_matching_score = get_bbox_matching_score(og_detection_bboxs, trans_detection_bboxs, og_detection_classes, trans_detection_classes)
                matching_scores.append(bbox_matching_score)
                # print("   BBox Matching Score: ", bbox_matching_score)
                score, affected = get_affected_matching_score(predictions_og.first(), predictions_trans.first(), affected_threshold)
                affected_scores.append(score)
                affected_ToF.append(affected)
                print(score)
                print("Affected? ", affected)

        if match_found:
            # Get the mean transplant metrics for all transplants of that original image
            mean_trans_conf = np.mean([m[0] for m in trans_metrics])
            mean_trans_f1 = np.mean([m[1] for m in trans_metrics])

            og_and_trans_metrics.append((mean_conf_og, f1_score_og, mean_trans_conf, mean_trans_f1))

        if max_images is not None and idx == max_images:
            break


    overall_mean_og_conf = np.mean([m[0] for m in og_and_trans_metrics])
    overall_mean_og_f1 = np.mean([m[1] for m in og_and_trans_metrics])

    overall_mean_trans_conf = np.mean([m[2] for m in og_and_trans_metrics])
    overall_mean_trans_f1 = np.mean([m[3] for m in og_and_trans_metrics])

    overall_mean_matching_score = np.mean(matching_scores)
    overall_mean_affected_matching_score = np.mean(affected_scores)

    print(np.sum(affected_ToF))
    print(len(affected_ToF))
    perc_affected = np.sum(affected_ToF)/len(affected_ToF)

    mean_trans_obj_classification_acc = np.mean([m[0] for m in transplanted_object_metrics])
    mean_trans_obj_iou = np.mean([m[1] for m in transplanted_object_metrics])

    return overall_mean_og_conf, overall_mean_og_f1, overall_mean_trans_conf, overall_mean_trans_f1, overall_mean_matching_score, mean_trans_obj_classification_acc, mean_trans_obj_iou, overall_mean_affected_matching_score, perc_affected
    
