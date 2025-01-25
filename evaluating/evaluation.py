import numpy as np
import fiftyone.zoo as foz
import fiftyone as fo
from .utils import *

def get_model(model_name):
    return foz.load_zoo_model(model_name)

def evaluate_datasets(og_dataset, transplanted_dataset, model_name, max_images=100, show_images=False, affected_threshold=0):
    model = get_model(model_name)
    stfu()

    og_and_trans_metrics = []
    matching_scores = []
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

        trans_metrics = []
        match_found = False
        for trans_idx, trans_sample in enumerate(transplanted_dataset):
            
            # finding the transplanted images that correspond to the original image
            if trans_sample.original_image_id == sample.id:
                print("Evaluating image:", trans_sample.filename)

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

    return overall_mean_og_conf, overall_mean_og_f1, overall_mean_trans_conf, overall_mean_trans_f1, overall_mean_matching_score, overall_mean_affected_matching_score, perc_affected
    
