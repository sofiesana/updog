import numpy as np
import logging
import fiftyone.zoo as foz
import fiftyone as fo
from utils import get_bboxs, get_pred_classes, get_mean_conf, get_f1_score, get_bbox_matching_score

if __name__ == "__main__":
    model = foz.load_zoo_model('faster-rcnn-resnet50-fpn-coco-torch')

    print("importing dataset")
    dataset = fo.load_dataset("testing")
    
    logger = logging.getLogger("fiftyone.utils.eval.detection")
    logger.setLevel(logging.WARNING)

    max_images = 2
    num_trans_per_og = 1

    og_images = dataset[:1] # Placeholder datasets right now
    transplanted_images = dataset[:2] # Will change when we got the datasets

    # session = fo.launch_app(og_images)

    og_and_trans_metrics = []
    matching_scores = []

    for idx, sample in enumerate(og_images):
        # Get model predictions on base image #
        print("sfp: ", sample.filepath)
        predictions_og = dataset.match({"filepath": sample.filepath})
        predictions_og.apply_model(model, label_field="predictions")

        # Get metrics on base image #
        results_og = predictions_og.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")
        mean_conf_og = get_mean_conf(results_og) # Get confidences
        f1_score_og = get_f1_score(results_og) # Get F1 scores
        print("OG Mean Confidence: ", mean_conf_og, " - F1 Score: ", f1_score_og)
        og_detection_bboxs = get_bboxs(predictions_og) # Get bounding boxes
        og_detection_classes = get_pred_classes(predictions_og) # Get list of each detections class

        trans_metrics = []
        transplated_subset = transplanted_images[idx * num_trans_per_og : (idx + 1) * num_trans_per_og]
        for trans_idx, trans_sample in enumerate(transplated_subset):
            ### Transplant object into new image ###
            # Do Sophies Code

            # Get model predictions on new (elephant) image #
            print("   tsfp: ", trans_sample.filepath)
            predictions_trans = dataset.match({"filepath": trans_sample.filepath})
            predictions_trans.apply_model(model, label_field="predictions")

            # Get metrics on new (elephant) image #
            results_trans = predictions_trans.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")
            mean_conf_trans = get_mean_conf(results_trans) # Get confidences
            f1_score_trans = get_f1_score(results_trans) # Get F1 scores
            trans_metrics.append((mean_conf_trans, f1_score_trans))
            print("   Trans Mean Confidence: ", mean_conf_trans, " - F1 Score: ", f1_score_trans)

            trans_detection_bboxs = get_bboxs(predictions_trans) # Get bounding boxes
            trans_detection_classes = get_pred_classes(predictions_trans) # Get list of each detections class

            bbox_matching_score = get_bbox_matching_score(og_detection_bboxs, trans_detection_bboxs, og_detection_classes, trans_detection_classes)
            matching_scores.append(bbox_matching_score)
            print("   BBox Matching Score: ", bbox_matching_score)

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

    print()
    print("OG Images --- Mean Conf: ", overall_mean_og_conf, " - Mean F1: ", overall_mean_og_f1)
    print("Trans Images --- Mean Conf: ", overall_mean_trans_conf, " - Mean F1: ", overall_mean_trans_f1)
    print("Mean BBox Matching Score: ", overall_mean_matching_score)
    
