import fiftyone.zoo as foz
import fiftyone as fo
from .utils import *
import numpy as np
import json

class Evaluator():
    def __init__(self, og_dataset, trans_dataset, model_name):
        self.og_dataset = og_dataset
        self.trans_dataset = trans_dataset
        self.model_name = model_name
        self.affected_dict = {}
        self.affected_perc = {}
        
        self.threshold_list = [0.3, 0.5, 0.7, 0.95, 0.99]
        for t in self.threshold_list:
            self.affected_dict[t] = []
            self.affected_perc[t] = None
        self.affected_scores = []

        self.og_metrics = {}
        self.og_metrics["conf"] = []
        self.og_metrics["f1"] = []
        
        self.trans_metrics = {}
        self.trans_metrics["conf"] = []
        self.trans_metrics["f1"] = []

        self.overall_og_mean_metrics = {}
        self.overall_trans_mean_metrics = {}

        self.trans_obj_metrics = {}
        self.trans_obj_metrics["cc"] = []
        self.trans_obj_metrics["iou"] = []
        
        self.overall_obj_metrics = {}

        self.matching_scores = None
        self.overall_mean_matching_score = None
        self.overall_mean_affected_matching_score = None

        self.filepath = f'evaluation/{trans_dataset.name}_{model_name}_results.json'
        pass

    def to_json(self):
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(element) for element in obj]
            return obj

        data = {
            "og_dataset": self.og_dataset.name,
            "trans_dataset": self.trans_dataset.name,
            "model_name": self.model_name,
            "affected_dict": convert_numpy(self.affected_dict),
            "affected_perc": convert_numpy(self.affected_perc),
            "threshold_list": convert_numpy(self.threshold_list),
            "affected_scores": convert_numpy(self.affected_scores),
            "og_metrics": convert_numpy(self.og_metrics),
            "trans_metrics": convert_numpy(self.trans_metrics),
            "overall_og_mean_metrics": convert_numpy(self.overall_og_mean_metrics),
            "overall_trans_mean_metrics": convert_numpy(self.overall_trans_mean_metrics),
            "trans_obj_metrics": convert_numpy(self.trans_obj_metrics),
            "overall_obj_metrics": convert_numpy(self.overall_obj_metrics),
            "matching_scores": convert_numpy(self.matching_scores),
            "overall_mean_matching_score": convert_numpy(self.overall_mean_matching_score),
            "overall_mean_affected_matching_score": convert_numpy(self.overall_mean_affected_matching_score)
        }

        with open(self.filepath, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def get_model(self, model_name):
        return foz.load_zoo_model(model_name)

    def get_metrics(self, predictions, verbose=False):
        results = predictions.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")
        
        mean_conf_og = get_mean_conf(results) # Get confidences
        f1_score_og = get_f1_score(results) # Get F1 scores

        detection_bboxs = get_bboxs(predictions) # Get bounding boxes
        detection_classes = get_pred_classes(predictions) # Get list of each detections class

        if verbose:
            print("Mean Confidence: ", mean_conf_og, " - F1 Score: ", f1_score_og)

        return mean_conf_og, f1_score_og, detection_bboxs, detection_classes

    def get_transplanted_object_metrics(self, object_id, predictions):
        results = predictions.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")
        results_ids = results.ytrue_ids
        # matching_id_index = results_ids.index(object_id)
        matching_id_index = np.where(results_ids == object_id)[0] 
        object_label = results.ytrue[matching_id_index]
        object_pred_class = results.ypred[matching_id_index]
        object_iou = results.ious[matching_id_index]

        if object_label == object_pred_class:
            correctly_classified = 1
        else:
            correctly_classified = 0

        if object_iou[0] is None:
            print(object_iou[0])
        object_iou[0] = np.nan

        return correctly_classified, object_iou[0]

    def evaluate_datasets(self, max_images=100, show_images=False):
        model = self.get_model(self.model_name)
        stfu()

        og_and_trans_metrics = []
        self.matching_scores = []
        transplanted_object_metrics = [] # List of tuples (correctly_classified, object_iou)
        affected_scores = []
        affected_ToF = []

        for idx, sample in enumerate(self.og_dataset):
            # Get model predictions on base image #
            if show_images:
                show_data_image(sample)

            # getting predictions form the model on the original image
            predictions_og = self.og_dataset.match({"filepath": sample.filepath})
            predictions_og.apply_model(model, label_field="predictions")
            print(predictions_og)

            # Get metrics on base image #
            mean_conf_og, f1_score_og, og_detection_bboxs, og_detection_classes = get_metrics(predictions_og)
            self.og_metrics["conf"].append(mean_conf_og)
            self.og_metrics["f1"].append(f1_score_og)

            results = predictions_og.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")

            trans_metrics = []
            match_found = False
            for trans_idx, trans_sample in enumerate(self.trans_dataset):
                
                # finding the transplanted images that correspond to the original image
                if trans_sample.original_image_id == sample.id:
                    print("Evaluating image:", trans_sample.filename)

                    match_found = True
                    last_detection_id = trans_sample.ground_truth.detections[-1].id
                    if show_images:
                        show_data_image(trans_sample)

                    # Get model predictions on new (elephant) image #
                    predictions_trans = self.trans_dataset.match({"filepath": trans_sample.filepath})
                    predictions_trans.apply_model(model, label_field="predictions")

                    # Get metrics on new (elephant) image #
                    mean_conf_trans, f1_score_trans, trans_detection_bboxs, trans_detection_classes = self.get_metrics(predictions_trans)
                    trans_metrics.append((mean_conf_trans, f1_score_trans))
                      

                    # Get transplanted object metrics #
                    trans_obj_classified_correctly, trans_obj_iou = self.get_transplanted_object_metrics(last_detection_id, predictions_trans)
                    transplanted_object_metrics.append((trans_obj_classified_correctly, trans_obj_iou))
                    self.trans_obj_metrics["cc"].append(trans_obj_classified_correctly)
                    self.trans_obj_metrics["iou"].append(trans_obj_iou)
                    
                    # Get bbox matching score #
                    bbox_matching_score = get_bbox_matching_score(og_detection_bboxs, trans_detection_bboxs, og_detection_classes, trans_detection_classes)
                    self.matching_scores.append(bbox_matching_score)
                    # print("   BBox Matching Score: ", bbox_matching_score)
                    for t in self.threshold_list:
                        score, affected = get_affected_matching_score(predictions_og.first(), predictions_trans.first(), t)
                        self.affected_dict[t].append(affected)
                    self.affected_scores.append(score)

            if match_found:
                # Get the mean transplant metrics for all transplants of that original image
                mean_trans_conf = np.mean([m[0] for m in trans_metrics])
                mean_trans_f1 = np.mean([m[1] for m in trans_metrics])
                self.trans_metrics["conf"].append(mean_trans_conf)
                self.trans_metrics["f1"].append(mean_trans_f1) 

                og_and_trans_metrics.append((mean_conf_og, f1_score_og, mean_trans_conf, mean_trans_f1))

            if max_images is not None and idx == max_images:
                break
        
        for metric in self.og_metrics:
            self.overall_og_mean_metrics[metric] = np.mean(self.og_metrics[metric])
            self.overall_trans_mean_metrics[metric] = np.mean(self.trans_metrics[metric])

        overall_mean_og_conf = np.mean([m[0] for m in og_and_trans_metrics])
        overall_mean_og_f1 = np.mean([m[1] for m in og_and_trans_metrics])

        overall_mean_trans_conf = np.mean([m[2] for m in og_and_trans_metrics])
        overall_mean_trans_f1 = np.mean([m[3] for m in og_and_trans_metrics])

        overall_mean_matching_score = np.mean(self.matching_scores)
        self.overall_mean_matching_score = overall_mean_matching_score
        
        overall_mean_affected_matching_score = np.mean(self.affected_scores)
        self.overall_mean_affected_matching_score = overall_mean_affected_matching_score

        for t in self.threshold_list:
            self.affected_perc[t] = np.sum(self.affected_dict[t])/len(self.affected_dict[t])

        self.to_json()

        self.overall_obj_metrics["cc"] = np.nanmean(self.trans_obj_metrics["cc"], )
        # Convert list to a numpy array, replacing None with np.nan
        self.trans_obj_metrics["iou"] = np.array([[x if x is not None else np.nan for x in self.trans_obj_metrics["iou"]]])
        self.overall_obj_metrics["iou"] = np.nanmean(self.trans_obj_metrics["iou"])
        mean_trans_obj_classification_acc = np.nanmean([m[0] for m in transplanted_object_metrics])
        mean_trans_obj_iou = np.nanmean([m[1] for m in transplanted_object_metrics])

        self.to_json()

        return overall_mean_og_conf, overall_mean_og_f1, overall_mean_trans_conf, overall_mean_trans_f1, overall_mean_matching_score, mean_trans_obj_classification_acc, mean_trans_obj_iou, overall_mean_affected_matching_score