import numpy as np

def get_mean_conf(results):
  confidences = results.confs
  non_none_confidences = [c for c in confidences if c is not None]
  mean_conf = np.mean(non_none_confidences)

  return mean_conf


def get_f1_score(results):
  f1_score = results.metrics()['fscore']
  return f1_score


def get_bboxs(model_predictions):
  detection_bboxs = []
  for sample in model_predictions:  # Iterate through samples in the view
    for detection in sample.predictions.detections:  # Access predictions for each sample
        detection_bboxs.append(detection.bounding_box)

  return detection_bboxs


def get_pred_classes(model_predictions):
  detection_classes = []
  for sample in model_predictions:  # Iterate through samples in the view
    for detection in sample.predictions.detections:  # Access predictions for each sample
        detection_classes.append(detection.label)

  return detection_classes


def get_bbox_iou(bbox1, bbox2):
    # Get the coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate the area of intersection
    i_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of union
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    u_area = bbox1_area + bbox2_area - i_area

    # Calculate iou
    iou = i_area / u_area

    return iou


def get_bbox_matching_score(bboxs_list1, bboxs_list2, classes_list1, classes_list2, match_class = True):
  # This function calculates the iou between each bbox in the bbox list 1
  # to the highest matching bbox in the second bbox list. The final score is
  # the mean IoU.
  # The best score is 1, the worst is 0
  unaltered_bboxs_list2 = bboxs_list2.copy()

  ious = []
  for bbox1 in bboxs_list1:
    max_iou = 0
    max_iou_bbox = None

    for bbox2 in bboxs_list2:
      if match_class:
        class_1 = classes_list1[bboxs_list1.index(bbox1)]
        class_2 = classes_list2[unaltered_bboxs_list2.index(bbox2)]
        if class_1 == class_2:
          iou = get_bbox_iou(bbox1, bbox2)
          if iou > max_iou:
            max_iou = iou
            max_iou_bbox = bbox2
      else:
        iou = get_bbox_iou(bbox1, bbox2)
        if iou > max_iou:
          max_iou = iou
          max_iou_bbox = bbox2

    ious.append(max_iou)
    if max_iou_bbox is not None:
      bboxs_list2.remove(max_iou_bbox) # Once closest is found then remove (could maybe change)

  mean_iou = np.mean(ious)

  return mean_iou

