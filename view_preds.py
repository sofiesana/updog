import fiftyone as fo
import fiftyone.zoo as foz

def show_predictions(ds_og_name, ds_trans_name, model_name, dataset_to_view, max_images=2):
    ds_og = fo.load_dataset(ds_og_name)
    ds_trans = fo.load_dataset(ds_trans_name)

    model = foz.load_zoo_model(model_name)
    ds_og.apply_model(model, label_field="predictions")
    results = ds_og.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
    )

    print(results.metrics())

    # Idk if this is needed cuz the predictions are already applied in the dataset
    # But they are based on the last model applied, so this can be used to change the model
    # for og_idx, sample in enumerate(ds_og):
    #     pred_og = ds_og.match({"filepath": sample.filepath})
    #     pred_og.apply_model(model, label_field="predictions")

    #     for trans_idx, trans_sample in enumerate(ds_trans):
    #         if trans_sample.original_image_id == sample.id:

    #             pred_trans = ds_trans.match({"filepath": sample.filepath})
    #             pred_trans.apply_model(model, label_field="predictions")

    #     if og_idx >= max_images:
    #         break

    if dataset_to_view == "og":
        # View the predictions using the FiftyOne App
        session = fo.launch_app(ds_og)
        session.wait()
    elif dataset_to_view == "trans":
        # View the predictions using the FiftyOne App
        session = fo.launch_app(ds_trans)
        session.wait()

        
     

if __name__ == '__main__':
    ds_og = 'coco-2017-validation-25'
    ds_trans = 'transdata_0_n25'

    models_to_test =  ['rtdetr-l-coco-torch']# ['yolov8m-world-torch'] # 

    # dataset_to_view = "og" 
    dataset_to_view = "og" 

    for model in models_to_test:
        show_predictions(ds_og, ds_trans, model, dataset_to_view, max_images=25)
    