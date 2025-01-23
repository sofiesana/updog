import fiftyone as fo
import os
import pickle
import json

def create_fiftyone_dataset_from_pickles(pickle_folder, dataset):

    for json_file in os.listdir(pickle_folder):
        if json_file.endswith(".json"):
            with open(os.path.join(pickle_folder, json_file), 'rb') as f:
                sample_json = json.load(f)
                sample = fo.Sample.from_dict(sample_json)
                dataset.add_sample(sample)

    print(f"Created FiftyOne dataset '{dataset_name}' with {len(dataset)} samples")

    return dataset

# Example usage
if __name__ == "__main__":
    pickle_folder = 'transplantation/outputs/transplanted_samples'
    dataset_name = 'transplanted_dataset'
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name)
    create_fiftyone_dataset_from_pickles(pickle_folder, dataset)
    session = fo.launch_app(dataset)
    session.wait()