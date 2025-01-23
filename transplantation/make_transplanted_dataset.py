import fiftyone as fo
import os
import pickle

def create_fiftyone_dataset_from_pickles(pickle_folder, dataset_name="transplanted_dataset"):
    dataset = fo.Dataset(dataset_name)

    for pickle_file in os.listdir(pickle_folder):
        if pickle_file.endswith(".pkl"):
            with open(os.path.join(pickle_folder, pickle_file), 'rb') as f:
                sample = pickle.load(f)
                dataset.add_sample(sample)

    print(f"Created FiftyOne dataset '{dataset_name}' with {len(dataset)} samples")

# Example usage
if __name__ == "__main__":
    pickle_folder = 'transplantation/outputs'
    create_fiftyone_dataset_from_pickles(pickle_folder)