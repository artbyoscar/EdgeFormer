import torch
import os

# Load the dataset
dataset_path = 'data/focused/text_dataset.pt'
if os.path.exists(dataset_path):
    dataset = torch.load(dataset_path)
    print(f"Dataset type: {type(dataset)}")
    
    # If it's a list, check the first few items
    if isinstance(dataset, list):
        print(f"Dataset length: {len(dataset)}")
        print(f"First item type: {type(dataset[0])}")
        
        # Check if each item is a tensor or dictionary
        if isinstance(dataset[0], torch.Tensor):
            print(f"Input tensor shape: {dataset[0].shape}")
            print(f"First few values: {dataset[0][:10]}")
        elif isinstance(dataset[0], dict):
            print(f"Dictionary keys: {dataset[0].keys()}")
            if 'input_ids' in dataset[0]:
                print(f"Input tensor shape: {dataset[0]['input_ids'].shape}")
                print(f"First few values: {dataset[0]['input_ids'][:10]}")
    else:
        print("Dataset is not a list")
else:
    print(f"Dataset file not found at {dataset_path}")