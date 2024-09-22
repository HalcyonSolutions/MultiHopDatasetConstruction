import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd

# Define a custom dataset class (if needed)
class CustomDataset(Dataset):
    def __init__(self, dataset, n_hops=2, transform=None):
        self.data = pd.read_csv(dataset)[:100]
        self.info = {'dataset':dataset, 'n_hops':n_hops}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        row = self.data.iloc[idx]
        path = row[:(2*self.info['n_hops']+1)].values
        question = row['question']
        answer = row['answer']
        return path, question, answer

# Function to create a DataLoader
def create_dataloader(dataset, n_hops, batch_size=32, shuffle=True, num_workers=0):
    # Create the dataset
    dataset = CustomDataset(dataset=dataset, n_hops=n_hops)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

# Example usage
if __name__ == "__main__":
    dataset = 'datasets/FbWiki_TriviaQA.csv'
    n_hops = 2

    dataloader = create_dataloader(dataset, n_hops)
    for path, question, answer in dataloader:
        print(path, question, answer)