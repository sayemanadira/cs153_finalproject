import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FingersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for filename in sorted(os.listdir(root_dir)):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root_dir, filename)

                base_name = os.path.splitext(filename)[0]   # remove extension
                suffix = base_name.split('_')[-1]          # e.g. "5R"
                label = int(suffix[0])                     # e.g. 5

                self.image_paths.append(path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loader(root_dir, batch_size=32, shuffle=True, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, 
                                    translate=(0.1, 0.1), 
                                    scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = FingersDataset(root_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def load_data(root_dir, batch_size=32, shuffle=True, transform=None):
    """
    Loads the dataset and returns a DataLoader.
    Args:
        root_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Transformations to apply to the images.
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    return get_data_loader(root_dir, batch_size, shuffle, transform)