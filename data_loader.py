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
        
        # Assuming images are in root_dir and filenames end with two characters indicating fingers and hand, e.g., "[...]_5R.jpg"
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                self.image_paths.append(os.path.join(root_dir, filename))
                # Extract label from the last two characters before extension (first char is number of fingers)
                base_name = filename.split('.')[0]
                last_two = base_name[-2:]
                label = int(last_two[0])
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