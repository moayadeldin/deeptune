import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision
from utilities import transformations

class ImageDataset(Dataset):

    def __init__(self,root_dir, transform=None):

        """Args:
            root_dir (string) : Directory with all images, organized in class folders.
            transform (callable, optional): Transformations to be applied.
        """

        self.root_dir = root_dir
        if transform is None:
            self.transform = transformations
        else:
            self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir)) # returning the names of the folders as each name represents a class

        # We load all images and their respective labels and append their paths to their corresponding lists.

        for label_idx, class_folder in enumerate(self.classes):

            class_folder_path = os.path.join(root_dir,class_folder)

            for img_name in os.listdir(class_folder_path):

                img_path = os.path.join(class_folder_path,img_name)

                self.image_paths.append(img_path)

                self.labels.append(label_idx)

    
    def __len__(self):

        return len(self.image_paths)
    
    def __getitem__(self,idx):

        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(img)

    
        return image,label
    


