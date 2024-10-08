from datasets import load_dataset
from torch.utils.data import Dataset
import json
import torch
import os


class ToyDataSet(Dataset):  # for toy demo
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        data = load_dataset("Trelis/chess_pieces")
        self.train_shape = len(data['train'])
        self.test_shape = len(data['test'])
        self.dataset = data[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': example['image']},
                                             {'type': 'text', 'text': 'What kind of chess pieces in this picture?'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': example['caption']}]}
            ]
        }


class LocalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with subfolders, each containing image-caption pairs.
            processor (transformers AutoProcessor): Processor for tokenizing text and handling images.
            transform (callable, optional): Optional transform to be applied to images (if needed).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.captions = []

        # Traverse through all folders and gather image-caption pairs
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.txt'):
                        # txt_path = os.path.join(folder_path, file)
                        # img_path = txt_path.replace('.txt', '.jpg')  # Assuming image format is .jpg
                        # if os.path.exists(img_path):
                        #     with open(txt_path, 'r', encoding='utf-8') as f:
                        #         caption = f.read().strip()
                        #     self.image_paths.append(img_path)
                        #     self.captions.append(caption)
                        txt_path = os.path.join(folder_path, file)
                        img_path_jpg = txt_path.replace('.txt', '.jpg')
                        img_path_png = txt_path.replace('.txt', '.png')

                        if os.path.exists(img_path_jpg):
                            img_path = img_path_jpg
                        elif os.path.exists(img_path_png):
                            img_path = img_path_png
                        else:
                            continue

                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        self.image_paths.append(img_path)
                        self.captions.append(caption)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        return {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': img_path},
                                             {'type': 'text',
                                              'text': 'Describe the character’s position and the camera angle relative to the character. Include whether the shot is from the front, rear, left, or right side. Additionally, specify if the camera is at a high, low, or mid level, and mention whether it captures a close-up, mid shot, or full body view'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': caption}]}
            ]
        }
