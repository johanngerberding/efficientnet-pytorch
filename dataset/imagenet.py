import os 
import glob 
import cv2 
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class ImageNet(Dataset):
    def __init__(self, imgs_root, annotations_root, label_file, is_train, transform):
        self.imgs_root = imgs_root
        self.annotations_root = annotations_root
        self.is_train = is_train
        self.transform = transform
        self.id_to_label = self.read_labels(label_file)
        self.annotations =  self.create_annos()

    def __getitem__(self, idx):
        label = self.annotations[idx]['label']
        
        img = cv2.imread(self.annotations[idx]['img_path'], cv2.IMREAD_COLOR)
        label = torch.tensor(label)

        if self.transform: 
            img = self.transform(image=img)["image"]
        
        return img, label

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def xml_to_label(filepath: str) -> str:
        "Extract classification label from xml"
        tree = ET.parse(filepath)
        root = tree.getroot()
        res = root.findall('object')
        name = res[0].find('name').text

        return name

    @staticmethod
    def read_labels(filepath: str) -> dict:
        "Create a dict mapping for class labels"
        with open(filepath, 'r') as mapping:
            labels = mapping.readlines()
        id_to_labels = {label: i for i, label in enumerate(labels)}
        
        return id_to_labels

    def get_imgs(self) -> list:
        "Create list of img paths"
        imgs = []
        if self.is_train:
            for dir in os.listdir(self.imgs_root):
                p = os.path.join(self.imgs_root, dir)
                imgs_list = glob.glob(p + "/*.JPEG")
                
                imgs += imgs_list
        else:
            imgs = glob.glob(self.imgs_root + "/*.JPEG")
        
        return imgs
    
    def get_annotations(self) -> list:
        annos = []
        if not self.is_train:
            annos = glob.glob(self.annotations_root + "/*.xml")
        return annos 
    
    def create_annos(self) -> dict:
        "Create dict with annotations"
        imgs = self.get_imgs()
        annotations = dict()
        if not self.is_train:
            annos = self.get_annotations()
            for i, img in enumerate(imgs):
                id = os.path.split(img)[1][:-4]
                ann = [an for an in annos if id in an][0]
                label = self.xml_to_label(ann)
                annotations[i] = {
                    'img_path': img,
                    'label': [v for k,v in self.id_to_label.items() if label in k][0]
                }
        else: 
            for j, im in enumerate(imgs):
                _, filename = os.path.split(im)
                label = filename.split("_")[0]
                annotations[j] = {
                    'img_path': im,
                    'label': [v for k,v in self.id_to_label.items() if label in k][0]
                }
        
        return annotations