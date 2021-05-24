import os
import numpy as np
import cv2 as cv
import json

class SOBADatasetManager:
    def __init__(self, path_to_dataset):
        self.dataset_path = path_to_dataset
        self.annotations_path = os.path.join(path_to_dataset, 'annotations/SOBA_train.json') 

        with open(self.annotations_path, 'r') as f:
            self._annotation_data = json.load(f)

        self.image_no = len(self._annotation_data['images'])

    def get_img(self, ind):
        if ind < 0 or ind >= self.image_no:
            return None

        img_path = os.path.join(self.dataset_path, 'SOBA', self._annotation_data['images'][ind]['file_name'])
        return cv.imread(img_path)

    def get_imgs(self, indx):
        return [self.get_img(ind) for ind in indx]

    def get_rand_img(self):
        ind = np.random.randint(0, self.image_no)
        return self.get_img(ind)

    def sample_images(self, sample_no, deterministic=False):
        if deterministic:
            np.random.seed(1)
        choices = np.random.choice(self.image_no, sample_no)
        return [self.get_img(choice) for choice in choices]