import os
import random
import glob
from PIL import Image
import numpy as np

import torch
from torch.utils import data
import torch.nn.functional as F


class OpenSetDataset(data.Dataset):
	def __init__(self, data_path, class_list, transforms=None):
		self.data_path = data_path
		self.class_list = class_list
		self.transforms = transforms
		self.img_details = []
		for c_idx, c in enumerate(self.class_list):
			imgs = glob.glob(os.path.join(self.data_path, c, "*"))
			for img in imgs:
				self.img_details.append([img, c_idx])
		imgs = glob.glob(os.path.join(self.data_path, "unknown_class", "*"))
		for img in imgs:
			self.img_details.append([img, -1])
		random.shuffle(self.img_details)

	def __len__(self):
		return len(self.img_details)


	def __getitem__(self, index):
		image = Image.open(self.img_details[index][0])
		image = np.array(image)[:,:,:3]
		image = Image.fromarray(image)
		label = np.zeros(len(self.class_list))
		flag = np.zeros(2)
		if self.img_details[index][1] > 0:
			label[self.img_details[index][1]] = 1
			flag[0] = 1
		else:
			label = label * (1/len(self.class_list))
			flag[1] = 1

		if self.transforms is not None:
			return self.transforms(image), label, flag
		return image, label, flag

