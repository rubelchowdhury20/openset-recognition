# Standard library imports
import os
import json
import math
import random

# Third party imports
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torchsummary import summary

# Local imports
import config
import dataloader
import utils

DEVICE = config.DEVICE

CURRENT_FREEZE_EPOCH = 0
CURRENT_UNFREEZE_EPOCH = 0
BEST_LOSS = 4													


# simple cross entropy cost
def xentropy_cost(target, pred):
	assert target.size() == pred.size(), "size fail ! "+str(target.size()) + " " + str(pred.size())
	logged_pred = torch.log(pred)
	cost_value = -torch.sum(target * logged_pred)/pred.shape[0]
	return cost_value

# objectosphere loss
def ring_loss(y_flag, embeddings):
	embeddings_sum = torch.sqrt(torch.sum(torch.square(embeddings), axis=1))
	print(embeddings_sum)
	error = torch.sqrt(torch.mean(torch.square(
			# loss for knowns having magnitude greater than knownsMinimumMag
			y_flag[:, 0] * torch.maximum(torch.ones(embeddings.shape[0], 1).to(device=DEVICE) * config.ARGS.knownsMinimumMag - embeddings_sum, torch.zeros_like(embeddings_sum))
			+
			# loss for unknows having magnitude greater than unknownsMaximumMag
			y_flag[:, 1] * embeddings_sum)))
	return error

def training(args):
	# declaring global variables
	global BEST_LOSS
	global CURRENT_FREEZE_EPOCH
	global CURRENT_UNFREEZE_EPOCH

	# steps for preparing and splitting the data for training
	data_path = config.ARGS.data_directory
	class_list = [os.path.basename(i) for i in glob.glob(os.path.join(data_path, "train", "*"))]
	class_list.remove("unknown_class")
	class_list.sort()
	train_dataset = dataloader.OpenSetDataset(data_path=os.path.join(data_path, "train"), class_list=class_list, transforms=config.data_transforms["train"])
	train_loader = data.DataLoader(train_dataset, **config.PARAMS)
	val_dataset = dataloader.OpenSetDataset(data_path=os.path.join(data_path, "val"), class_list=class_list, transforms=config.data_transforms["val"])
	val_loader = data.DataLoader(val_dataset, **config.PARAMS)

	# loading the pretrained model and changing the dense layer. Initially the convolution layers will be freezed
	base_model = models.resnet18(pretrained=True).to(DEVICE)
	for param in base_model.parameters():
		param.requires_grad = False
	num_ftrs = base_model.fc.in_features
	base_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.Linear(256, 128), nn.Linear(128, len(class_list)), nn.Softmax(dim=1))
	base_model = base_model.to(DEVICE)

	# registering a forward hook to extract features
	feature = {}
	def get_activation(name):
		def hook(model, input, output):
			feature[name] = output.detach()
		return hook

	base_model.fc[1].register_forward_hook(get_activation("embeddings"))

	# print(summary(base_model, (3, 224, 224)))

	if(config.ARGS.pretrained_weights):
		try:
			_, _, BEST_LOSS, base_model = utils.load_checkpoint(config.ARGS.pretrained_weights, base_model)
		except:
			print("Not able to load from the pretrained_weights")
	elif(config.ARGS.resume):
		try:
			CURRENT_FREEZE_EPOCH, CURRENT_UNFREEZE_EPOCH, BEST_LOSS, base_model = utils.load_checkpoint(os.path.join(config.ARGS.weights_directory, config.ARGS.checkpoint_name), base_model)
		except:
			print("not able to load checkpoint because of non-availability")

	# Initializing the loss function and optimizer
	optimizer = optim.SGD(base_model.parameters(), lr=config.LR, momentum=config.MOMENTUM)

	# # Decay LR by a factor of 0.1 every 7 epochs
	# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	# Printing total number of parameters
	n_parameters = sum([p.data.nelement() for p in base_model.parameters()])
	print('  + Number of params: {}'.format(n_parameters))

	entropy_loss_meter_train = utils.AverageMeter()
	ring_loss_meter_train = utils.AverageMeter()
	total_loss_meter_train = utils.AverageMeter()
	entropy_loss_meter_val = utils.AverageMeter()
	ring_loss_meter_val = utils.AverageMeter()
	total_loss_meter_val = utils.AverageMeter()

	# training for the first few iteration of freeze layers where apart from the dense layers all the other layres are frozen
	if(CURRENT_FREEZE_EPOCH < config.FREEZE_EPOCHS):
		for epoch in range(CURRENT_FREEZE_EPOCH + 1, config.FREEZE_EPOCHS + 1):
			entropy_loss_meter_train.reset()
			ring_loss_meter_train.reset()
			total_loss_meter_train.reset()
			entropy_loss_meter_val.reset()
			ring_loss_meter_val.reset()
			total_loss_meter_val.reset()
			base_model.train()
			for batch_index, (batch_imgs, batch_labels, batch_flags) in tqdm(enumerate(train_loader)):
				batch_imgs = batch_imgs.to(DEVICE)
				batch_labels = batch_labels.to(DEVICE)
				batch_flags = batch_flags.to(DEVICE)

				output = base_model(batch_imgs)
				embeddings = feature["embeddings"]

				entropy_loss = xentropy_cost(batch_labels, output)
				objectosphere_loss = ring_loss(batch_flags, embeddings)

				total_loss = config.ARGS.cross_entropy_loss_weight * entropy_loss + config.ARGS.ring_loss_weight * objectosphere_loss

				entropy_loss_meter_train.update(entropy_loss)
				ring_loss_meter_train.update(objectosphere_loss)
				total_loss_meter_train.update(total_loss)

				# compute gradient and do optimizer step
				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()

				if batch_index % 10 == 0:
					print("Train Progress(frozen)--\t"
						"Epoch: {} [{}/{}]\t"
						"Total loss:{:.4f} ({:.4f})\t"
						"Entropy loss:{:.4f} ({:.4f})\t"
						"Objectosphere loss:{:.4f} ({:.4f})".format(epoch, batch_index * config.PARAMS["batch_size"], len(train_loader.dataset),
						 											total_loss_meter_train.val, total_loss_meter_train.avg,
						 											entropy_loss_meter_train.val, entropy_loss_meter_train.avg,
						 											ring_loss_meter_train.val, ring_loss_meter_train.avg))


			base_model.eval()
			with torch.no_grad():
				for batch_index, (batch_imgs, batch_labels, batch_flags) in tqdm(enumerate(val_loader)):
					batch_imgs = batch_imgs.to(DEVICE)
					batch_labels = batch_labels.to(DEVICE)
					batch_flags = batch_flags.to(DEVICE)

					output = base_model(batch_imgs)
					embeddings = feature["embeddings"]

					entropy_loss = xentropy_cost(batch_labels, output)
					objectosphere_loss = ring_loss(batch_flags, embeddings)

					total_loss = config.ARGS.cross_entropy_loss_weight * entropy_loss + config.ARGS.ring_loss_weight * objectosphere_loss

					entropy_loss_meter_val.update(entropy_loss)
					ring_loss_meter_val.update(objectosphere_loss)
					total_loss_meter_val.update(total_loss)

					if batch_index % 10 == 0:
						print("Validation Progress(frozen)--\t"
							"Epoch: {} [{}/{}]\t"
							"Total loss:{:.4f} ({:.4f})\t"
							"Entropy loss:{:.4f} ({:.4f})\t"
							"Objectosphere loss:{:.4f} ({:.4f})".format(epoch, batch_index * config.PARAMS["batch_size"], len(val_loader.dataset),
							 											total_loss_meter_val.val, total_loss_meter_val.avg,
							 											entropy_loss_meter_val.val, entropy_loss_meter_val.avg,
							 											ring_loss_meter_val.val, ring_loss_meter_val.avg))


			# remember best loss and save checkpoint
			is_best = total_loss_meter_val.avg < BEST_LOSS
			BEST_LOSS = min(total_loss_meter_val.avg, BEST_LOSS)
			utils.save_checkpoint({
				'current_freeze_epoch': epoch,
				'current_unfreeze_epoch': 0,
				'state_dict': base_model.state_dict(),
				'best_loss': BEST_LOSS,
			}, is_best, epoch)
			CURRENT_FREEZE_EPOCH = epoch


	# Unfreezing the last few convolution layers
	for param in base_model.parameters():
		param.requires_grad = True
	ct = 0
	for name, child in base_model.named_children():
		ct += 1
		if ct < 7:
			for name2, parameters in child.named_parameters():
				parameters.requires_grad = False

	# training the remaining iterations with the last few layers unfrozen
	if(CURRENT_UNFREEZE_EPOCH < config.UNFREEZE_EPOCHS):
		for epoch in range(CURRENT_UNFREEZE_EPOCH + 1, config.UNFREEZE_EPOCHS + 1):
			entropy_loss_meter_train.reset()
			ring_loss_meter_train.reset()
			total_loss_meter_train.reset()
			entropy_loss_meter_val.reset()
			ring_loss_meter_val.reset()
			total_loss_meter_val.reset()
			base_model.train()
			for batch_index, (batch_imgs, batch_labels, batch_flags) in tqdm(enumerate(train_loader)):
				batch_imgs = batch_imgs.to(DEVICE)
				batch_labels = batch_labels.to(DEVICE)
				batch_flags = batch_flags.to(DEVICE)

				output = base_model(batch_imgs)
				embeddings = feature["embeddings"]

				entropy_loss = xentropy_cost(batch_labels, output)
				objectosphere_loss = ring_loss(batch_flags, embeddings)

				total_loss = config.ARGS.cross_entropy_loss_weight * entropy_loss + config.ARGS.ring_loss_weight * objectosphere_loss

				entropy_loss_meter_train.update(entropy_loss)
				ring_loss_meter_train.update(objectosphere_loss)
				total_loss_meter_train.update(total_loss)

				# compute gradient and do optimizer step
				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()

				if batch_index % 10 == 0:
					print("Train Progress(unfrozen)--\t"
						"Epoch: {} [{}/{}]\t"
						"Total loss:{:.4f} ({:.4f})\t"
						"Entropy loss:{:.4f} ({:.4f})\t"
						"Objectosphere loss:{:.4f} ({:.4f})".format(epoch, batch_index * config.PARAMS["batch_size"], len(train_loader.dataset),
						 											total_loss_meter_train.val, total_loss_meter_train.avg,
						 											entropy_loss_meter_train.val, entropy_loss_meter_train.avg,
						 											ring_loss_meter_train.val, ring_loss_meter_train.avg))


			base_model.eval()
			with torch.no_grad():
				for batch_index, (batch_imgs, batch_labels, batch_flags) in tqdm(enumerate(val_loader)):
					batch_imgs = batch_imgs.to(DEVICE)
					batch_labels = batch_labels.to(DEVICE)
					batch_flags = batch_flags.to(DEVICE)

					output = base_model(batch_imgs)
					embeddings = feature["embeddings"]

					entropy_loss = xentropy_cost(batch_labels, output)
					objectosphere_loss = ring_loss(batch_flags, embeddings)

					total_loss = config.ARGS.cross_entropy_loss_weight * entropy_loss + config.ARGS.ring_loss_weight * objectosphere_loss

					entropy_loss_meter_val.update(entropy_loss)
					ring_loss_meter_val.update(objectosphere_loss)
					total_loss_meter_val.update(total_loss)


					if batch_index % 10 == 0:
						print("Validation Progress(unfrozen)--\t"
							"Epoch: {} [{}/{}]\t"
							"Total loss:{:.4f} ({:.4f})\t"
							"Entropy loss:{:.4f} ({:.4f})\t"
							"Objectosphere loss:{:.4f} ({:.4f})".format(epoch, batch_index * config.PARAMS["batch_size"], len(val_loader.dataset),
							 											total_loss_meter_val.val, total_loss_meter_val.avg,
							 											entropy_loss_meter_val.val, entropy_loss_meter_val.avg,
							 											ring_loss_meter_val.val, ring_loss_meter_val.avg))



			# remember best loss and save checkpoint
			is_best = total_loss_meter_val.avg < BEST_LOSS
			BEST_ACC = min(total_loss_meter_val.avg, BEST_LOSS)
			utils.save_checkpoint({
				'current_freeze_epoch': CURRENT_FREEZE_EPOCH,
				'current_unfreeze_epoch': epoch,
				'state_dict': base_model.state_dict(),
				'best_loss': BEST_LOSS,
			}, is_best, CURRENT_FREEZE_EPOCH+epoch)
			CURRENT_UNFREEZE_EPOCH = epoch


