##############################################################################
# Some objects to manage the different datasets used
# Authored by Ammar Mian, 24/06/2019
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2019 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import numpy as np
import scipy.io

class dataset_class(object):
	"""Contain all relevant information and data of a dataset"""
	def __init__(self, name, data, ground_truth, figsize):
		self.name = name
		self.data = data
		self.ground_truth = ground_truth
		self.figsize = figsize
		self.number_rows = data.shape[0]
		self.number_columns = data.shape[1]
		self.number_canals = data.shape[2]
		self.number_dates = data.shape[3]

def fetch_dataset(name, path_to_data):
	""" A function to fetch dataset by name."""

	if name == "SDMS FP0120_FP0121_FP0124":
		data = scipy.io.loadmat(path_to_data + 'SDMS/FP0120_FP0121_FP0124.mat')['image_time_series']
		ground_truth = scipy.io.loadmat(path_to_data + 'SDMS/ground_truth_FP0120_FP0121_FP0124.mat')['ground_truth']
		figsize = (5,5)

	elif name == "SDMS FP0121_FP0124":
		data = scipy.io.loadmat(path_to_data + 'SDMS/FP0120_FP0121_FP0124.mat')['image_time_series'][:,:,:,1:]
		ground_truth = scipy.io.loadmat(path_to_data + 'SDMS/ground_truth_FP0121_FP0124.mat')['ground_truth']
		figsize = (5,5)

	elif name == "UAVSAR Scene 1":
		data = np.load(path_to_data + 'UAVSAR/Scene_1.npy')
		ground_truth = np.load(path_to_data + 'UAVSAR/ground_truth_scene_1.npy')
		figsize = (3.5,5)

	elif name == "UAVSAR Scene 2":
		data = np.load(path_to_data + 'UAVSAR/Scene_2.npy')
		ground_truth = np.load(path_to_data + 'UAVSAR/ground_truth_scene_2.npy')
		figsize = (3.5, 5)

	elif name == "UAVSAR Scene 3":
		data = np.load(path_to_data + 'UAVSAR/Scene_3.npy')
		ground_truth = np.load(path_to_data + 'UAVSAR/ground_truth_scene_3.npy')
		ground_truth = np.sum(ground_truth, axis=2)!=0
		figsize = (5,3)

	else :
		print("Error: dataset %s is not recognized" % name)
		print("Choice is among: [SDMS FP0120_FP0121_FP0124, SDMS FP0121_FP0124, UAVSAR Scene 1, UAVSAR Scene 2, UAVSAR Scene 3]")
		return None

	return dataset_class(name, data, ground_truth, figsize)
