from torch.utils.data import Dataset, Sampler
from typing import Union, Callable, Optional
import random

#TODO Future
#Implement different sampling_factors for different classes.


class ImbalancedDatasetSampler(Sampler):
	''' ImbalancedDatasetSampler Class

	PyTorch Sampler class for dealing imbalanced datasets by either over- or under-sampling the classes. Can also interpolate between these modes.
	Labels can be provided via the 'labels' argument or via a callable object that is provided. The object can either return single int objects for every index in the dataset
	or a list of length len(dataset), with entries denoting the class of the datapoint at that index.

	Args:
		 dataset (Dataset): The dataset the Sampler should sample from
		 num_classes (int): The number of classes in the dataset
		 shuffle (bool) = False: If set to true every invocation of the iterator will shuffle the indices.
		 sampling_factor (float): Multi purpose value for letting you interpolate the size of the balanced dataset. The behaviour depends on the type of the variable. Type in [int,float].
		 	type(sampling_factor) == float:
		 		If 0.0 <= sampling_factor <= 1.0:  The size of each class will be linearly interpolated between the size of the smallest class and the largest class.
		 		If -1.0 < sampling_factor < 0.0: Corresponds to undersampling the largest class by a factor of -sampling_factor. Example: sampling_factor == -0.5 corresponds to undersampling the largest class by a factor of 1/2.
		 		If sampling_factor < 1.0: Corresponds to oversampling the smallest class by a factor of -sampling_factor. Example: sampling_factor == -2 corresponds to oversampling the smallest class by a factor of 2.
		 		If sampling_factor == -1.0 or sampling_factor > 1.0: Undefined behaviour. Will raise RuntimeExceptions.
		 	type(sampling_factor) == int:
		 		If sampling_factor >= 2: Set the size of each class to sampling_factor by over or undersampling
		 		Else: Raise Exception (to not collide with the float ranges if the wrong type is used)
		 	type(sampling_factor) == str:
		 		If sampling_factor == "oversampling": Equivalent to sampling_factor == 1.0
		 		If sampling_factor == "undersampling": Equivalent to sampling_factor == 0.0

		 labels ([list, list[list]): The labels of the dataset. Can be either a list-type object of length len(dataset), where the entry at position i, tells the class of the i-th dataset element. Alternativly, a list of lists of length num_classes is accepted, where each nested list contains the indices for the class.
		 callback_get_label (Callable): A callable that provides either the label at a single index (callsignature: callback_get_label(index)) or a list of labels (see previous argument). Mode of operation can be set by the callback_type argument.
		 callback_type (str) = "single": Choices: "single", "multi". See callback_get_label.
	'''

	def __init__(self, dataset: Dataset, num_classes: int, shuffle: bool = False,
				 sampling_factor: Union[str, float] = None, labels: Optional[Union[list, list[list]]] = None,
				 callback_get_label: Optional[Callable] = None, callback_type="single"):
		super(ImbalancedDatasetSampler).__init__()

		if sampling_factor is None:
			raise Exception("sampling_factor is None")

		if sampling_factor == "oversampling":
			sampling_factor = 1.0
		elif sampling_factor == "undersampling":
			sampling_factor = 0.0

		if labels is not None and callback_get_label is not None:
			raise RuntimeError("You cannot specify a callback if you provide the labels directly.")

		assert callback_type in ["single", "multi"]

		self.num_classes = num_classes

		# define custom callback
		self.callback_get_label = callback_get_label
		self.__shuffle = shuffle

		if labels is None:
			labels = []

			# If callback is specified use that function, if not use __getitem__. Return type should be [data_point, label]
			if callback_get_label is None:
				callback_get_label = lambda x: dataset.__getitem__(x)[1]
				callback_type = "single"

			if callback_type == "single":

				for i in range(len(dataset)):
					label = callback_get_label(i)
					labels.append(label)

			elif callback_type == "multi":
				labels = callback_get_label()

		self.labels = [[] for _ in range(self.num_classes)]

		# Labels can be eiter a list of len(dataset) or an object of [num_classes, *].
		if len(labels) == len(dataset):
			for i, label in enumerate(labels):
				self.labels[label].append(i)
		else:
			sum = 0
			assert len(labels) == self.num_classes
			for i, cl in enumerate(labels):
				self.labels[i] = cl[i]
				sum += len(self.labels[i])

			assert sum == len(dataset)

		num_labels = [len(i) for i in self.labels]
		min_size = min(num_labels)
		max_size = max(num_labels)

		if type(sampling_factor) is not int:
			#Linear interpolate the class size between the largest and smallest class
			if 0.0 <= sampling_factor <= 1.0:
				inter_class_distance = abs(max_size - min_size)
				self.class_size = min_size + int(sampling_factor * inter_class_distance)
			#Downsample the largest class by a factor
			elif -1.0 < sampling_factor < 0.0:
				self.class_size = int(max_size * -sampling_factor)
			#Upsample the smallest class by a factor
			elif sampling_factor < -1.0:
				self.class_size = int(min_size * -sampling_factor)
			else:
				raise NotImplementedError(
					"Called sampling factor with value -1.0 (behaviour undefined) or a float greater than 1.0")

		else:
			#Directly set the class size
			if sampling_factor == 1:
				raise NotImplementedError(
					"Undefined behaviour, when setting sampling_factor = int(1). Collides with float(1.0).")
			self.class_size = sampling_factor

		assert self.class_size <= max_size and self.class_size >= min_size

		self.__indices = self.__build_indices()

	def __build_indices(self):
		# Build dataset

		indices = []

		for c in range(self.num_classes):
			# Determine if we need to over- or undersample
			class_l = self.labels[c]
			num_in_class = len(class_l)

			# We have fewer actual labels than required. We need to oversample
			if num_in_class <= self.class_size:
				reps = self.class_size // num_in_class
				sample = self.class_size % num_in_class
				assert sample == self.class_size - reps * num_in_class
			# We need to undersample
			else:
				reps = 0
				sample = self.class_size

			new_ind = (class_l * reps) + random.sample(class_l, sample)
			indices += new_ind

		assert len(indices) == self.class_size * self.num_classes

		return indices

	def __iter__(self):

		if self.__shuffle:
			random.shuffle(self.__indices)

		return iter(self.__indices)

	def __len__(self):
		return self.class_size * self.num_classes
