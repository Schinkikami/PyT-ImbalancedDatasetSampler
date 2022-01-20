# PyT-ImbalancedDatasetSampler

#Introduction

An imbalanced dataset sampler for PyTorch.
This an alternative for ufoym's [ImbalancedDatasetSampler](https://github.com/ufoym/imbalanced-dataset-sampler).

It does not randomly draw elements by a multinomial distribution, which may unwantetly draw some elements multiple times
while skipping over other elements, but instead wraps the dataset around when oversampling.

Additionally, a finer control over the over- and undersampling procedure is allowed, by manually setting the number of data-points per-class
or specifing per-class over- and undersampling factors.
We do not require external packages other than pyTorch.

For a nice explanation with great visuals of over- and undersampling visit [ufomy's](https://github.com/ufoym/imbalanced-dataset-sampler) github project. 

#Description

Create an instance of the `ImbalancedDatasetSampler` and pass it to the pyTorch `DataLoader`


```python
from imbalanced_sampler import ImbalancedDatasetSampler

sampler = ImbalancedDatasetSampler(dataset=dataset, sampling_factor=s_f, shuffle=True, ....)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=sampler,
    ... )
```
The `ImbalancedDatasetSampler` takes multiple arguments allowing for different behaviour (also take a look at the docstring):
- dataset:torch.Dataset - the dataset
- sampling_factor:[str,float,int] - Sets the degree of over- or under-sampling for the classes. If set to an int, each class will be scaled to that many data points. If set to a float between 0 and 1, the class size will be interpolated between the smallest and largest class. A float between -1.0 and 0.0 will undersample the largest class to class_size(largest)*-sampling_factor. A value smaller than -1.0 will oversample the smallest class by a factor of -sampling_factor. All classes will always have the same size. 
- num_classes: Number of classes. If set to None will be automatically determined.
- shuffle: If set to true, will shuffle the dataset at each epoch. Otherwise will always return the same order. However, if a class is undersampled, it is still non-deterministc with each iterator instanciation.
- labels: You can pass the labels directly here.
- callback_get_label: A `Callable` that will be called to generate the labels. Will default to `lambda x: __getitem__(idx)[1]` on the dataset if not provided.
- callback_type: If set to "single" the function will be called with the index of the dataset `callback_get_label(idx)`. If set to multi, no arguments will be passed and it is expected to return a list of labels with length `len(dataset)`.


#How to install
You can directly install the `imbalanced_sampler` package from the provided `setup.py` file

```pip install .```

Alternativly install from the provided `tar.gz` file

```pip install dist/imbalanced_sampler-0.1.tar.gz```

Finally, you could just copy the `ImbalancedDatasetSampler.py` to your projects location.

#Future

I want to add support for manually setting the classes to different sizes.
