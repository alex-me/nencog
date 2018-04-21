# nencog project

## sources maintained by alex


## CIFAR-100 selection

*	[select_cifar.py](./select_cifar.py) selects from CIFAR-100 the subset
 	of categories that will be used as useful for the context modeling

One should first install a local copy of the full [CIFAR100 dataset](http://www.cs.toronto.edu/~kriz/)
using the following command:

```shell
curl -O http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar zxvf cifar-100-python.tar.gz
```

The directory where the original dataset is installed should be set in the
global variable `cifar_100` in [select_cifar.py](./select_cifar.py).

This script can be executed in order to generate the subset:

```shell
python select_cifar.py
```

or imported as a module in other programs:

```python
import select_cifar

# import the CIFAR-100 selected categories
classes   = select_cifar.classes

# import paths to the personalized CIFAR dataset
path      = select_cifar.cifar_sel
```

## Classification with TensorFlow

*	[scnn_cifar.py](./scnn_cifar.py) is a simple convolutional architecture
 	for a classifer on the personalized subset of CIFAR-100 images

The neural architecture is defined using
[TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim),
the lightweight library for quick definition and training of models in TensorFlow

The global variable `model_dir` defines the directory where checkpoints are
stored and retrieved.

The program can be used for training, evaluating, and making predictions on
single images.
