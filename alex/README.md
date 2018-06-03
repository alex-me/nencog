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

*	[cnn_cifar.py](./cnn_cifar.py) has the same functionalities as
 	[scnn_cifar.py](./scnn_cifar.py) but does not need the Slim library,
	the neural network is build using TensorFlow primitives, organized in
	[cnn.py](./cnn.py)

The reason for this slim-free code is that a CIFAR classifier that uses only
TensorFlow primitives may be more straightforward to be embedded in nengo
models.

The parametes defining the neural architecture, collected in the dictionary
`nn_arch`, can be imported from the external file [neural_arch.py](./neural_arch.py),
so to make easier experimenting with different parameters.

The global variables `model_dir` and `model_name` defines the directory Cand the
names checkpoints.

## Integrating TensorFlow and nengo

*	[spa_scnn.py](./spa_scnn.py) is a simple nengo model that integrates
	scnn_cifar.py connecting its logits to SPA nengo nodes
 	for a classifer on the personalized subset of CIFAR-100 images

Note that this stage has been troublesome, and standard TensorFlow graphs
gneretate many errors whtn cast as `nengo_dl.TensorNode`, at the end only the
slim model [scnn_cifar.py](./scnn_cifar.py), with several adjustments, was
imported without errors.
