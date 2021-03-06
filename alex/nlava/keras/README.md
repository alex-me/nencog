# nencog project

## sources maintained by alex


## Keras models for the identification of LAVA relevant objects

Since version 2.2.0 of [nengo_dl](https://www.nengo.ai/nengo-dl/) Nengo can
accept [Keras](https://keras.io/) models as `TensorNode` nodes, and since Keras
is the most compact and easy format for TensorFlow models, it is worth moving to
this solution, abandoning the previous [TensorFlow based convolutional
models](../README.md).

*	[arch.py](./arch.py) is the main library of Keras primitives for
building models that can be specified using [simple configuration
files](config/README.md). Models are made conbining arbitrary sequences of four
types of Keras layers: convolutions, dense, pooling, and flattening layers.
The last layer of the models is alwyas a binary classificator, since we are
interesting in searching for specifc predefined objects inside a LAVA image.

*	[trainer.py](./trainer.py) is the package that take care of running the
training of a model defined by [arch.py](./arch.py) saving the best weights in a
[HDF5 binary](https://www.hdfgroup.org/) file format.

*	[cnfg.py](./cnfg.py) is a small modul that handles [configuration files](config/README.md).

*	[mesg.py](./mesg.py) handles messages, allowing redirection of `stdout`,
useful when training on a remote GPU machine.

*	[pack_lava.py](../pack_lava.py) is part of the [LAVA utilities](../README.md)
and is used by [trainer.py](./trainer.py) and [exec_main.py](exec_main.py) for
setting specifications about the LAVA datasets for each object category.
It is necessary to create a symbolic link to this module:

```shell
ln -s ../pack_lava.py
```

*	[exec_main.py](exec_main.py) is the main for training and saving the
Keras models of LAVA objects. Its usage can be obtained with:

```shell
python exec_main.py -h
```

A typical execution uses the following arguments:

```shell
python exec_main.py -c config/BAG -g '0,' -f '0.5' -Tssr
```

Where `-c config/BAG` tells the program to build a model based on the
configuration file `config/BAG` (which obviously refers to the object BAG), 
`-g '0,'` instruct TensorFlow to use the first available GPU, and `-f '0.5'`
specifies to use up to half of the GPU available memory. Note that GPU
specification is a tuple (for example with `-g '2,3,4,'` you will use multiple
GPUs), if you specify the number `0` (i.e. `-g 0`) the program will not attempt
to use any GPU and train on CPU instead. The flags `-Tssr` instruct to perform
the training, to save along the model also the configuration file and all the
sources, and to write messages on a log file, redirecting `stdout`.

The trained model, along with the additional information on the model, are
stored in the folder `res` and in a unique subfolder named by the time stamp.


## generation of datasets

*	[do_ln.py](do_ln.py) is a script that generates symbolik links to the
image samples, that should be already generated (see [..](../README.md)). The
links are organized in folder trees compliant with Keras
[flow_from_directory](https://keras.io/preprocessing/image/) method, used in
[trainer.py](./trainer.py).

*	NOTE that there is the alternative to use a single file to be loaded in
memory, generated by [pack_lava.py](../pack_lava.py), see [..](../README.md).

*	the choice between using
[flow_from_directory](https://keras.io/preprocessing/image/) and training from
disk, or training in memory when executing [exec_main.py](exec_main.py), is
automatic and depends from the pathname specified in the `dir_dset` field of the
[configuration file](config/README.md).

If the pathname has extension `.gz` it is assumed as a single file containing
the dataset to be read in memory, otherwise it should be the root of the
dirctory tree valid for training. The default root in [do_ln.py](do_ln.py) ha
name `link`, therefore tor traning from disk for the `bag` category, the correct
entry in the [configuration file](config/README.md) should be the following:

```python
dir_dset                "link/bag"
```


## integrated Keras-Nengo models for the LAVA diambiguation task

*	[kn.py](kn.py) is the first working integration of the models in Keras
defined and trained with [exec_main.py](exec_main.py).
The program expect to find the necessary Keras models in the folder
[./model_dir](./model_dir/README.md).

The Nengo model, defined with the class `Nn`, instantiate special nodes of the
class `Kinter` that interface the Keras models through `nengo_dl.TensorNode`.
There are as many `Kinter` nodes as the different categories of objects to be
found in a LAVA scene.

There are two main exectuive functions:
*	`recall_sentence()` read a sentence, and
recall the nengo/keras model on all subwindows of the relevant image, for the
categories of objects mentioned in the sentence, and returns the list of
coordinates where the objects have been found with highest probability.
*	`disambiguate()` read also a sentence and triggers the nengo/keras model,
in addition to `recall_sentence()` it process the coordinates of the hits for
guessing the correct syntax. Note that this final part is _not_ a neural
process.

*Note:* there is currently no working interface to Keras in the stable version
of nengo-dl, it is necessary to download and install the development version
`2.2.0.dev0`

