# nencog project

## sources maintained by alex


## utilities for managing the LAVA dataset

### Lava Corpus
Language and Vision Ambiguities (LAVA) is a multimodal corpus that supports the
study of ambiguous language grounded in vision, see Yevgeni Berzak, Andrei
Barbu, Daniel Harari, Boris Katz and Shimon Ullman (2015). *Do You See What I
Mean? Visual Resolution of Linguistic Ambiguities* for reference

*	[labeled_images.py](./labeled_images.py) is used to list the images with
 	labels from the original json file of the dataset

### Sample extraction
All useful images hase been manually annotated for the presence of significant
objects, together with their embedding box, data are stored in
[coords.txt](./coords/coords.txt)

*	[put_boxes.py](./put_boxes.py) scan all images annotated in
[coords.txt](./coords/coords.txt) and generated the corresponding images with
boxes in overlay

*	[coord_stat.py](./coord_stat.py) produces a statistics of the boxes
embedding objects in the vairous categories

*	[extract_samples.py](./extract_samples.py) extract all samples

*	[extract_no_samples.py](./extract_no_samples.py) extract samples that do
not contain objects of the relevant categories, to be used by the neural
segmentation

### Generation of datasets
The samples extracted are then organized into datasets, ready to be used for
training network models on the categories of objects of interest.

*	[pack_lava.py](./pack_lava.py) pack samples into datasets - note that
less samples than those available are packed into datasets, due to limitations
in the pickle protocols, and in the critical usage of RAM.
The datasets are stored in the directory [data](./data/).


## CNN models for the identification of relevant objects

*	[cnn.py](./cnn.py) is the library of TensorFlow primitives for building
models, that is a copy of [cnn.py](../my_cifar/cnn.py) already used in the
experiments with [a CIFAR-100 selection](../my_cifar/README.md), adapted for the
use with LAVA images, and for the task of searching objects.

Since TF models built with [cnn.py](../my_cifar/cnn.py) were succesfully
integrated in Nengo, there is hope that the integration for the LAVA models will
be smooth.

*	[cnn_lava.py](./cnn_lava.py) is a higher level of NN modeling that uses
[cnn.py](./cnn.py), customized for LAVA objects and datasets, and allows for
the simujltaneous use of multiple TF graphs and sessions, each for a catagory of
LAVA objects. The program includes functions to perform training, testing,
single image recall, batch recall. 
The trained models are stored in the directory [model_dir](./model_dir/).

*	[cnn_lava.py](./search_lava.py) performs object searching on full LAVA
images for a single object or for multiple objects, calling functions of
[cnn_lava.py](./cnn_lava.py), and returns list with the most probable boxes
embedding the objects. It can also generate images annotated with the boxes
found in overlay, using the functions of [put_boxes.py](./put_boxes.py).



