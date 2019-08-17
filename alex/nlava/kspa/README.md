# nencog project

## sources maintained by alex


## Nengo-Keras integrated models for disambiguation of sentences in the LAVA corpus

In this folder complete models integrating Nengo and [Keras](https://keras.io/)
with the package
[nengo_dl](https://www.nengo.ai/nengo-dl/) have been finalized to disambiguate
the sentences in the LAVA corpus ([see][../README.md].

There is no need to access the scripts used to define and train [the Keras
architectures][../keras/README.md] for the image processing of LAVA images, all
what is needed is to create a symbolic link to the folder with the trained
models:

```shell
ln -s ../keras/model_dir
```

*	[kspa_model.py](./kspa_model.py) is the main module with Nengo
primitives for building and executing the integrated model.
The Nengo model, defined with the class `Nn`, instantiate special nodes of the
class `Kinter` that interface the Keras models through `nengo_dl.TensorNode`.
There are as many `Kinter` nodes as the different categories of objects to be
found in a LAVA scene. Results of the localization of objects are fed to Nengo
`spa.State` Semantic Pointers. The closeness of localizations of the
categories that play a syntactic role in the LAVA sentence, embedded in the
`spa.State` Semantic Pointers, is the key for guessing the correct syntactic
representation of the ambiguous sentence.
The class `Nn` comprises two separated Nengo models: a `nengo.Network` with
`nengo_dl.TensorNode` collecting results of the Keras models, and a
`spa.Network` using SPA states, for cdisambiguating the sentences based on
visual processing.

*	[lava_geo.py](./lava_geo.py) is a module, need by
[kspa_model.py](./kspa_model.py), providing all the necessary information about
LAVA image geometries, especially for the purpose of scanning the image in
subwindows of the proper size for the different categories of objects. It is
derived from [search_lava.py](../search_lava.py), but specialized for the
requirements of [kspa_model.py](./kspa_model.py).

*	[display_lava.py](./display_lava.py) is a utility for displaying the
probability function for an object in a LAVA image, as detected by the Keras
models. It uses the same reduction of probabilities from the two dimensions of
the image to the horizontal dimesions as in [kspa_model.py](./kspa_model.py).

*	[exec_kspa.py](./exec_kspa.py) is the main script, that executes the
Nengo model defined in [kspa_model.py](./kspa_model.py)

Its usage can be obtained with:

```shell
python exec_kspa.py -h
```

*	[exec_keras.py](exec_keras.py) is a script that executes the Keras part
of the modle, computing probabilities for all LAVA images relevant for a list of
sentences, and storing the results in files. This is a sort of batch execution
of the most time consuming conputation, which is invariant with respect to the
evolution of SPA states in the full Nengo model.
By executing [exec_kspa.py](./exec_kspa.py) with the flag `-b` only the SPA
model will be executed, and the probabilities of object presence derived by
Keras models, will be read from files.


*Note:* there was a previous disambiguation script [kn.py](../kn.py), with the
purpose of debuggind the Nengo-Keras interface, but the Nengo part was
essential, and the disambiguation done as post-processing in a non neural way.
