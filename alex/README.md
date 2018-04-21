# nencog project

## sources maintained by alex


# CIFAR-100 selection

*	[select_cifar.py](./select_cifar.py) selects from CIFAR-100 the subset
 	of categories that will be used as useful for the context modeling

One should first install a local copy of the full [CIFAR100 dataset](http://www.cs.toronto.edu/~kriz/)
using the following command:

```
curl -O http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar zxvf cifar-100-python.tar.gz
```

The directory where the original dataset is installed should be set in the
global variable `cifar_100` in [select_cifar.py](./select_cifar.py).
