# FastSparse
> Customizable Fastai+PyTorch implementation of sparse model training methods (SET, SNFS, RigL).


## _Warning: this repo is undergoing active development_

TODOs:
 - test dynamic training callback
 - finish documenting this page
  - PyTorch example
  - under-the-hood explanation
  - fully custom example
  - Drop/Redist/Grow criterion
 - implement distributed training (?)

## Install

`pip install fastsparse`

## How to use

### Fastai example

With this package, you can train your model using the latest dynamic sparse training techniques. It only takes 4 additional lines of code!

```python
from fastai.vision.all import *
from fastsparse.core import *                            # <-- import this package

path = untar_data(URLs.MNIST)
dls = ImageDataLoaders.from_folder(path, 'training', 'testing')
learn = cnn_learner(dls, resnet34, metrics=error_rate, pretrained=False)
sparse_hooks = sparsify_model(learn.model, sparsity=0.9) # <-- initial sparsity + enforce masks
cbs = DynamicSparseTrainingCallback(**RigL_kwargs)       # <-- dynamic mask updates

learn.fit_one_cycle(1, cbs=cbs)

for h in sparse_hooks: h.remove()                        # <-- stop enforcing masks
```

Simply omit the `DynamicSparseTrainingCallback` to train a fixed-sparsity model as a baseline.

### PyTorch example

TODO

### Training with Large Batch Sizes

Authors of the Rigged Lottery paper hypothesize that the effectiveness of using the gradient magnitude for determining which connections to grow is partly due to their large batch size (4096 for ImageNet). Those without access to multi-gpu clusters can achieve effective batch sizes of this size by using fastai's `GradientAccumulation` callback, which has been tested to be compatible with this package's `DynamicSparseTrainingCallback`.

## Under-The-Hood

Here's what's going on. 

When you run `sparsify_model(learn.model, 0.9)`, this adds sparse masks and add pre_forward hooks to enforce masks on weights during forward pass.

> By default, a uniform sparsity distribution is used. Change the sparsity distribution to Erdos-Renyi with `sparsify_model(learn.model, 0.9, sparse_init_f=erdos_renyi)`, or pass in your custom function (see [Customization](#Customization)

> To avoid adding pre_forward hooks, use `sparsify_model(learn.model, 0.9, enforce_masks=False)`.

When you add the `DynamicSparseTrainingCallback` callback, ... TODO complete section

## Customization

There are several places to modify the behavior of fastsparse to accomplish custom behaviors. For an example, check out the implementation of RigL.

### 1. Initial sparsity distribution:

Define your own initial sparsity distribution by setting `sparsify_method` in `sparsify_model` to a custom function. For example, this function (included in library) makes the first layer dense, and all following layers to a fixed sparsity.

```python
def first_layer_dense_uniform(params:list, model_sparsity:float):
    sparsities = [1.] + [model_sparsity] * (len(params) - 1)
    return sparsities
```

### 2. Drop Criterion

...

### 3. Redistribute Criterion

...

### 4. Grow Criterion

...
