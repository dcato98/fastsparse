# FastSparse
> Customizable Fastai+PyTorch implementation of sparse model training methods (SET, SNFS, RigL).


## _Warning: this repo is undergoing active development_

## Getting Started

### Install

`pip install fastsparse`

### Sparse Algorithms

This network implements the following sparse algorithms:

| Abbr. | Sparse Algorithm | in FastSparse | Notes |
| :- | :- | :- | :- |
|  | static sparsity baseline | omit `DynamicSparseTrainingCallback` |  |
| SET | [Sparse Evolutionary Training](https://arxiv.org/abs/1901.09181) (Jan 2019) | `DynamicSparseTrainingCallback(**SET_presets)` |  |
| SNFS | [Sparse Networks From Scratch](https://arxiv.org/abs/1907.04840) (Jul 2019) | `DynamicSparseTrainingCallback(**SNFS_presets)` | redistribution not implemented* |
| RigL | [Rigged Lottery](https://arxiv.org/abs/1911.11134) (Nov 2019) | `DynamicSparseTrainingCallback(**RigL_presets)` |  |

\*Authors of the RigL paper demonstrate that using SNFS + Erdos-Renyi-Kernel distribution - redistribution outperforms SNFS + uniform sparsity + redistribution (at least on the measured benchmarks).

### Fastai demo

With just 4 additional lines of code, you can train your model using the latest dynamic sparse training techniques. This example achieves >99% accuracy on MNIST using a ResNet34 with only 1% of the weights.

```python
# (0) install the library
# ! pip install fastsparse 

from fastai.vision.all import *

# (1) import this package
import fastsparse as sparse

path = untar_data(URLs.MNIST)
dls = ImageDataLoaders.from_folder(path, 'training', 'testing')
learn = cnn_learner(dls, resnet34, metrics=error_rate, pretrained=False)

# (2) sparsify initial model + enforce masks
sparse_hooks = sparse.sparsify_model(learn.model, 
                                     model_sparsity=0.99,
                                     sparse_f=sparse.erdos_renyi_sparsity)

# (3) schedule dynamic mask updates
cbs = [sparse.DynamicSparseTrainingCallback(**sparse.SNFS_presets, 
                                            batches_per_update=32)]

learn.fit_one_cycle(5, cbs=cbs)

# (4) remove hooks that enforce masks
sparse_hooks.remove()
```

Simply omit the `DynamicSparseTrainingCallback` to train a fixed-sparsity model as a baseline.

### PyTorch demo (*not implemented yet*)

```python
import torch
from torchvision import models

data = ...
model = ...
opt = ...
opt = DynamicSparseTrainingOptimizerWrapper(model, opt, **RigL_kwargs)

### Modified training step
# sparse_opt.step(...) will determine whether to:
#  (A) take a regular opt step, or
#  (B) update network connectivity
def sparse_train_step(model, xb, yb, loss_func, sparse_opt, step, pct_train):
    preds = model(xb)
    loss = loss_func(preds, yb)
    loss.backward()
    sparse_opt.step(step, pct_train)
    sparse_opt.zero_grad()
```

### Save/Reload demo

Here is an example of saving a model and reloading it to resume training.

```python
from fastai.vision.all import *
from fastsparse import *

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_folder(untar_data(URLs.MNIST_TINY))
learn = cnn_learner(dls, resnet18, metrics=accuracy, pretrained=False)
sparse_hooks = sparsify_model(learn.model, model_sparsity=0.9, sparse_f=erdos_renyi_sparsity)
dst_kwargs = {**SNFS_presets, **{'batches_per_update': 8}}
cbs = DynamicSparseTrainingCallback(**dst_kwargs)

learn.fit_flat_cos(5, cbs=cbs)

# (0) save model as usual (masks are stored automatically)
save_model('sparse_tiny_mnist', learn.model, learn.opt)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.482798</td>
      <td>0.698934</td>
      <td>0.505007</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.302442</td>
      <td>0.656283</td>
      <td>0.512160</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.238623</td>
      <td>0.175693</td>
      <td>0.935622</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.203908</td>
      <td>0.028619</td>
      <td>0.992847</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.162143</td>
      <td>0.033945</td>
      <td>0.989986</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


```python
### manually restart notebook ###

# (1) then recreate learner as usual

from fastai.vision.all import *
from fastsparse import *

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_folder(untar_data(URLs.MNIST_TINY))
learn = cnn_learner(dls, resnet18, metrics=accuracy, pretrained=False)

# (2) re-sparsify model (this adds the masks to the parameters)
sparse_hooks = sparsify_model(learn.model, model_sparsity=0.9, sparse_f=erdos_renyi_sparsity) # <-- initial sparsity + enforce masks

# (3) load model as usual
load_model('sparse_tiny_mnist', learn.model, learn.opt)

# (5) check validation loss & accuracy to verify we've loaded it successfully
val_loss, val_acc = learn.validate()
print(f'validation loss: {val_loss}, validation accuracy: {val_acc}')

# (4) optionally, continue training; otherwise remove sparsity-preserving hooks
sparse_hooks.remove()
```

    /home/dc/anaconda3/envs/fastai/lib/python3.8/site-packages/fastai/learner.py:53: UserWarning: Could not load the optimizer state.
      if with_opt: warn("Could not load the optimizer state.")






    validation loss: 0.033944640308618546, validation accuracy: 0.9899857044219971


### Training with Large Batch Sizes

Authors of the Rigged Lottery paper hypothesize that the effectiveness of using the gradient magnitude for determining which connections to grow is partly due to their large batch size (4096 for ImageNet). Those without access to multi-gpu clusters can achieve effective batch sizes of this size by using fastai's `GradientAccumulation` callback, which has been tested to be compatible with this package's `DynamicSparseTrainingCallback`.

### Training with Small # of Epochs

Dynamic sparse training algorithms work by modifying the network connectivity during training, dropping some weights and allowing others to regrow. By default, network connectivity is modified at the end of each epoch. When training with few epochs, however, there will be few chances to explore which weights to connect. To update more frequently, in `DynamicSparseTrainingCallback`, set `batches_per_update` to a smaller # of batches than occur in one training epoch. Varying the number of batches per update trades off the frequency of updates with stability in making good updates.

## Customization

There are many ways to implement and test your own dynamic sparse algorithms using FastSparse.

### Custom Initial Sparsity Distribution:

Define your own initial sparsity distribution by setting `sparsify_method` in `sparsify_model` to a custom function. For example, this function (included in library) will keep the first layer dense and set the remaining layers to a fixed sparsity.

```python
def first_layer_dense_uniform(params:list, model_sparsity:float):
    sparsities = [1.] + [model_sparsity] * (len(params) - 1)
    return sparsities
```

### Custom Drop Criterion

While published papers like SNFS and RigL refer to 'drop criterion', this library implements the reverse, a 'keep criterion'. This is a function that returns a score for each weight, where the largest `M` scores will be and `M` is determined by the decay schedule. For example, both Sparse Networks From Scratch and Rigged Lottery both use the magnitude of the weights (in FastSparse: `weight_magnitude`).

This can easily be customized in FastSparse by defining your own keep score function:

```python
def custom_keep_scoring_function(param, opt):
    score = ...
    assert param.shape == score.shape
    return score
```

Then pass your custom function into the sparse training callback:

```python
DynamicSparseTrainingCallback(..., keep_score_f=custom_keep_scoring_function)
```

### Custom Grow Criterion

The grow criterion is a function that returns a score for each weight, where the largest `N` scores will be and `N` is determined by the decay schedule. For example, Sparse Networks From Scrath grows weights according to the momentum of the gradient, while Rigged Lottery uses the magnitude of the gradient (in FastSparse, `gradient_momentum` and `gradient_magnitude` respectively).

```python
def custom_grow_scoring_function(param, opt):
    score = ...
    assert param.shape == score.shape
    return score
```

Then pass your custom function into the sparse training callback:

```python
DynamicSparseTrainingCallback(..., grow_score_f=custom_grow_scoring_function)
```

## Replication Results

In machine learning, is very easy for seemingly insignificant differences in algorithmic implementation to have a noticeable impact on final results. Therefore, this section compares results from this implementation to results reported in published papers.

TODO...

## Under-The-Hood Details

Here's what's going on. 

When you run `sparsify_model(learn.model, 0.9)`, this adds sparse masks and add pre_forward hooks to enforce masks on weights during forward pass.

> By default, a uniform sparsity distribution is used. Change the sparsity distribution to Erdos-Renyi with `sparsify_model(learn.model, 0.9, sparse_init_f=erdos_renyi)`, or pass in your custom function (see [Customization](#Customization)

> To avoid adding pre_forward hooks, use `sparsify_model(learn.model, 0.9, enforce_masks=False)`.

When you add the `DynamicSparseTrainingCallback` callback, ... TODO complete section
