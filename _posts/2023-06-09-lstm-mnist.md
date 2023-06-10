---
layout: post
title:  Sequential MNIST with an LSTM
date:   2023-06-09 22:19:30 -0700
categories: ml
---

The sequential MNIST dataset can be found [here](https://edwin-de-jong.github.io/blog/mnist-sequence-data/). Essentially it turns the classic images into brush strokes giving each image as an array of dx and dy. This notebook uses an LSTM to detect the numbers and achieves 93.1% accuracy. I do not do any data augmentation which most likely is lowering the performance and definetly harms its real life usage (I wrote a simple script to draw and recognize numbers and the way I drew numbers mattered a lot - a 7 drawn in reverse was detected differently than a normally drawn 7).

## Imports


```python
import os
import re

import torch

import numpy as np
import pandas as pd

from rich import print
from tqdm import tqdm

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from matplotlib import pyplot as plt

torch.cuda.is_available()
```




    True



## Data
To get the data, downlaod it from [here](https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz). After extracting it, it should all be in a folder called sequences. All of the images are labaled with `{train/test}img-{x}-inputdata.txt` where `{x}` is a number from 1 - 60000 (not its label). The classic labels are found in 2 files, `trainlabels.txt` and `testlabels.txt`.

To get an image and its label, you need to find the file with `x = idx` and the x'th line of either `{train/test}labels.txt`. The labels files are a long list of numbers. The i'th number corresponds with the file with that number.

Each file is a list of `dx`, `dy`, `eos`, and `eof` seperated by space. A sample is shown below.

```txt
9 14 0 0
0 1 0 0
-1 1 0 0
0 1 0 0
0 1 0 0
0 1 0 0
0 1 0 0
0 1 0 0
0 1 0 0
1 0 0 0
1 0 0 0
1 1 0 0
1 -1 0 0
........
```


```python
path = "data/sequences/"
files = os.listdir(path)
train_data = [path + f for f in files if os.path.isfile(path + f) and "inputdata" in f and "trainimg" in f]
test_data = [path + f for f in files if os.path.isfile(path + f) and "inputdata" in f and "testimg" in f]
print("Found", len(train_data), "training images and", len(test_data), "testing images")

train_labels = np.genfromtxt(path + "trainlabels.txt", dtype=int)
test_labels = np.genfromtxt(path + "testlabels.txt", dtype=int)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Found <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">60000</span> training images and <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10000</span> testing images
</pre>



### Sample data
First we define some helper functions. `get_idx` has a simple regex to convert file names into what idx it is. We will use this to sort the files for quicker access later. `make_img` is an extremely slow but easy way to create an image which integrates `dx` and `dy` to find which pixels should be made black. `eos` defines the end of a stroke, when the pen needs to be lifted to go somewhere else. `eof` defines the end of the file and is the last pixel.


```python
def get_idx(f):
    idx = re.search("-\d+-", f)
    idx = int(f[idx.start() + 1:idx.end() - 1])
    
    return idx


def make_img(data):
    img = np.zeros((28, 28))
    x, y = (0, 0)

    for row in data:
        dx = row[0]
        dy = row[1]
        eos = row[2]
        eof = row[3]

        x = x + dx
        y = y + dy

        if eos == 0 and eof == 0:
            img[x][y] = 1.0
        elif eof == 1:
            img[x][y] = 0.75
        else:
            img[x][y] = 0.5
    
    return img
```


```python
train_data = sorted(train_data, key=lambda f: get_idx(f))
test_data = sorted(test_data, key=lambda f: get_idx(f))

file = train_data[1]
idx = get_idx(file)

print("Index: ", idx, "File:", file, "Label:", train_data[idx])

data = np.genfromtxt(file, delimiter=' ', dtype=int)

print(data)

img = make_img(data)
plt.imshow(img.T)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Index:  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> File: data/sequences/trainimg-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>-inputdata.txt Label: data/sequences/trainimg-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>-inputdata.txt
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">[[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">]</span>
 <span style="font-weight: bold">[</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">]]</span>
</pre>

   
![png](/assets/mnist-lstm_files/mnist-lstm_7_3.png)
    


### Dataset

To turn our array into a torch dataset, we need to efficiently get from an index to a corresponding file/label pair. Since we already got the file list sorted above and have the label list, we won't regenerate those in the dataset. We will take in a list of labels that has already been loaded from the label file and take a list of image file paths that is sorted before in the correct order. This makes the dataset just need to load from a file.


```python
class SeqMNIST(Dataset):
    def __init__(self, labels, data):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = np.genfromtxt(self.data[idx], delimiter=' ', dtype=int)

        return label, img
```


```python
train_dataset = SeqMNIST(train_labels, train_data)
test_dataset = SeqMNIST(test_labels, test_data)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
```


```python
for i, (label, data) in enumerate(train_dataset):
    ax = plt.subplot(1, 6, i + 1)
    plt.tight_layout()
    ax.set_title('{}'.format(label))
    ax.axis('off')
    img = make_img(data)
    plt.imshow(img.T)

    if i == 5:
        plt.show()
        break
```


    
![png](/assets/mnist-lstm_files/mnist-lstm_11_0.png)
    


## Model
For the model, because the sequences can be of any lenght, we are using an LSTM. Our model is very simple with only an LSTM layer feeding into a linear and softmax layer that outputs to the 10 classes.


```python
class LSTMTagger(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=10):
        super(LSTMTagger, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.hidden2tag = nn.Linear(hidden_size * num_layers, output_size)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq)[-1]
        lstm_out = torch.flatten(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.softmax(tag_space, dim=0)
        return tag_scores
```


```python
model = LSTMTagger(4, 32, 6, 10).cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(model)
print(sum(p.numel() for p in model.parameters()), "trainable parameters")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LSTMTagger</span><span style="font-weight: bold">(</span>
  <span style="font-weight: bold">(</span>lstm<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LSTM</span><span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #808000; text-decoration-color: #808000">num_layers</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span><span style="font-weight: bold">)</span>
  <span style="font-weight: bold">(</span>hidden2tag<span style="font-weight: bold">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">in_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">192</span>, <span style="color: #808000; text-decoration-color: #808000">out_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, <span style="color: #808000; text-decoration-color: #808000">bias</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="font-weight: bold">)</span>
<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">49034</span> trainable parameters
</pre>



## Training
For training, we will save the model every time the validation accuracy increases. I was paranoid of losing the model so I made it save too liberally. So when running this, it saved way too many times and I had to clean my folder of all the random model checkpoints I had.


```python
criterion = nn.CrossEntropyLoss()

train_loss = []
train_acc = []
val_loss = []
val_acc = []

max_val = 0

for epoch in range(50):
    avg_loss = 0
    correct = 0
    en = tqdm(enumerate(train_loader))
    for step, (label, seq) in en:
        optimizer.zero_grad()
        
        bx = Variable(seq).cuda().squeeze().float()
        output = model(bx)
        
        maxi = torch.argmax(output)
        if maxi.item() == label.item():
            correct += 1
        
        label = label.cuda().squeeze()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        
        en.set_description("{}/{} - Train accuracy {:.4f}".format(step, len(train_loader), correct / (step + 1)))
        
    print("Completed epoch", epoch, "with avg. loss", avg_loss / len(train_loader), "and accuracy", correct / len(train_loader))
    train_loss.append(avg_loss / len(train_loader))
    train_acc.append(correct / len(train_loader))
    
    avg_loss = 0
    correct = 0
    en = tqdm(enumerate(test_loader))
    for step, (label, seq) in en:
        with torch.no_grad():
            bx = Variable(seq).cuda().squeeze().float()
            output = model(bx)
            
            maxi = torch.argmax(output)
            if maxi.item() == label.item():
                correct += 1
            
            label = label.cuda().squeeze()
            loss = criterion(output, label)
            
            en.set_description("{}/{} - Val accuracy {:.4f}".format(step, len(test_loader), correct / (step + 1)))
    
    val_loss.append(avg_loss / len(test_loader))
    val_acc.append(correct / len(test_loader))
    
    if correct / len(test_loader) > max_val:
        max_val = correct / len(test_loader)
        print("Saving checkpoint with", max_val, "accuracy on epoch", epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            }, "lstm-chk-{}.pt".format(epoch))
```

```python
torch.save(model, 'models/lstm-trained.pt')
```


```python
model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted.pt')
```
