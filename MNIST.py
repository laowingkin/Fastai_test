import numpy as np
import pandas as pd
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(do_flip=False),
                                 size=26)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(1)