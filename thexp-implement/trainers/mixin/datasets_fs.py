"""
dataset mixin for few-shot learning
"""

from thexp import Trainer
import numpy as np
from trainers import *
from thexp.contrib.data import splits
from thexp import DatasetBuilder, DataBundler

from data.constant import norm_val
from data.transforms import ToTensor
from data.dataxy import datasets
from data.transforms import Weak, Strong, ToNormTensor


class NwayKshotMixin(Trainer):
    def datasets(self, params: FewshotParams):
        data_size = params.n_way * params.k_shot
        query_size = params.n_way * params.k_query

