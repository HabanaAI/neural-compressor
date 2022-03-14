#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import copy
import numpy as np
from .pruner import pruner_registry, Pruner
from .magnitude import BasicMagnitudePruner
from ..utils import logger

@pruner_registry
class GroupLassoPruner(BasicMagnitudePruner):
    def __init__(self, model, local_config, global_config):
        super(GroupLassoPruner, self).__init__(model, local_config, global_config)
        self.alpha = local_config.parameters['alpha']

    def on_post_grad(self):
        for weight_name in self.weights:
            weight_grad = self.model.get_gradient(weight_name)
            weight = np.array(self.model.get_weight(weight_name))
            reshaped_weight = self.pattern.reshape(weight)
            normed = np.linalg.norm(reshaped_weight, 2, axis=(1,3))
            sparsity_per_layer = (normed.numel() - np.count_nonzero(normed)) / normed.numel()
            if sparsity_per_layer < self.sparsity:
                coeff = self.alpha / normed
                coeff[np.isinf(coeff)] = 0
                coeff = self.pattern.repeat_mask(coeff).reshape(weight.shape)
                weight_grad += coeff * weight
            else:
                weight_grad[weight == 0] = 0
            self.model.update_gradient(weight_name, weight_grad)