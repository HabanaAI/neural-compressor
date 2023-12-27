#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
import json

try:
    from neural_compressor.utils.utility import LazyImport

    torch = LazyImport("torch")
    from neural_compressor.utils import logger
except:
    import logging

    import torch

    logger = logging.getLogger()

import numpy
from tqdm import tqdm

from .utils import *


class AutoAlpha:
    def __init__(
        self,
        model,
        dataloader,
        absorb_to_layer,
        input_mins,
        input_maxes,
        auto_alpha_config,
        device,
        q_func=None,
        example_inputs=None,
    ):
        self.model = model.to("cpu")
        self.dataloader = dataloader
        self.q_func = q_func
        self.absorb_to_layer = absorb_to_layer
        self.input_mins = input_mins
        self.input_maxes = input_maxes
        self.auto_alpha_config = auto_alpha_config
        self.device = device
        self.ordered_modules = []
        self.check_configs()
        self.quant_layers = []
        for key in input_mins.keys():
            self.quant_layers.append(key)
        self.sq_layers = []
        for key in absorb_to_layer.keys():
            self.sq_layers.extend(absorb_to_layer[key])
        self.attach_module_name()
        self.example_inputs = example_inputs
        if self.example_inputs is None:
            self.example_inputs = self._get_example_input()

    @torch.no_grad()
    def tune(self):
        self.ordered_modules = self._get_ordered_module()
        tmp = 1

    def check_configs(self):
        assert self.dataloader is not None or self.q_func is not None

    def attach_module_name(self):
        for n, m in self.model.named_modules():
            if n in self.quant_layers:
                m.name = n

    def _get_ordered_module(self):
        hook_handles = []
        for n, m in self.model.named_modules():
            if n in self.quant_layers:
                hook_func = self._get_save_name_hook()
                hook_handle = m.register_forward_hook(hook_func)
                hook_handles.append(hook_handle)
        if self.example_inputs is not None:
            model_forward_per_sample(self.model, self.example_inputs, self.device)
        else:
            self.q_func(self.model)
        for handle in hook_handles:
            handle.remove()
        return self.ordered_modules

    def _get_save_name_hook(self):
        ##set layer name
        def save_name_hook(module, inputs, outputs):
            self.ordered_modules.append(module.name)

        return save_name_hook

    def _get_example_input(self):
        if self.dataloader is None and self.example_inputs is None:
            return None
        if self.example_inputs is None:
            try:
                for idx, (input, label) in enumerate(self.dataloader):
                    self.example_inputs = input
                    break
            except:
                for idx, input in enumerate(self.dataloader):
                    self.example_inputs = input
                    break

        return self.example_inputs

    # def model_forward_data(self, iters, shuffle):
    #     pass
