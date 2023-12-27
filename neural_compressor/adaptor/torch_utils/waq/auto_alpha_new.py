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


class WrapperLayer(torch.nn.Module):
    ##todo add minmax tuning later, different bits?
    def __init__(self, layer, input_min, input_max, act_bits=8, weight_bits=8):
        super(WrapperLayer, self).__init__()
        self.orig_layer = layer
        self.add_module("orig_layer", layer)  # set orig_layer in get/set_module
        self.quant = False
        self.q_input = None
        self.input_max = input_max
        self.input_min = input_min
        self.weight_scale = None
        self.input_scale = None
        self.input_q_min, self.input_q_max = 0, 2**act_bits  ## asym default
        self.weight_q_min, self.weight_q_max = -(2.0 ** (weight_bits - 1)), 2.0 ** (weight_bits - 1) - 1.0
        self.eps = 1e-5
        # self.eps = torch.finfo(torch.float32).eps

    def _calculate_qparams(self, input_min, input_max, input_scale=1.0):
        # calculate scale and zero_point
        min_val = torch.min(input_min * input_scale)
        max_val = torch.max(input_max * input_scale)
        # work when min_val bigger than zero.
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        scale = (max_val_pos - min_val_neg) / float(self.input_q_max - self.input_q_min)
        scale = torch.max(scale, torch.tensor(self.eps, device=scale.device))
        zero_point = self.input_q_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, self.input_q_min, self.input_q_max)
        return scale, zero_point

    def _get_weight_scale(self):
        # get weight scale and zero_point
        from torch.ao.quantization.observer import default_per_channel_weight_observer

        obs = default_per_channel_weight_observer()
        obs(self.sq_linear.weight)
        scale, _ = obs.calculate_qparams()  ##/scale,need to have a check
        return scale

    def enable_quant(self):
        self.quant = True

    def disable_quant(self):
        self.quant = False

    @torch.no_grad()
    def update_scale(self, input_scale, weight_scale):
        self.layer = copy.deepcopy(self.orig_layer)
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.x_scale, self.x_zp = self._calculate_qparams(self.input_min, self.input_max, self.input_scale)
        self.layer.weight *= weight_scale
        self.w_scale = self._get_weight_scale(self.layer.weight)

    # def quant_dequant_w(self, weight, scale, zp):
    #     weight = weight / scale
    # def qdq_x(self, x, input_scale):
    #     x=x*input_scale
    #     q_x = torch.round(x/self.x_scale+self.x_zp)
    #     q_x = torch.round(x / scale + bias)
    #     q_x.clamp_(q_min, q_max)

    ##TODO better tradeoff performance and memory, currently it's too slow
    def q_dq_forward(self, x, input_scale, weight_scale):
        layer_copy = self.layer

        if weight_scale is not None:
            layer_copy.weight *= weight_scale
        q_dq_weight = quant_dequant_w(layer_copy)
        layer_copy.weight.data.copy_(q_dq_weight)
        if input_scale is None:
            x = quant_dequant_x(x, self.input_min, self.input_max)
        else:
            x = input_scale * x
            x = quant_dequant_x(x, self.input_min * input_scale, self.input_max * input_scale)  ##FIXME
        output = layer_copy(x)
        return output

    def forward(self, x):
        if self.quant:
            output = self.q_dq_forward(x, self.input_scale, self.weight_scale)
        else:
            output = self.orig_layer(x)
        self.output = output
        return output


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
        op_types,
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
        self.ordered_module_names = []
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
        self.ordered_module_names = self._get_ordered_module()
        pre_block_modules, in_block_modules, post_block_modules = self._parse_module_info(self.model)
        ##for module not in block, only layerwise loss is valid

        tmp = 1

    def tune_single_layer_outside_block(self, name):
        model = get_module(name)

    def _parse_module_info(self, model):
        block_names = get_block_names(model)
        pre_block_modules = []
        in_block_modules = []
        post_block_modules = []
        for block_name in block_names:
            block_module = get_module(self.model, block_name)
            for n, m in block_module.named_modules():
                if hasattr(m, "name"):
                    in_block_modules.append(m.name)
        for name in self.ordered_module_names:
            if name not in in_block_modules:
                pre_block_modules.append(name)
            else:
                break
        for name in reversed(self.ordered_module_names):
            if name not in in_block_modules:
                post_block_modules.append(name)
            else:
                break
        return pre_block_modules, in_block_modules, post_block_modules

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
        return self.ordered_module_names

    def _get_save_name_hook(self):
        ##set layer name
        def save_name_hook(module, inputs, outputs):
            self.ordered_module_names.append(module.name)

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
