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

from collections import OrderedDict
from functools import partial

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
        self.input_max = input_max
        self.input_min = input_min
        self.act_bits = act_bits
        self.weight_bits = weight_bits
        self.weight_scale = 1.0
        self.input_scale = 1.0
        self.eps = 1e-5
        # self.eps = torch.finfo(torch.float32).eps

    def _calculate_qparams(self, input_min, input_max, input_scale=1.0):
        # calculate scale and zero_point
        input_q_min, input_q_max = 0, 2**self.act_bits  ## asym default
        min_val = torch.min(input_min * input_scale)
        max_val = torch.max(input_max * input_scale)
        # work when min_val bigger than zero.
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        scale = (max_val_pos - min_val_neg) / float(input_q_max - input_q_min)
        scale = torch.max(scale, torch.tensor(self.eps, device=scale.device))
        zero_point = input_q_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, input_q_min, input_q_max)
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
        self.x_q_scale, self.x_q_zp = self._calculate_qparams(self.input_min, self.input_max, self.input_scale)
        self.layer.weight *= weight_scale
        self.w_q_scale = self._get_weight_scale(self.layer.weight)
        self.layer_quant = copy.deepcopy(self.layer)

        self.layer_quant.weight *= self.weight_scale
        q_dq_weight = quant_dequant_w(self.layer_quant.weight, self.w_q_scale, self.weight_bits)
        self.layer_quant.weight.data.copy_(q_dq_weight)

    # def quant_dequant_w(self, weight, scale, zp):
    #     weight = weight / scale
    # def qdq_x(self, x, input_scale):
    #     x=x*input_scale
    #     q_x = torch.round(x/self.x_scale+self.x_zp)
    #     q_x = torch.round(x / scale + bias)
    #     q_x.clamp_(q_min, q_max)

    ##TODO better tradeoff performance and memory, currently it's too slow
    def q_dq_forward(self, x):
        x = self.input_scale * x
        x = quant_dequant_x(x, self.x_q_scale, self.x_q_zp, self.act_bits)  ##FIXME
        output = self.layer_quant(x)
        return output

    def forward(self, x):
        if self.quant:
            output = self.q_dq_forward(x)
        else:
            output = self.orig_layer(x)
        self.output = output
        return output


def move_input_to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = move_input_to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res = []
        for inp in input:
            input_res.append(move_input_to_device(inp, device))
        input = input_res
    return input


class SaveBlockInputs:
    """Cache the inputs of the first block."""

    def __init__(self, model, dataloader, block_name):  ##TODO support qfunc
        """Initializes the SaveInputs class.

        Args:
            model: The model to be used.
            dataloader: The dataloader for the input data.
            seqlen (int): The sequence length.
            block_name (str): The name of the block.
        """
        self.model = model.eval()
        self.dataloader = dataloader
        self.inputs = {}
        self.block_name = block_name

    @torch.no_grad()
    def get_forward_func(self, name):
        """Gets the forward function.

        Args:
            name (str): The name of the function.

        Returns:
            function: The forward function.
        """

        def forward(_, hidden_states, *positional_args, **kwargs):
            dim = int((hasattr(self.model, "config") and "chatglm" in self.model.config.model_type))
            if name in self.inputs:
                # data = torch.cat([self.inputs[name]["input_ids"], hidden_states.to("cpu")], dim=dim)
                self.inputs[name]["input_ids"].append(hidden_states.to("cpu"))
            else:
                self.inputs[name] = {}
                self.inputs[name]["input_ids"] = [hidden_states.to("cpu")]

            if "positional_inputs" not in self.inputs[name]:
                self.inputs[name]["positional_inputs"] = []
            for idx, item in enumerate(positional_args):
                self.inputs[name]["positional_inputs"] = move_input_to_device(positional_args)

            for key in kwargs.keys():
                if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) or (key == "alibi"):
                    if "attention_mask" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = []
                        if kwargs[key] is not None:
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key].append(kwargs[key].to("cpu"))
                            else:
                                self.inputs[name][key] = [kwargs[key].to("cpu")]
                    elif "alibi" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if isinstance(kwargs[key], torch.Tensor):
                            alibi = kwargs[key]
                            batch = kwargs["attention_mask"].shape[0]
                            alibi = alibi.reshape(batch, -1, alibi.shape[1], alibi.shape[2])
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key].append(alibi.to("cpu"))
                            else:
                                self.inputs[name][key] = [alibi.to("cpu")]
                    elif key not in self.inputs[name].keys():
                        self.inputs[name][key] = move_input_to_device(kwargs[key], device=torch.device("cpu"))
            raise NotImplementedError

        return forward

    @torch.no_grad()
    def get_inputs(self, n_samples=512):
        """Gets the inputs.

        Args:
            n_samples (int): The number of samples.

        Returns:
            dict: The inputs.
        """
        total_cnt = 0
        self._replace_forward()
        for data in self.dataloader:
            if data is None:
                continue
            if isinstance(data, tuple) or isinstance(data, list):
                data = data[0]
            if isinstance(data, torch.Tensor):
                input_ids = data.to(self.model.device)
            else:
                input_ids = data["input_ids"].to(self.model.device)
            if total_cnt + input_ids.shape[0] > n_samples:
                input_ids = input_ids[: n_samples - total_cnt, ...]
            try:
                self.model(input_ids)
            except NotImplementedError:
                pass
            except Exception as error:
                logger.error(error)
            total_cnt += input_ids.shape[0]
            if total_cnt >= n_samples:
                break
        self._recover_forward()
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataloader or decease the sequence length"
            )
            exit()
        elif total_cnt < n_samples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantification. "
                f"Effective samples size:{total_cnt}, Target sample size:{n_samples}"
            )
        res = self.inputs[self.block_name]
        if "input_ids" in res.keys():
            total_samples = 0
            for item in res["input_ids"]:
                total_samples += item.shape[0]
            if total_samples < n_samples:
                logger.warning("only cache {total_samples}")

        return res

    def _recover_forward(self):
        """Recovers the forward function."""
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
                break

    def _replace_forward(self):
        """Replaces the forward function."""
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.orig_forward = m.forward
                m.forward = partial(self.get_forward_func(n), m)
                break


def block_forward(block, input_ids, input_others, amp=False, amp_dtype=torch.float16, device=torch.device("cpu")):
    """Performs a forward pass through a block with the given inputs.

    Args:
    block: The block to perform the forward pass on.
    input_ids: The input IDs.
    input_others: A dictionary containing other input data.
    amp: A boolean indicating whether to use automatic mixed precision.
    amp_dtype: The data type for automatic mixed precision.
    device: The target device.

    Returns:
    output: The output of the forward pass.
    """
    if input_ids.device != device:
        # input_ids, input_others = move_to_device(input_ids, input_others, device)
        input_ids = move_input_to_device(input_ids, device)
        input_others = move_input_to_device(input_others, device)
    if "alibi" in input_others.keys():
        attention_mask = input_others["attention_mask"]
        alibi = input_others["alibi"]
        if alibi is not None:
            alibi = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
        if amp and not check_is_cpu(device):
            with autocast(device_type="cuda", dtype=amp_dtype):  # pragma: no cover
                output = block(
                    input_ids, attention_mask=attention_mask, alibi=alibi
                )  ##TODO is this correct for all models with alibi?
        elif amp and check_is_cpu(device):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
        else:
            output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
    else:
        input_tuple = input_others.pop("positional_inputs", None)
        if amp and not check_is_cpu(device):
            with autocast(device_type="cuda", dtype=amp_dtype):  # pragma: no cover
                output = block.forward(input_ids, *input_tuple, **input_others)
        elif amp and check_is_cpu(device):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block.forward(input_ids, *input_tuple, **input_others)
        else:
            output = block.forward(input_ids, *input_tuple, **input_others)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    return output


class AutoAlpha:
    def __init__(
        self,
        model,
        dataloader,
        absorb_to_layer,  ##this only for sq layers
        init_alpha=0.5,
        alpha_min=0.0,
        alpha_max=1.0,
        alpha_step=0.1,
        shared_criterion="mean",
        loss_type="block_wise",
        use_quant_input=True,
        n_samples=None,  ##512 for cuda, 128 for cpu?
        bs=8,
        half_precision=False,
        op_types=[torch.nn.Linear],  ##need to support other layers
        fp_layers=[],
        device="cpu",
        q_func=None,
        example_inputs=None,
        act_bits=8,
        w_bits=8,
    ):
        self.model = model.to("cpu")
        self.model.eval()
        self.dataloader = dataloader
        self.q_func = q_func
        self.absorb_to_layer = absorb_to_layer
        self.init_alpha = init_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_step = alpha_step
        self.shared_criterion = shared_criterion
        self.loss_type = loss_type
        self.use_quant_input = use_quant_input
        self.n_samples = n_samples
        self.bs = bs
        self.half_precision = half_precision
        self.device = device
        if self.n_samples is None:
            self.n_samples = 512 if "cpu" not in str(self.device) else 128

        self.ordered_module_names = []
        self.check_configs()
        self.quant_layers = self._get_quant_layer_names(op_types, fp_layers)
        self.sq_layers = []
        for key in absorb_to_layer.keys():
            self.sq_layers.extend(absorb_to_layer[key])
        self.attach_module_name()
        self.example_inputs = example_inputs
        if self.example_inputs is None:
            self.example_inputs = self._get_example_input()
        if self.loss_type == "model_wise" and "cpu" not in str(self.device):
            try:
                self.model.to(device)
                if self.example_inputs is not None:
                    model_forward_per_sample(self.model, self.example_inputs, self.device)
            except:
                logger.warning(
                    "Due to memory constraints, the model-wide loss needs to be replaced with block-wise loss."
                )

                self.loss_type = "block_wise"
            self.model.to("cpu")
        if self.loss_type == "block_wise":
            pass  ##
            ##TODO check whether could do in block-wise and use quant_input way easily, otherwise using layer-wise,for example, mutimodal model
        self.act_bits = act_bits
        self.w_bits = w_bits

    def save_block_input(self, block_name):
        save_input = SaveBlockInputs(self.model, self.dataloader, block_name)
        first_block_inputs = save_input.get_inputs(self.n_samples)
        return first_block_inputs

    def _get_quant_layer_names(self, op_types, fp_layers):
        quant_layers = []
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(op_types)) and n not in fp_layers:
                quant_layers.append(n)
        return quant_layers

    def tune_group_layers_with_block_loss(self, group, loss_module, loss_input):
        pass

    def tune_single_block(self, module, input):  ##block_loss
        ##module {block_name:[q,k,v],[fc1,][fc2]}
        ##calibration
        block_name = module[0]
        ordered_submodules = module[1]
        module = get_module(self.model, block_name)
        module.to(self.device)
        from .calibration import Calibration

        calib = Calibration(module, dataloder=None, q_func=self.block_qfunc)
        self.q_func_input = input
        input_mins, input_maxes = calib.calibrate(-1)
        ##wrapper layers
        for n, m in module.named_modules():
            if not hasattr(m, "name"):
                continue
            new_layer = WrapperLayer(m, input_mins[n], input_maxes[n], self.act_bits, self.w_bits)
            set_module(module, n, new_layer)
        ##for each layer group, tune
        for group in ordered_submodules:
            self.tune_group_layers_with_block_loss(group, module, inpur)

        tmp = 1

    def block_qfunc(self, module):
        input_ids = self.q_func_input["input_ids"]
        self.q_func_input.pop("input_ids")
        input_others = self.q_func_input
        for index in range(len(input_ids)):
            input_ids_tmp = input_ids[index]
            input_others_tmps = {}
            for k, v in input_others.items():
                if len(v) > 1:
                    input_others_tmps[k] = v[index]
                else:
                    input_others_tmps[k] = v

            block_forward(
                module,
                input_ids=input_ids_tmp,
                input_others=input_others_tmps,
                amp=self.half_precision,
                device=self.device,
            )

    def tune_with_quant_input(self, modules):
        first_block_name = None
        for key in modules.keys():
            first_block_name = key
            break
        block_inputs = self.save_block_input(first_block_name)
        for item in modules.items():
            self.tune_single_block(item, block_inputs)
            ##TODO reset the block_inputs

    def tune_wo_quant_input(self, block_names, layer_names):
        pass

    @torch.no_grad()
    def tune(self):
        self.ordered_module_names = self._get_ordered_module()
        pre_block_modules, in_block_modules, post_block_modules = self._parse_module_info(
            self.model
        )  ##TODO there are bug for multimodal model
        assert len(pre_block_modules) == 0, "only support zero len pre block modules currently"  ##TODO handle it later
        if len(in_block_modules) != 0 and self.use_quant_input:
            self.tune_with_quant_input(in_block_modules)
            self.tune_wo_quant_input([], post_block_modules)
        else:
            self.tune_wo_quant_input(in_block_modules, post_block_modules)

        tmp = 1

    def tune_single_layer_outside_block(self, name):
        model = get_module(name)

    def _parse_module_info(self, model):
        """Parses the module in pre/in/post blocks and returns the quantized modules for each block.

        Args:
            model: The input model to parse.
        Returns:
            pre_block_quant_modules: List of quantized modules in the pre block.
            in_block_quant_modules: Dictionary of quantized modules in the in block {"block_name":[[q,k,v],[fc1],[fc2]]}.
            post_block_quant_modules: List of quantized modules in the post block.["lm head"]
        """
        ##first parse the module in pre/in/post blocks
        block_names = get_block_names(model)
        tmp_pre_block_quant_modules = []
        tmp_in_block_quant_modules = OrderedDict()
        tmp_in_block_quant_modules_list = []
        tmp_post_block_quant_modules = []

        for block_name in block_names:
            tmp_in_block_quant_modules[block_name] = tmp_in_block_quant_modules.get(block_name, [])
            block_module = get_module(self.model, block_name)
            for n, m in block_module.named_modules():
                if hasattr(m, "name"):
                    tmp_in_block_quant_modules[block_name].append(m.name)
                    tmp_in_block_quant_modules_list.append(m.name)
        for name in self.ordered_module_names:
            if name not in tmp_in_block_quant_modules_list:
                tmp_pre_block_quant_modules.append(name)
            else:
                break
        for name in reversed(self.ordered_module_names):
            if name not in tmp_in_block_quant_modules_list and name not in tmp_pre_block_quant_modules:
                tmp_post_block_quant_modules.append(name)
            else:
                break

        ##make in block ordered
        layer_to_block_name = OrderedDict()
        for block_name, layer_name_in_blocks in tmp_in_block_quant_modules.items():
            for layer in layer_name_in_blocks:
                layer_to_block_name[layer] = block_name
        tmp_in_block_quant_modules = {}
        for name in self.ordered_module_names:
            if name not in layer_to_block_name.keys():
                continue
            block_name = layer_to_block_name[name]
            tmp_in_block_quant_modules[block_name] = tmp_in_block_quant_modules.get(block_name, [])
            tmp_in_block_quant_modules[block_name].append(name)

        handled = []
        pre_block_quant_modules = []
        in_block_quant_modules = OrderedDict()
        post_block_quant_modules = []
        shared_info = {}
        for key in self.absorb_to_layer:
            layer_names = self.absorb_to_layer[key]  ##all shad name
            for layer_name in layer_names:
                shared_info[layer_name] = list(set(layer_names) & set(self.quant_layers))

        for name in tmp_pre_block_quant_modules:
            if name in handled:
                continue
            shared_layers = list(set(shared_info[name]) & set(self.quant_layers))
            pre_block_quant_modules.append(shared_layers)
            handled.extend(shared_layers)

        for key in tmp_in_block_quant_modules.keys():
            names = tmp_in_block_quant_modules[key]
            in_block_quant_modules[key] = in_block_quant_modules.get(key, [])
            for name in names:
                if name in handled:
                    continue
                shared_layers = list(set(shared_info[name]) & set(self.quant_layers))
                in_block_quant_modules[key].append(shared_layers)
                handled.extend(shared_layers)

        for name in tmp_post_block_quant_modules:
            if name in handled:
                continue
            shared_layers = list(set(shared_info[name]) & set(self.quant_layers))
            post_block_quant_modules.append(shared_layers)
            handled.extend(shared_layers)

        return pre_block_quant_modules, in_block_quant_modules, post_block_quant_modules

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
