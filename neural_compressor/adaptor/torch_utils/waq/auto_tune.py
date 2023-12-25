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


class AlphaTuner:
    def __init__(self, model, dataloader, input_info, auto_alpha_args, layers_info, device, sq_info, block_info):
        """Initialize the AlphaTuner with necessary parameters and components."""
        self.model = model
        self.dataloader = dataloader
        self.input_maxes = input_info["input_maxes"]
        self.input_mins = input_info["input_mins"]
        self.input_maxes_abs = input_info["input_maxes_abs"]
        self.calib_sample_num = auto_alpha_args["calib_sample_num"]
        self.alpha_min = auto_alpha_args["alpha_min"]
        self.alpha_max = auto_alpha_args["alpha_max"]
        self.alpha_step = auto_alpha_args["alpha_step"]
        self.shared_criterion = auto_alpha_args["shared_criterion"]
        self.default_alpha = auto_alpha_args["default_alpha"]
        self.op_types = sq_info["op_types"]
        self.absorb_to_layer = layers_info["absorb_to_layer"]
        self.weight_scale_dict = {}
        self.max_value_info = {}  # to record max values for alpha tune
        self.record_max_info = sq_info["record_max_info"]
        self.insert_mul = sq_info["insert_mul"]
        self.allow_absorb = sq_info["allow_absorb"]
        self.weight_clip = sq_info["weight_clip"]
        self._save_scale = sq_info["_save_scale"]
        self.device = device
        self.block_info = block_info

    def _add_blockwise_observer(self, block_modules):
        """
        :param block_modules: the block modules which the observer will insert to
        :return:
        """
        self.blockwise_hook_handles = []
        for key in block_modules.keys():
            hook_func = self._save_blockwise_hook(key)
            hook_handle = block_modules[key].register_forward_hook(hook_func)
            self.blockwise_hook_handles.append(hook_handle)

    def _save_blockwise_hook(self, name):
        """A forward hook to save inputs/outputs of a block
        :param name: the block name
        :return: A hook function."""

        def save_blockwise_hook(module, inputs, outputs):
            self.block_inputs[name] = inputs[0]
            self.block_outputs[name] = outputs[0]

        return save_blockwise_hook

    def _reshape_scale_for_input(self, layer, scale):
        """Reshape the scale for input feature in channel
        :param layer:

        :param scale:
        :return:
        """
        if hasattr(layer, "orig_layer"):
            layer = layer.orig_layer
        if isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def _reshape_scale_for_weight(self, layer, scale):
        """Reshape the scale for weight input channel, depthwise output channel
        :param layer:  torch module
        :param scale: orig scale
        :return: reshaped scale."""
        if hasattr(layer, "orig_layer"):
            layer = layer.orig_layer
        if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:  ##only depthwise conv could hit here
            scale = scale.view(scale.shape[0], 1, 1, 1)  ##mount on output channel

        elif isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def _get_all_hook_module_names(self):
        module_names = []
        for n, module in self.model.named_modules():
            if isinstance(module, tuple(self.op_types)):
                module_names.append(n)
        return module_names

    def _update_scales_for_auto(self, absorb_scales, weight_scales):
        for key in self.absorb_to_layer.keys():
            layer_names = self.absorb_to_layer[key]
            for layer_name in layer_names:
                layer = get_module(self.model, layer_name)
                input_scale = absorb_scales[key]
                weight_scale = weight_scales[layer_name]
                input_scale = self._reshape_scale_for_input(layer, input_scale)
                weight_scale = self._reshape_scale_for_weight(layer, weight_scale)
                layer.update_scale(input_scale, weight_scale)  ##FIXME

    def _change_qdq_for_auto(self, enable=True):
        module_names = self._get_all_hook_module_names()
        for name in module_names:
            name = name.split(".orig_layer")[0]
            module = get_module(self.model, name)
            if not hasattr(module, "orig_layer"):  # skip module if it's not used in calibration
                continue
            if enable:
                module.enable_quant()
            else:
                module.disable_quant()

    def _qdq_model_wrapper_for_auto(self, save_q_input=False):
        """Wrapper all the module with qdq
        :return:"""
        module_names = self._get_all_hook_module_names()
        self.to_unwrap_module_names = module_names
        for name in module_names:
            if name not in self.input_mins:  # skip module if it's not used in calibration
                continue
            module = get_module(self.model, name)
            new_module = WrapperLayer(module, self.input_mins[name], self.input_maxes[name], save_q_input=save_q_input)
            set_module(self.model, name, new_module)

    def _qdq_model_unwrapper_for_auto(self):
        module_names = self.to_unwrap_module_names
        for name in module_names:
            module = get_module(self.model, name)
            if not hasattr(module, "orig_layer"):  # skip module if it's not used in calibration
                continue
            set_module(self.model, name, module.orig_layer)

    def _reshape_in_channel_to_last(self, layer_name):
        """Move the input channel to the last dim
        :param layer_name: Layer name
        :return: The reshaped weight."""
        layer = get_module(self.model, layer_name)
        if layer.__class__.__name__ == "WrapperLayer":
            layer = layer.orig_layer

        weight = layer.weight  ##TODO oc*ic, support transposed conv
        if len(weight.shape) == 4:
            weight = weight.permute(0, 2, 3, 1)
            weight = weight.reshape(-1, weight.shape[-1])
        return weight

    def _cal_scales(self, absorb_to_layer, input_maxes, alpha=0.5, tuning=False):
        """Cal the adjust scales
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, a float of a dict
        :return:"""
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            alpha_tmp = alpha[key] if isinstance(alpha, dict) else alpha
            if alpha_tmp < 0:
                scale = torch.ones((1), device=self.device)
            else:
                input_max = absorb_to_input_maxes[key]
                layer_names = absorb_to_layer[key]
                weights = []
                for layer_name in layer_names:
                    weight = self._reshape_in_channel_to_last(layer_name)
                    weights.append(weight)

                weight_max_per_channel = torch.max(torch.abs(torch.cat(weights, dim=0)), dim=0)[0]
                if self.weight_clip:
                    weight_max_per_channel = weight_max_per_channel.clamp(min=1e-5)
                if self.record_max_info and not tuning:
                    # the input of layers with same absorb layer is the same.
                    input_minmax = [self.input_mins[layer_names[0]], self.input_maxes[layer_names[0]]]
                    self.max_value_info[key] = {}
                    self.max_value_info[key]["alpha"] = alpha_tmp
                    self.max_value_info[key]["input_minmax"] = input_minmax
                    self.max_value_info[key]["weight_max"] = weight_max_per_channel
                    self.max_value_info[key]["absorbed_layer"] = layer_names
                    continue

                if self._save_scale:
                    if key in self.weight_scale_dict and alpha_tmp in self.weight_scale_dict[key]:
                        scale = self.weight_scale_dict[key][alpha_tmp]
                    else:
                        scale = cal_scale(input_max, weights, alpha_tmp)
                else:
                    scale = cal_scale(input_max, weights, alpha_tmp)

            absorb_scales_info[key] = 1.0 / scale
            absorb_scales_info[key][scale == 0] = 0
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                ##self._scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
                if self._save_scale:
                    if layer_name not in self.weight_scale_dict:
                        self.weight_scale_dict[layer_name] = {}
                    self.weight_scale_dict[layer_name][alpha_tmp] = scale
        return absorb_scales_info, weight_scales_info

    def _get_auto_loss(self, output, output_q, loss_type="abs", loss_alpha=1.0):
        """Get the loss for auto tuning
        :param output: Fp32 output for one layer
        :param output_q: Quant output for one layer
        :param loss_type: The type of loss
        :param loss_alpha: Loss alpha i for mean scale error
        :return: A tensor of the loss."""
        if len(output.shape) <= 2:
            max_value = torch.max(torch.abs(output))
        else:
            output = output.reshape(output.shape[0], -1)
            output_q = output_q.reshape(output_q.shape[0], -1)
            max_value = torch.max(torch.abs(output), dim=-1).values.unsqueeze(-1)
            max_value = torch.clip(max_value, 1e-5)
        output = output / max_value  ##FIXME need copy not replace
        output_q = output_q / max_value
        # if loss_type == "nsr":  # nsr is unused at this point.
        #     output[output == 0] = 1e-5
        #     loss = torch.sum(torch.log(1.0 + torch.abs(output - output_q) / torch.abs(output)))
        #     return loss
        if loss_type == "abs":
            return torch.sum(torch.pow(torch.abs(output - output_q), 0.5))
        else:
            return torch.sum((output - output_q) ** 2)

    def _get_sq_layer_names(self):
        """Get the all the hook sq layer
        :return: All the sq layer names."""
        ##TODO this may not fit for folding=False
        module_names = []
        for key in self.absorb_to_layer:
            module_names += self.absorb_to_layer[key]
        return module_names

    def _get_best_alpha(self, absorb_to_layer, loss_alphas, shared_criterion):
        def dict_to_list(dic):
            res = []
            for key in dic.keys():
                res.append((key, dic[key]))
            return res

        best_alpha = {}
        for ln_name in absorb_to_layer.keys():
            layer_names = absorb_to_layer[ln_name]
            cur_shared_criterion = shared_criterion
            if len(layer_names) == 1:
                cur_shared_criterion = "min"
            if cur_shared_criterion == "mean":
                loss_tmp = {}
                for alpha in loss_alphas[layer_names[0]].keys():
                    if alpha not in loss_tmp.keys():
                        loss_tmp[alpha] = 0
                    for layer_name in layer_names:
                        loss_tmp[alpha] += loss_alphas[layer_name][alpha]
                res = dict_to_list(loss_tmp)
                res.sort(key=lambda x: x[1])

                best_alpha[ln_name] = float(res[0][0])

            elif cur_shared_criterion == "min" or cur_shared_criterion == "max":
                tmp_best_alpha = []
                for layer_name in layer_names:
                    res = dict_to_list(loss_alphas[layer_name])
                    res.sort(key=lambda x: x[1])
                    tmp_best_alpha.append(float(res[0][0]))
                if cur_shared_criterion == "min":
                    best_alpha[ln_name] = min(tmp_best_alpha)
                else:
                    best_alpha[ln_name] = max(tmp_best_alpha)

            else:
                raise NotImplementedError
        return best_alpha

    def _get_one_batch_auto_loss(self, input, alpha_space, orig_best_alpha, input_maxes):
        self._change_qdq_for_auto(enable=False)
        module_names = self._get_sq_layer_names()

        forward_wrapper(self.model, input, self.device)  ##disable quant and get fp32 output

        fp32_output = {}
        for name in module_names:
            module = get_module(self.model, name)
            fp32_output[name] = module.output
            module.output = None
        self._change_qdq_for_auto(enable=True)
        absorb_input_scales, weight_scales = self._cal_scales(
            self.absorb_to_layer, input_maxes, orig_best_alpha, tuning=True
        )
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        forward_wrapper(self.model, input, self.device)  ##save quant_input
        for mod_name in module_names:  # save fp32 values
            mod = get_module(self.model, mod_name)
            if mod_name in self.fp32_output_val:
                self.fp32_output_val[mod_name].append(torch.norm(mod.output))
            else:
                self.fp32_output_val[mod_name] = [torch.norm(mod.output)]
            del mod

        loss_alphas = {}
        for name in module_names:
            module = get_module(self.model, name)
            loss = self._get_auto_loss(fp32_output[name], module.output)
            cur_alpha = orig_best_alpha
            if isinstance(orig_best_alpha, dict):
                cur_alpha = orig_best_alpha[name]
            key_name = str(cur_alpha)
            loss_alphas[name] = {key_name: loss}
        # for name in module_names:
        #     loss_alphas[name]={}
        for alpha in alpha_space:
            absorb_input_scales, weight_scales = self._cal_scales(self.absorb_to_layer, input_maxes, alpha, tuning=True)
            self._update_scales_for_auto(absorb_input_scales, weight_scales)
            for name in module_names:
                losses = loss_alphas[name]
                if str(alpha) in losses.keys():
                    continue
                module = get_module(self.model, name)
                output = module.q_dq_forward(module.q_input, module.input_scale, module.weight_scale)
                loss = self._get_auto_loss(fp32_output[name], output)
                loss_alphas[name][str(alpha)] = loss
        return loss_alphas

    def _get_one_batch_auto_loss_blockwise(self, input, alpha_space, orig_best_alpha, input_maxes):
        self._change_qdq_for_auto(enable=False)
        module_names = self._get_sq_layer_names()

        block_modules = {}
        for key in self.block_names:
            block_modules[key] = get_module(self.model, key)
        self._add_blockwise_observer(block_modules)

        forward_wrapper(self.model, input, self.device)  ##disable quant and get fp32 output

        fp32_output = {}
        for block_name in self.block_names:
            fp32_output[block_name] = self.block_outputs[block_name]
        self._change_qdq_for_auto(enable=True)
        absorb_input_scales, weight_scales = self._cal_scales(
            self.absorb_to_layer, input_maxes, orig_best_alpha, tuning=True
        )
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        forward_wrapper(self.model, input, self.device)  ##save quant_input
        for mod_name in module_names:  # save fp32 values
            mod = get_module(self.model, mod_name)
            if mod_name in self.fp32_output_val:
                self.fp32_output_val[mod_name].append(torch.norm(mod.output))
            else:
                self.fp32_output_val[mod_name] = [torch.norm(mod.output)]
            del mod

        loss_alphas = {}

        for block_name in self.block_names:
            block = get_module(self.model, block_name)
            loss = self._get_auto_loss(fp32_output[block_name], self.block_outputs[block_name])
            cur_alpha = orig_best_alpha
            if isinstance(orig_best_alpha, dict):
                cur_alpha = orig_best_alpha[self.block_to_module[block_name][0]]
            key_name = str(cur_alpha)
            loss_alphas[block_name] = {key_name: loss}
        # for name in module_names:
        #     loss_alphas[name]={}
        for alpha in alpha_space:
            absorb_input_scales, weight_scales = self._cal_scales(self.absorb_to_layer, input_maxes, alpha, tuning=True)
            self._update_scales_for_auto(absorb_input_scales, weight_scales)

            for block_name in self.block_names:
                losses = loss_alphas[block_name]
                if str(alpha) in losses.keys():
                    continue
                block = get_module(self.model, block_name)
                block_copy = copy.deepcopy(block)
                for name in self.block_to_module[block_name]:
                    if name == block_name and len(self.block_to_module[block_name]) == 1:
                        module, module_copy = block, block_copy
                    else:
                        module = get_module(block, name)
                        module_copy = copy.deepcopy(module)
                    if module.weight_scale is not None:
                        module_copy.orig_layer.weight *= module.weight_scale
                    q_dq_weight = quant_dequant_w(module_copy.orig_layer)
                    module_copy.orig_layer.weight.data.copy_(q_dq_weight)
                    module_copy.do_blockwise = True
                    if not (name == block_name and len(self.block_to_module[block_name]) == 1):
                        set_module(block_copy, name, module_copy)
                try:
                    output = block_copy(self.block_inputs[block_name])[0]
                except:  # Llama model decoder_layer forward requires position_id
                    position_ids = torch.arange(self.block_inputs[block_name].size()[1])
                    position_ids = position_ids.view(self.block_inputs[block_name].size()[0], -1)
                    output = block_copy(self.block_inputs[block_name], position_ids=position_ids)[0]
                loss = self._get_auto_loss(fp32_output[block_name], output)
                loss_alphas[block_name][str(alpha)] = loss
                del block_copy  # release memory
        return loss_alphas

    def opwise_rank(self, loss_alphas, best_alphas):
        max_op, max_ratio, max_key = "", 0, ""
        ratio_info = {}
        for key in self.absorb_to_layer:
            for op_name in self.absorb_to_layer[key]:
                fp32_norm, loss_ = (
                    torch.sum(torch.stack(self.fp32_output_val[op_name])),
                    loss_alphas[op_name][str(best_alphas[key])],
                )
                ratio = loss_ / fp32_norm
                max_op = op_name if ratio > max_ratio else max_op
                max_key = key if ratio > max_ratio else max_key
                max_ratio = max(ratio, max_ratio)
                ratio_info[op_name] = ratio
                logger.debug(
                    f"final loss: {op_name}: {loss_}; @alpha {best_alphas[key]}; \
                    fp32_output norm: {fp32_norm}; ratio: {ratio}"
                )
        import operator

        ratio_info = dict(sorted(ratio_info.items(), key=operator.itemgetter(1), reverse=True))
        for key in list(ratio_info.keys()):
            logger.debug(f"sorted opname-ratio: {key}:  {ratio_info[key]}")
        if max_op != "":
            logger.debug(
                f"max loss: {max_op}: {loss_alphas[max_op][str(best_alphas[max_key])]} @alpha {best_alphas[max_key]}\
                fp32_output norm: {torch.sum(torch.stack(self.fp32_output_val[max_op]))}; ratio: {max_ratio}"
            )
        return None

    def default_tune_setup(self):
        round_num = max(  # Initialize the alpha search space
            len(str(self.alpha_min).split(".")[1]),
            len(str(self.alpha_max).split(".")[1]),
            len(str(self.alpha_step).split(".")[1]),
        )
        self.alpha_space = numpy.round(
            numpy.arange(self.alpha_min, self.alpha_max + self.alpha_step, self.alpha_step), round_num
        ).tolist()
        ##wrapper new module
        self._qdq_model_wrapper_for_auto(save_q_input=True)

        # model scale calculation and update  -- lyt
        absorb_input_scales, weight_scales = self._cal_scales(
            self.absorb_to_layer, self.input_maxes_abs, self.default_alpha, tuning=True
        )
        self._update_scales_for_auto(absorb_input_scales, weight_scales)
        return absorb_input_scales, weight_scales

    def _auto_tune_alpha(self):
        """Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly."""
        logger.info("Start alpha tuning")

        absorb_input_scales, weight_scales = self.default_tune_setup()

        total_cnt, tmp_cnt = 0, 0
        alpha_update_iter, tune_cnt = 0, 4
        # multiply_factor is used to combine samples to calib_sample_num // 4 before summarizing the best alpha
        multiply_factor = (
            self.calib_sample_num // tune_cnt if self.calib_sample_num >= tune_cnt else self.calib_sample_num
        )
        self.fp32_output_val = {}
        best_alphas = self.default_alpha

        if not self.dataloader:
            logger.info(f"Auto-tuning failed due to no dataloader, using {best_alphas} instead.")
            self._qdq_model_unwrapper_for_auto()
            return best_alphas
        bar = tqdm(self.dataloader, total=self.calib_sample_num, desc="auto tune alpha")
        try:
            for input, label in bar:
                loss_alphas = {}
                best_alphas_per_module = best_alphas
                if isinstance(best_alphas, dict):
                    for key in self.absorb_to_layer.keys():
                        layer_names = self.absorb_to_layer[key]
                        for layer_name in layer_names:
                            best_alphas_per_module[layer_name] = best_alphas_per_module[key]
                loss_tmp = self._get_one_batch_auto_loss(
                    input, self.alpha_space, best_alphas_per_module, self.input_maxes_abs
                )
                if loss_alphas == {}:
                    loss_alphas = loss_tmp
                else:
                    for key in loss_alphas.keys():
                        cur_loss = loss_alphas[key]
                        for alpha_key in cur_loss.keys():
                            cur_loss[alpha_key] += loss_tmp[key][alpha_key]
                total_cnt += self.dataloader.batch_size
                tmp_cnt += self.dataloader.batch_size
                if tmp_cnt // multiply_factor >= 1:
                    alpha_update_iter += 1
                    tmp_cnt = 0
                    best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
                    for key in best_alphas.keys():
                        logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                    absorb_input_scales, weight_scales = self._cal_scales(
                        self.absorb_to_layer, self.input_maxes_abs, best_alphas, tuning=True
                    )
                    self._update_scales_for_auto(absorb_input_scales, weight_scales)
                    # does not need to reset the weight_scale_dict, because use the weight of ori_layer, no change
                    # self.weight_scale_dict = {}
                if total_cnt >= self.calib_sample_num:
                    break
        except:
            for input in bar:
                loss_alphas = {}
                best_alphas_per_module = best_alphas
                if isinstance(best_alphas, dict):
                    for key in self.absorb_to_layer.keys():
                        layer_names = self.absorb_to_layer[key]
                        for layer_name in layer_names:
                            best_alphas_per_module[layer_name] = best_alphas_per_module[key]

                loss_tmp = self._get_one_batch_auto_loss(
                    input, self.alpha_space, best_alphas_per_module, self.input_maxes_abs
                )
                if loss_alphas == {}:
                    loss_alphas = loss_tmp
                else:
                    for key in loss_alphas.keys():
                        cur_loss = loss_alphas[key]
                        for alpha_key in cur_loss.keys():
                            cur_loss[alpha_key] += loss_tmp[key][alpha_key]
                total_cnt += self.dataloader.batch_size
                tmp_cnt += self.dataloader.batch_size
                if tmp_cnt // multiply_factor >= 1:
                    alpha_update_iter += 1
                    tmp_cnt = 0

                    best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
                    for key in best_alphas.keys():
                        logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                    absorb_input_scales, weight_scales = self._cal_scales(
                        self.absorb_to_layer, self.input_maxes_abs, best_alphas, tuning=True
                    )
                    self._update_scales_for_auto(absorb_input_scales, weight_scales)
                    # self.weight_scale_dict = {}
                if total_cnt >= self.calib_sample_num:
                    break

        best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, shared_criterion)
        for key in best_alphas.keys():
            logger.info(f"Final alpha {key}:{best_alphas[key]}")

        self.opwise_rank(loss_alphas, best_alphas)
        self._qdq_model_unwrapper_for_auto()
        logger.info("auto tuning done")

        return best_alphas

    def _auto_tune_alpha_blockwise(self):
        """Perform blockwise-alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly."""
        logger.info("Start block-wise alpha tuning")

        self.block_names = self.block_info["block_names"]
        self.block_to_module = self.block_info["block_to_module"]
        self.block_inputs, self.block_outputs = {}, {}

        absorb_input_scales, weight_scales = self.default_tune_setup()

        total_cnt, tmp_cnt = 0, 0
        alpha_update_iter, tune_cnt = 0, 4
        # multiply_factor is used to combine samples to calib_sample_num // 4 before summarizing the best alpha
        multiply_factor = (
            self.calib_sample_num // tune_cnt if self.calib_sample_num >= tune_cnt else self.calib_sample_num
        )
        self.fp32_output_val = {}
        best_alphas = self.default_alpha

        if not self.dataloader:
            logger.info(f"Auto-tuning failed due to no dataloader, using {best_alphas} instead.")
            self._qdq_model_unwrapper_for_auto()
            return best_alphas
        bar = tqdm(self.dataloader, total=self.calib_sample_num, desc="auto tune alpha")
        try:
            for input, label in bar:
                loss_alphas = {}
                best_alphas_per_module = best_alphas
                if isinstance(best_alphas, dict):
                    for key in self.absorb_to_layer.keys():
                        layer_names = self.absorb_to_layer[key]
                        for layer_name in layer_names:
                            best_alphas_per_module[layer_name] = best_alphas_per_module[key]
                loss_tmp = self._get_one_batch_auto_loss_blockwise(
                    input, self.alpha_space, best_alphas_per_module, self.input_maxes_abs
                )
                if loss_alphas == {}:
                    for block_name in self.block_names:
                        for key in self.block_to_module[block_name]:
                            loss_alphas[key] = loss_tmp[block_name]
                else:
                    for block_name in self.block_names:
                        for key in self.block_to_module[block_name]:
                            cur_loss = loss_alphas[key]
                            for alpha_key in cur_loss.keys():
                                cur_loss[alpha_key] += loss_tmp[block_name][alpha_key]

                total_cnt += self.dataloader.batch_size
                tmp_cnt += self.dataloader.batch_size
                if tmp_cnt // multiply_factor >= 1:
                    alpha_update_iter += 1
                    tmp_cnt = 0
                    best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
                    for key in best_alphas.keys():
                        logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                    absorb_input_scales, weight_scales = self._cal_scales(
                        self.absorb_to_layer, self.input_maxes_abs, best_alphas, tuning=True
                    )
                    self._update_scales_for_auto(absorb_input_scales, weight_scales)
                    # does not need to reset the weight_scale_dict, because use the weight of ori_layer, no change
                    # self.weight_scale_dict = {}
                if total_cnt >= self.calib_sample_num:
                    break
        except:
            for input in bar:
                loss_alphas = {}
                best_alphas_per_module = best_alphas
                if isinstance(best_alphas, dict):
                    for key in self.absorb_to_layer.keys():
                        layer_names = self.absorb_to_layer[key]
                        for layer_name in layer_names:
                            best_alphas_per_module[layer_name] = best_alphas_per_module[key]

                loss_tmp = self._get_one_batch_auto_loss_blockwise(
                    input, self.alpha_space, best_alphas_per_module, self.input_maxes_abs
                )
                if loss_alphas == {}:
                    for block_name in self.block_names:
                        for key in self.block_to_module[block_name]:
                            loss_alphas[key] = loss_tmp[block_name]
                else:
                    for block_name in self.block_names:
                        for key in self.block_to_module[block_name]:
                            cur_loss = loss_alphas[key]
                            for alpha_key in cur_loss.keys():
                                cur_loss[alpha_key] += loss_tmp[block_name][alpha_key]

                total_cnt += self.dataloader.batch_size
                tmp_cnt += self.dataloader.batch_size
                if tmp_cnt // multiply_factor >= 1:
                    alpha_update_iter += 1
                    tmp_cnt = 0

                    best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
                    for key in best_alphas.keys():
                        logger.info(f"Auto alpha update iter: {alpha_update_iter}, {key}: {best_alphas[key]}")
                    absorb_input_scales, weight_scales = self._cal_scales(
                        self.absorb_to_layer, self.input_maxes_abs, best_alphas, tuning=True
                    )
                    self._update_scales_for_auto(absorb_input_scales, weight_scales)
                    # self.weight_scale_dict = {}
                if total_cnt >= self.calib_sample_num:
                    break

        best_alphas = self._get_best_alpha(self.absorb_to_layer, loss_alphas, self.shared_criterion)
        for key in best_alphas.keys():
            logger.info(f"Final alpha {key}:{best_alphas[key]}")

        self.opwise_rank(loss_alphas, best_alphas)
        self._qdq_model_unwrapper_for_auto()
        logger.info("block-wise auto tuning done")

        return best_alphas
