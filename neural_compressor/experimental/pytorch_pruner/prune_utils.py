#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
import yaml

try:
    from ...conf.dotdict import DotDict
except:
    from .dot_dict import DotDict  ##TODO
from .logger import logger


def check_config(prune_config):
    assert prune_config['start_step'] >= 0, "start_step should be greater than 0"
    assert prune_config['end_step'] >= -1, "end_step should be greater than 0"
    assert prune_config['end_step'] >= prune_config['start_step'], \
        "end_step should be greater than start_step"
    assert prune_config['target_sparsity'] >= 0 and prune_config['target_sparsity'] < 1.0, \
        "begin_pruning_step should be in range [0,1)"
    assert prune_config['update_frequency_on_step'] > 0, "update_frequency_on_step should be greater than 0"
    assert prune_config['max_sparsity_ratio_per_layer'] >= 0 and prune_config['max_sparsity_ratio_per_layer'] < 1, \
        "update_frequency_on_step should be greater than 0"
    assert prune_config['prune_domain'] == "global" or prune_config['prune_domain'] == "local", \
        "only support 'global' and 'local' prune domain"
    if "x" in prune_config["pattern"]:
        pattern = prune_config["pattern"].split('_')[-1].split('x')
        N = int(pattern[0])
        M = int(pattern[1])
        assert N > 0, "N should be greater than 0"
        assert M > 0, "M should be greater than 0"
    if ":" in prune_config["pattern"]:
        pattern = prune_config["pattern"].split('_')[-1].split(':')
        N = int(pattern[0])
        M = int(pattern[1])
        assert N > 0, "N should be greater than 0"
        assert M > N, "M should be greater than N"
        max_ratio = float(N) / M
        assert prune_config['target_sparsity'] <= max_ratio, \
            "in N:M pattern, the max sparsity is N/M={}".format(max_ratio)
        prune_config['max_sparsity_ratio_per_layer'] = min(max_ratio, prune_config['max_sparsity_ratio_per_layer'])

def reset_non_value_to_default(obj, key, default):
     if isinstance(obj, dict):
        if (not key in obj.keys()) or obj[key] == None:
            return default
        else:
            return obj[key]
     else:
         if not hasattr(obj, key) or getattr(obj, key) == None:
             return default
         else:
             return getattr(obj, key)

def process_and_check_config(val):
    val = val["pruning"]['approach']['weight_compression_pytorch']
    start_step = reset_non_value_to_default(val, "start_step", 0)
    end_step = reset_non_value_to_default(val, "end_step", 0)
    not_to_prune_names = reset_non_value_to_default(val, "not_to_prune_names", [])
    prune_layer_type = reset_non_value_to_default(val, "prune_layer_type", ['Conv2d', 'Linear'])
    target_sparsity = reset_non_value_to_default(val, "target_sparsity", 0.0)  ## be care of this val
    update_frequency_on_step = int(reset_non_value_to_default(val, "update_frequency_on_step", 1))
    prune_domain = reset_non_value_to_default(val, "prune_domain", "global")
    prune_type = reset_non_value_to_default(val, "prune_type", "snip_momentum")
    sparsity_decay_type = reset_non_value_to_default(val, "sparsity_decay_type", "exp")
    max_sparsity_ratio_per_layer = reset_non_value_to_default(val, "max_sparsity_ratio_per_layer", 0.98)
    names = reset_non_value_to_default(val, "names", [])
    exclude_names = reset_non_value_to_default(val, "exclude_names", [])
    pattern = reset_non_value_to_default(val, "pattern", "tile_pattern_4x1")

    pruners_info = []
    for info in val['pruners']:
        pruner = {}
        pruner['start_step'] = reset_non_value_to_default(info, 'start_step', start_step)
        pruner['end_step'] = reset_non_value_to_default(info, 'end_step', end_step)
        pruner['not_to_prune_names'] = reset_non_value_to_default(info, 'not_to_prune_names', not_to_prune_names)
        pruner['prune_layer_type'] = reset_non_value_to_default(info, 'prune_layer_type', prune_layer_type)
        pruner['target_sparsity'] = reset_non_value_to_default(info, 'target_sparsity', target_sparsity)
        pruner['update_frequency_on_step'] = reset_non_value_to_default(info, 'update_frequency_on_step', \
                                                                 update_frequency_on_step)
        pruner['prune_domain'] = reset_non_value_to_default(info, 'prune_domain', prune_domain)
        pruner['prune_type'] = reset_non_value_to_default(info, 'prune_type', prune_type)
        pruner['sparsity_decay_type'] = reset_non_value_to_default(info, 'sparsity_decay_type', sparsity_decay_type)
        pruner['max_sparsity_ratio_per_layer'] = reset_non_value_to_default(info, 'max_sparsity_ratio_per_layer', \
                                                                 max_sparsity_ratio_per_layer)
        pruner['names'] = reset_non_value_to_default(info, 'names', names)
        pruner['exclude_names'] = reset_non_value_to_default(info, 'exclude_names',
                                                  exclude_names)
        pruner['pattern'] = reset_non_value_to_default(info, 'pattern',
                                            pattern)                                                  

        check_config(pruner)
        pruner_info = DotDict(pruner)
        pruners_info.append(pruner_info)
    return pruners_info


def process_config(config):
    if isinstance(config, str):
        try:
            with open(config, 'r') as f:
                content = f.read()
                try:
                    from .schema_check import schema
                except ImportError:
                    from ...conf.config import schema
                val = yaml.safe_load(content)
                schema.validate(val)
        except FileNotFoundError as f:
            logger.error("{}.".format(f))
            raise RuntimeError(
                "The yaml file is not exist. Please check the file name or path."
            )
        except Exception as e:
            logger.error("{}.".format(e))
            raise RuntimeError(
                "The yaml file format is not correct. Please refer to document."
            )

    elif isinstance(config, DotDict):
        val = config
    else:
        assert False, f"not supported type {config}"

    return process_and_check_config(val)


def parse_to_prune(model, config):
    """keep target pruned layers"""
    modules = {}
    if config["names"] == None or config["names"] == []:
        config["names"] = [".*"]
    for raw in config["names"]:
        try:
            pattern = re.compile(raw)
        except:
            assert False, f"regular expression match does not support {raw}"
        for name, module in filter(lambda t: pattern.search(t[0]), model.named_modules()):
            if type(module).__name__ in config["prune_layer_type"]:
                modules[name] = module
    return modules


def parse_not_to_prune(modules, config):
    """drop non pruned layers"""
    not_to_prune = config["exclude_names"]
    not_to_prune.extend(config["not_to_prune_names"])

    patterns = [re.compile(s) for s in not_to_prune]
    if len(patterns) <= 0:
        return modules
    new_module = {}
    for name in modules.keys():
        if any([p.search(name) for p in patterns]):
            continue
        new_module[name] = modules[name]
    return new_module
