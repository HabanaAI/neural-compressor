# -*- coding: utf-8 -*-
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
"""Tuning class."""

import os
from typing import List

from neural_compressor.ux.components.config_generator.quantization_config_generator import (
    QuantizationConfigGenerator,
)
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.utils.consts import Strategies
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.json_serializer import JsonSerializer


class Tuning(Optimization):
    """Tuning class."""

    def __init__(
        self,
        optimization_data: dict,
        project_data: dict,
        dataset_data: dict,
    ) -> None:
        """Initialize Tuning class."""
        super().__init__(optimization_data, project_data, dataset_data)
        self.sampling_size = optimization_data["sampling_size"]
        self.tuning_details: TuningDetails = TuningDetails(optimization_data["tuning_details"])
        if dataset_data["template_path"]:
            self.command = ["python", dataset_data["template_path"]]

    def execute(self) -> None:
        """Execute tuning."""
        pass

    @property
    def optimization_script(self) -> str:
        """Get optimization script path."""
        return os.path.join(os.path.dirname(__file__), "tune_model.py")

    @property
    def configuration_data(self) -> dict:
        """Get configuration data for tuning config generator."""
        configuration_data: dict = super().configuration_data
        accuracy_criterion = self.tuning_details.accuracy_criterion
        configuration_data.update(
            {
                "sampling_size": self.sampling_size,
                "tuning_strategy": self.tuning_details.strategy,
                "accuracy_criterion": {
                    accuracy_criterion.type: accuracy_criterion.threshold,
                },
                "objective": self.tuning_details.objective,
                "exit_policy": self.tuning_details.exit_policy,
                "random_seed": self.tuning_details.random_seed,
            },
        )
        return configuration_data

    def generate_config(self) -> None:
        """Generate yaml config."""
        config_generator: QuantizationConfigGenerator = QuantizationConfigGenerator(
            configuration_path=self.config_path,
            data=self.configuration_data,
        )
        config_generator.generate()

    @property
    def parameters(self) -> List[str]:
        """Get optimization parameters."""
        if self.config_path is None:
            raise InternalException("Could not find path to config.")
        return [
            "--input-graph",
            self.input_graph,
            "--output-graph",
            self.output_graph,
            "--config",
            self.config_path,
            "--framework",
            self.framework,
        ]


class AccuracyCriterion(JsonSerializer):
    """Interface for accuracy criterion."""

    type: str
    threshold: float


class TuningDetails(JsonSerializer):
    """Interface for tuning details."""

    strategy: str
    accuracy_criterion: AccuracyCriterion
    objective: str
    exit_policy: dict
    random_seed: int

    def __init__(self, data: dict):
        """Initialize tuning details with data."""
        super().__init__()
        """Initialize tuning detials interface with data."""
        self.strategy = data.get("tuning_strategy", Strategies.BASIC.value)
        self.accuracy_criterion = AccuracyCriterion()
        self.accuracy_criterion.type = data.get("accuracy_criterion_type", "relative")
        self.accuracy_criterion.threshold = data.get("accuracy_criterion_threshold", 0.1)
        self.objective = data.get("objective", "performance")
        self.exit_policy = data.get("exit_policy", {"timeout": 0})
        self.random_seed = data.get("random_seed", 9527)
