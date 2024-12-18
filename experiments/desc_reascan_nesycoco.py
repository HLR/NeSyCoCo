#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_left.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/20/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
from typing import Union, List, Dict
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from reascan import ReaSCANDataset, get_json_file_name

import jacinle
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.config.environ_v2 import configs, set_configs, def_configs
from left.models.model import LeftModel
from left.models.losses import ReaSCANAttrClsLoss, ReaSCANAttrClsLossV2, ReaSCANAttrClsLossV3

g_attribute_concepts = {
    'color': ['red', 'blue', 'yellow', 'green'],
    'shape': ['square', 'circle', "box", 'cylinder'],
    'size': ['small', 'big']
}

logger = get_logger(__file__)


with def_configs():
    configs.model.learned_belong_fusion = 'min'
    configs.load_image = True
    configs.specify_objects = True
    configs.load_image_from_disk = True


with set_configs():
    configs.model.domain = 'reascan'
    configs.model.scene_graph = '2d'
    configs.model.concept_embedding = 'vec'
    configs.model.embedding_type = 'glove'
    configs.train.refexp_add_supervision = True
    configs.train.attrcls_add_supervision = True
    configs.train.concept_add_supervision = False
    configs.model.normalize_scores = True
    configs.model.concept_embedding = 'vec'


_g_strict = False
# _g_strict = True

import left.generalized_fol_executor as fol_executor
fol_executor.g_options.use_softmax_iota = False
fol_executor.g_options.use_mul_and = True
fol_executor.g_options.normalize = configs.model.normalize_scores
fol_executor.g_options.use_max_for_rel = True
fol_executor.g_options.normalize_iota = True

def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)


class ExecutionFailed(Exception):
    pass


class Model(LeftModel):
    def __init__(self, domain, parses: Dict[str, Union[str, List[str]]], output_vocab, custom_transfer=None):
        super().__init__(domain, output_vocab)
        self.parses = parses
        self.custom_transfer = custom_transfer
        self.attrcls_loss = ReaSCANAttrClsLossV3(add_supervision=configs.train.attrcls_add_supervision, normalize=not configs.model.normalize_scores)

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        f_sng = self.forward_sng(feed_dict)
        outputs['results'] = list()
        outputs['groundings'] = list()
        outputs['executions'] = list()
        outputs['parsings'] = list()
        outputs['execution_traces'] = list()
        counter = 0
        for i in range(len(feed_dict.input_command)):
            context_size = feed_dict.num_objects[i]
            trimmed_f_sng = {
                'attribute': f_sng[i].object_feature[:context_size, :],
                'relation': f_sng[i].relation_feature[:context_size, :context_size, :]
            }


            grounding = self.grounding_cls(
                trimmed_f_sng, self, self.training,
                apply_relation_mask=True, attribute_concepts={k.capitalize(): v for k, v in g_attribute_concepts.items()},
                learned_belong_fusion=configs.model.learned_belong_fusion
            )
            outputs['groundings'].append(grounding)
            with self.executor.with_grounding(grounding):
                question = feed_dict.input_command[i]

                parsing, program, execution, trace = None, None, None, None

                if question in self.parses:
                    raw_parsing = self.parses[question]
                    if isinstance(raw_parsing, list):
                        raw_parsing = raw_parsing[0]
                    try:
                        try:
                            parsing = self.parser.parse_expression(raw_parsing)
                            program = parsing
                        except Exception as e:
                            raise ExecutionFailed('Parsing failed for question: {}. {}'.format(question, e)) from e

                        try:
                            if not self.training:
                                with self.executor.record_execution_trace() as trace_getter:
                                    execution = self.executor.execute(program)
                                    trace = trace_getter.get()
                            else:
                                execution = self.executor.execute(program)
                        except (KeyError, AttributeError, AssertionError, IndexError) as e:
                            logger.exception('Execution failed for question: {}\nProgram: {}.'.format(question, program))
                            raise ExecutionFailed('Execution failed for question: {}\nProgram: {}.'.format(question, program)) from e
                    except ExecutionFailed as e:
                        counter += 1
                        if _g_strict is True:
                            raise e
                        else:
                            print('--> Execution failed for question: {}.'.format(question))
                            print('    Program: {}.'.format(raw_parsing))
                            print(jacinle.indent_text(str(e), level=2))
                        parsing, program, execution, trace = None, None, None, None

                        if isinstance(self.parses[question], list):
                            self.parses[question].remove(raw_parsing)
                            if len(self.parses[question]) == 0:
                                self.parses.pop(question)
                        else:
                            self.parses.pop(question)
            outputs['results'].append((parsing, program, execution))
            outputs['executions'].append(execution)
            outputs['parsings'].append(parsing)
            outputs['execution_traces'].append(trace)

        # update_from_loss_module(monitors, outputs, self.qa_loss(outputs['executions'], feed_dict.answer, feed_dict.question_type))
        if configs.train.attrcls_add_supervision and len(outputs) > 0:
            update_from_loss_module(monitors, outputs, self.attrcls_loss(feed_dict,f_sng,outputs, self))
        
        update_from_loss_module(monitors, outputs, self.refexp_lossv2(outputs['executions'], feed_dict.target_location,None))#int64
        if self.training:
            loss = torch.tensor(0.0).to(next(self.parameters()).device)
            loss += monitors.get('loss/refexp', 0.0)
            if configs.train.attrcls_add_supervision:
                loss += monitors.get('loss/attrcls',0.0)
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs

    def forward_sng(self, feed_dict):
        f_scene = self.resnet(feed_dict.image)
        f_sng = self.scene_graph(f_scene, feed_dict.objects.float(), feed_dict.num_objects)
        return f_sng


def make_model(args, domain, parses, output_vocab):
    return Model(domain, parses, output_vocab)
