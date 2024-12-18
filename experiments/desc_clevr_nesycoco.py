#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_left.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 02/20/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

from typing import Union, List, Dict

import jacinle
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.config.environ_v2 import configs, set_configs
from concepts.benchmark.clevr.clevr_constants import g_attribute_concepts, g_relational_concepts
from left.models.model import LeftModel
from left.models.losses import CLEVRConceptSupervisionLoss

logger = get_logger(__file__)


with set_configs():
    configs.model.domain = 'clevr'
    configs.model.scene_graph = '2d'
    configs.model.concept_embedding = 'vec'
    configs.train.refexp_add_supervision = True
    configs.train.attrcls_add_supervision = False
    configs.train.concept_add_supervision = False
    configs.model.normalize_scores = True
    configs.model.embedding_type = "glove"


_g_strict = False

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

        self.concept_supervision_loss = CLEVRConceptSupervisionLoss(
            g_attribute_concepts, g_relational_concepts, configs.train.concept_add_supervision
        )
        self.custom_transfer = custom_transfer
        
    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        f_sng = self.forward_sng(feed_dict)
        outputs['results'] = list()
        outputs['groundings'] = list()
        outputs['executions'] = list()
        outputs['parsings'] = list()
        outputs['execution_traces'] = list()

        for i in range(len(feed_dict.question_raw)):
            context_size = feed_dict.objects_length[i]
            trimmed_f_sng = {
                'attribute': f_sng[i]['attribute'][:context_size, :],
                'relation': f_sng[i]['relation'][:context_size, :context_size, :]
            }

            grounding = self.grounding_cls(trimmed_f_sng, self, self.training, apply_relation_mask=True, attribute_concepts={k.capitalize(): v for k, v in g_attribute_concepts.items()})
            outputs['groundings'].append(grounding)
            with self.executor.with_grounding(grounding):
                question = feed_dict.question_raw[i]
                parsing, program, execution, trace = None, None, None, None
                if question in self.parses:

                    raw_parsing = self.parses[question]
                    if isinstance(raw_parsing, list):
                        raw_parsing = raw_parsing[0]

                    try:
                        try:
                            parsing = self.parser.parse_expression(raw_parsing)
                            program = parsing
                            # print(repr(program))
                        except Exception as e:
                            raise ExecutionFailed('Parsing failed for question: {}.'.format(question)) from e

                        try:
                            if not self.training:
                                with self.executor.record_execution_trace() as trace_getter:
                                    execution = self.executor.execute(program)
                                    trace = trace_getter.get()
                            else:
                                execution = self.executor.execute(program)
                        except (KeyError, AttributeError, AssertionError, IndexError, ValueError) as e:
                            logger.exception('Execution failed for question: {}\nProgram: {}.'.format(question, program))
                            raise ExecutionFailed('Execution failed for question: {}\nProgram: {}.'.format(question, program)) from e
                    except (ExecutionFailed, RuntimeError) as e:
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
                else:
                    print('No parse for question: {}'.format(question))                    

            outputs['results'].append((parsing, program, execution))
            outputs['executions'].append(execution)
            outputs['parsings'].append(parsing)
            outputs['execution_traces'].append(trace)

        if not self.custom_transfer:
            update_from_loss_module(monitors, outputs, self.qa_lossv2(outputs['executions'], feed_dict.answer, feed_dict.question_type))
            update_from_loss_module(monitors, outputs, self.concept_supervision_loss(outputs['groundings'], feed_dict))
        else:
            all_accuracies = list()
            if self.custom_transfer == 'ref':
                for result, groundtruth in zip(outputs['executions'], feed_dict['answer']):
                    if result is None:
                        this_accuracy = 0
                    elif result.dtype.typename != 'Object' or result.total_batch_dims != 1:
                        this_accuracy = 0
                    else:
                        this_accuracy = int(result.tensor.argmax().item() == groundtruth)
                    all_accuracies.append(this_accuracy)
            elif self.custom_transfer in ('puzzle', 'rpm'):
                for result, groundtruth in zip(outputs['executions'], feed_dict['answer']):
                    if result is None:
                        this_accuracy = 0
                    elif result.dtype.typename != 'bool' or result.total_batch_dims != 0:
                        this_accuracy = 0
                    else:
                        this_accuracy = int((result.tensor.item() > 0.6) == groundtruth)
                        # if this_accuracy == 0:
                        #     import ipdb; ipdb.set_trace()
                    all_accuracies.append(this_accuracy)
            else:
                raise NotImplementedError()
            monitors[f'acc/{self.custom_transfer}'] = sum(all_accuracies) / len(all_accuracies)
            monitors[f'acc/{self.custom_transfer}/n'] = len(all_accuracies)
        if self.training:
            loss = monitors['loss/qa']
            if configs.train.concept_add_supervision:
                loss += monitors['loss/concept_supervision']
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs

    def forward_sng(self, feed_dict):
        f_scene = self.resnet(feed_dict.image)
        f_sng = self.scene_graph(f_scene, feed_dict.objects, feed_dict.objects_length)
        f_sng = [{'attribute': sng[1], 'relation': sng[2]} for sng in f_sng]
        return f_sng


def make_model(args, domain, parses, output_vocab,custom_transfer):
    return Model(domain, parses, output_vocab, custom_transfer)


