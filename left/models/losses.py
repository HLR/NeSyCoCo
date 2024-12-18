#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/18/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.
import json
import collections
import torch
import torch.nn.functional as F
import jacinle
import jactorch.nn as jacnn
import torch.nn as nn
import numpy as np
import random
from concepts.dsl.tensor_value import TensorValue

from typing import Optional, Sequence, List, Dict

from concepts.benchmark.common.vocab import Vocab
from left.nn.losses import MultitaskLossBase
from left.models.reasoning.reasoning import NCOneTimeComputingGrounding

__all__ = ['RefExpLoss', 'AttrClsLoss', 'QALoss', 'PickPlaceLoss', 'ReaSCANAttrClsLoss', 'ReaSCANAttrClsLossV2']


class RefExpLossV2(jacnn.TorchApplyRecorderMixin, MultitaskLossBase):
    def __init__(self, add_supervision=True, softmax=False, one_hot=False, context_objects_only=False):
        super().__init__()
        self.add_supervision = add_supervision
        self.softmax = softmax
        self.one_hot = one_hot
        self.bce = nn.BCELoss()

    def forward(self, input, target, input_objects_length):
        monitors = dict()
        outputs = dict()

        batch_size = len(input)
        loss, acc, acc_instance = [], [], []
        for i in range(batch_size):
            this_input = input[i]
            this_target = target[i]
            # Softmax
            if isinstance(this_input, TensorValue):
                this_input = this_input.tensor 
            if this_input is None:
                continue           
            if this_input.size() == torch.Size([]):
                continue
            if this_input.size(0) <= this_target:
                continue

            
            try:
             
                # this_loss = self._batched_xent_loss(this_input, this_target)
                temp_targets = torch.zeros_like(this_input)
                temp_targets[this_target.item()] = 1
                this_loss = self.bce(this_input, temp_targets)
                a = float(torch.argmax(this_input) == this_target)
                ai = (this_input[this_target] > 0).float()
                loss.append(this_loss)
                acc.append(a)
                acc_instance.append(ai)
            
            # TODO(Jiayuan Mao @ 2023/04/05): document what's this except thing trying to catch...
            except Exception as e:
                pass
                loss.append(10)
                acc.append(0)
                acc_instance.append(0)
                # raise Exception(f"Error in loss calculation for index: {i}", i)

        avg_loss = sum(loss) / len(loss) if len(loss) != 0 else 0.0
        avg_acc = sum(acc) / len(acc) if len(acc) != 0 else 0.0
        avg_ai = sum(acc_instance) / len(acc_instance) if len(acc_instance) != 0 else 0.0
        if self.training and self.add_supervision:
            monitors['loss/refexp'] = avg_loss

        if self.training:
            monitors['acc/refexp'] = avg_acc
            monitors['acc/refexp/instance'] = avg_ai
        else:
            monitors['validation/acc/refexp'] = avg_acc
            monitors['validation/acc/refexp/instance'] = avg_ai

        return monitors, outputs

class RefExpLoss(jacnn.TorchApplyRecorderMixin, MultitaskLossBase):
    def __init__(self, add_supervision=True, softmax=False, one_hot=False, context_objects_only=False):
        super().__init__()
        self.add_supervision = add_supervision
        self.softmax = softmax
        self.one_hot = one_hot

    def forward(self, input, target, input_objects_length):
        monitors = dict()
        outputs = dict()

        batch_size = len(input)
        loss, acc, acc_instance = [], [], []
        for i in range(batch_size):
            this_input = input[i]
            this_target = target[i]

            # Softmax
            if isinstance(this_input, TensorValue):
                this_input = this_input.tensor 
            if this_input is None:
                continue           
            if this_input.size() == torch.Size([]):
                continue
            if this_input.size(0) <= this_target:
                continue
            try:
                this_loss = self._batched_xent_loss(this_input, this_target)
                a = float(torch.argmax(this_input) == this_target)
                ai = (this_input[this_target] > 0).float()

                loss.append(this_loss)
                acc.append(a)
                acc_instance.append(ai)
            # TODO(Jiayuan Mao @ 2023/04/05): document what's this except thing trying to catch...
            except:
                continue

        avg_loss = sum(loss) / len(loss) if len(loss) != 0 else 0.0
        avg_acc = sum(acc) / len(acc) if len(acc) != 0 else 0.0
        avg_ai = sum(acc_instance) / len(acc_instance) if len(acc_instance) != 0 else 0.0
        if self.training and self.add_supervision:
            monitors['loss/refexp'] = avg_loss

        if self.training:
            monitors['acc/refexp'] = avg_acc
            monitors['acc/refexp/instance'] = avg_ai
        else:
            monitors['validation/acc/refexp'] = avg_acc
            monitors['validation/acc/refexp/instance'] = avg_ai

        return monitors, outputs


class AttrClsLoss(MultitaskLossBase):
    def __init__(self, add_supervision=False):
        super().__init__()
        self.add_supervision = add_supervision

    def forward(self, feed_dict, f_sng, attribute_class_to_idx, idx_to_class):
        outputs, monitors = dict(), dict()

        objects = [f['attribute'] for f in f_sng]
        all_f = torch.stack(objects)
        object_labels = feed_dict['input_objects_class']

        all_scores = []
        attribute_concepts = list(attribute_class_to_idx.keys())
        attribute_concepts.sort()
        for concept in attribute_concepts:
            this_score = all_f[:, :, attribute_class_to_idx[concept]]
            all_scores.append(this_score)
        all_scores = torch.stack(all_scores, dim=-1)

        accs, losses = [], []
        concepts_to_accs, concepts_to_pred_concepts = [], []
        for b in range(object_labels.size(0)):
            gt_concepts_to_accs = collections.defaultdict(list)
            gt_concepts_to_pred_concepts = collections.defaultdict(list)
            for i in range(object_labels.size(1)):
                gt_label = int(object_labels[b, i].cpu().numpy())
                gt_class = idx_to_class[gt_label].replace(' ', '_') + '_Object'

                if gt_class in attribute_concepts:
                    gt_class_index = attribute_concepts.index(gt_class)
                else:
                    continue

                pred_scores_for_object = all_scores[b, i, :]
                pred_max_class_index = int(torch.argmax(pred_scores_for_object).cpu().numpy())
                pred_class = attribute_concepts[pred_max_class_index]
                gt_concepts_to_pred_concepts[gt_class].append(pred_class)

                this_acc = float(pred_max_class_index == gt_class_index)
                accs.append(this_acc)
                gt_concepts_to_accs[gt_class].append(this_acc)

                this_loss = self._sigmoid_xent_loss(pred_scores_for_object, torch.tensor(gt_class_index).cuda())
                losses.append(this_loss)

            concepts_to_accs.append(gt_concepts_to_accs)
            concepts_to_pred_concepts.append(gt_concepts_to_pred_concepts)

        avg_acc = sum(accs) / len(accs) if len(accs) != 0 else 0.0
        avg_loss = sum(losses) / len(losses) if len(losses) != 0 else 0.0

        outputs['concepts_to_accs'] = concepts_to_accs
        outputs['concepts_to_pred_concepts'] = concepts_to_pred_concepts

        if self.training and self.add_supervision:
            monitors['loss/attrcls'] = avg_loss

        if self.training:
            monitors['acc/attrcls'] = avg_acc
            monitors['train/acc/attrcls'] = avg_acc
        else:
            monitors['validation/acc/attrcls'] = avg_acc

        return monitors, outputs

class ReaSCANAttrClsLoss(MultitaskLossBase):
    def __init__(self, add_supervision=False):
        super().__init__()
        self.add_supervision = add_supervision
        self.bce_loss = jacnn.BCEWithLogitsLoss()

    def forward(self, feed_dict, f_sng, outputs, left_model):
        outputs, monitors = dict(), dict()
        attribute_class_to_idx = left_model.attribute_embedding.concept2id
        idx_to_class = left_model.attribute_embedding.concepts
        for index in range(len(feed_dict['input_command'])): # batch size
            situation_dict = json.loads(feed_dict['situation_dict'][index])
            objects = [f["attribute"] for f in f_sng]
            all_f = torch.stack(objects)

            all_scores = []
            attribute_concepts = list(attribute_class_to_idx.keys())
            attribute_concepts.sort()
            for concept in attribute_concepts:
                this_score = all_f[:, :, attribute_class_to_idx[concept]]
                all_scores.append(this_score)
            all_scores = torch.stack(all_scores, dim=-1)

            accs, losses = [], []
            concepts_to_accs, concepts_to_pred_concepts = [], []
            object_index_list = []
            for object_info in situation_dict['placed_objects'].values():
                gt_concepts_to_accs = collections.defaultdict(list)
                gt_concepts_to_pred_concepts = collections.defaultdict(list)
                for concept in ["shape","color"]:
                    object_index = int(object_info["position"]["row"]) * 6 + int(object_info["position"]["column"])
                    object_index_list.append(object_index)
                    gt_class_name = object_info["object"][concept]
                    gt_class = gt_class_name + '_Object'
                    if gt_class in attribute_concepts:
                        gt_class_index = attribute_concepts.index(gt_class)
                    else:
                        continue

                    pred_scores_for_object = all_scores[index, object_index, :]
                    ## Loss calculation
                    this_loss = self._sigmoid_xent_loss(pred_scores_for_object, torch.tensor(gt_class_index).cuda())
                    losses.append(this_loss)
                    ## Accuracy calculation
                    ## Normalize pred_scores_for_object
                    pred_scores_for_object = pred_scores_for_object - pred_scores_for_object.min()
                    pred_scores_for_object = pred_scores_for_object / pred_scores_for_object.max()
                    ## Choose the top-4 high probability labels which are more than 0.7
                    pred_scores_for_object = list(enumerate(pred_scores_for_object.detach().cpu().numpy()))
                    pred_scores_for_object.sort(key=lambda x: x[1], reverse=True)
                    high_prob_indices = [x[0] for x in pred_scores_for_object if x[1] > 0.7]
                    high_prob_indices = high_prob_indices[:3]
                    this_acc = float(gt_class_index in high_prob_indices)

                    accs.append(this_acc)
                    gt_concepts_to_accs[gt_class].append(this_acc)

                    
                concepts_to_accs.append(gt_concepts_to_accs)
                concepts_to_pred_concepts.append(gt_concepts_to_pred_concepts)

            for cell_index in range(36):
                if cell_index not in object_index_list:
                    ## calculate loss that makes pred_scores_for_object to be all zero for the cells where no object is placed
                    for concept in ["shape","color"]:
                        pred_scores_for_object = all_scores[index, cell_index, :]
                        this_loss = self.bce_loss(pred_scores_for_object, torch.zeros_like(pred_scores_for_object))
                        losses.append(this_loss)                                      
            avg_acc = sum(accs) / len(accs) if len(accs) != 0 else 0.0
            avg_loss = sum(losses) / len(losses) if len(losses) != 0 else 0.0

            outputs['concepts_to_accs'] = concepts_to_accs
            outputs['concepts_to_pred_concepts'] = concepts_to_pred_concepts

            if self.training and self.add_supervision:
                monitors['loss/attrcls'] = avg_loss

            if self.training:
                monitors['acc/attrcls'] = avg_acc
                monitors['train/acc/attrcls'] = avg_acc
            else:
                monitors['validation/acc/attrcls'] = avg_acc

        return monitors, outputs

class ReaSCANAttrClsLossV2(MultitaskLossBase):
    def __init__(self, add_supervision=False):
        super().__init__()
        self.add_supervision = add_supervision
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, feed_dict, f_sng, outputs, left_model):
        monitors = dict()
        attribute_class_to_idx = left_model.attribute_embedding.concept2id

        accs, losses = [], []
        concepts_to_accs, concepts_to_pred_concepts = [], []
        for index in range(len(feed_dict['input_command'])): # batch size
            situation_dict = json.loads(feed_dict['situation_dict'][index])
            all_scores = outputs["groundings"][index].compute_all_similarity("attribute")
            placed_objs = situation_dict['placed_objects'].values()
            placed_objs = sorted(placed_objs, key=lambda obj: int(obj["position"]["row"]) * 6 + int(obj["position"]["column"]))

            for object_index, (object_info) in enumerate(placed_objs):
                gt_concepts_to_accs = collections.defaultdict(list)
                gt_concepts_to_pred_concepts = collections.defaultdict(list)
                for concept in ["shape","color"]:
                    gt_class_name = object_info["object"][concept]
                    gt_class = gt_class_name + '_Object'
                    if gt_class in attribute_class_to_idx:
                        gt_class_index = attribute_class_to_idx[gt_class]
                    else:
                        continue
                    pred_scores_for_object = all_scores[object_index]
                    this_loss = 1 - F.softmax(pred_scores_for_object, dim=0)[gt_class_index]
                    # this_loss = self._sigmoid_xent_loss(pred_scores_for_object, torch.tensor(gt_class_index).cuda())
                    losses.append(this_loss)
                    ## Accuracy calculation
                    ## Normalize pred_scores_for_object
                    pred_scores_for_object = pred_scores_for_object - pred_scores_for_object.min()
                    pred_scores_for_object = pred_scores_for_object / pred_scores_for_object.max()
                    ## Choose the top-4 high probability labels which are more than 0.7
                    pred_scores_for_object = list(enumerate(pred_scores_for_object.detach().cpu().numpy()))
                    pred_scores_for_object.sort(key=lambda x: x[1], reverse=True)
                    high_prob_indices = [x[0] for x in pred_scores_for_object if x[1] > 0.7]
                    high_prob_indices = high_prob_indices[:3]
                    this_acc = float(gt_class_index in high_prob_indices)

                    accs.append(this_acc)
                    gt_concepts_to_accs[gt_class].append(this_acc)

                    
                concepts_to_accs.append(gt_concepts_to_accs)
                concepts_to_pred_concepts.append(gt_concepts_to_pred_concepts)
                                    
        avg_acc = sum(accs) / len(accs) if len(accs) != 0 else 0.0
        avg_loss = sum(losses) / len(losses) if len(losses) != 0 else 0.0

        outputs['concepts_to_accs'] = concepts_to_accs
        outputs['concepts_to_pred_concepts'] = concepts_to_pred_concepts

        if self.training and self.add_supervision:
            monitors['loss/attrcls'] = avg_loss

        if self.training:
            monitors['acc/attrcls'] = avg_acc
            monitors['train/acc/attrcls'] = avg_acc
        else:
            monitors['validation/acc/attrcls'] = avg_acc

        return monitors, outputs


class ReaSCANAttrClsLossV3(MultitaskLossBase):
    def __init__(self, add_supervision=False, normalize=True):
        super().__init__()
        self.add_supervision = add_supervision
        self.normalize = normalize
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

        self.g_attribute_concepts = [
            ('color', ['red_Object', 'blue_Object', 'yellow_Object', 'green_Object']),
            ('shape', ['square_Object', 'circle_Object', 'cylinder_Object', 'box_Object']),
            ("box", ["box_Object"]),
            ("small", ["small_Object"]),
            ("big", ["big_Object"]),
        ]
        self.g_attribute_concept_ids = None
        
    def fill_attribute_concept_ids(self, attribute_class_to_idx):
        if self.g_attribute_concept_ids is not None:
            return
        self.g_attribute_concept_ids = {}
        for concept, concept_list in self.g_attribute_concepts:
            self.g_attribute_concept_ids[concept] = []
            for concept_name in concept_list:
                if concept_name in attribute_class_to_idx:
                    self.g_attribute_concept_ids[concept].append(attribute_class_to_idx[concept_name])

    def obj_attribute_loss(self, object_info, placed_objs, attribute_class_to_idx, attr_scores, object_index, concepts_to_pred_concepts):
        concept_loss = 0
        concept_accs = []
        gt_concepts_to_accs = collections.defaultdict(list)
        for concept, concept_attrs in self.g_attribute_concepts:
            if concept not in ["color", "shape"] or object_info["object"][concept] == "box":
                continue
            gt_class_name = object_info["object"][concept]
            gt_class = gt_class_name + '_Object'
            if self.g_attribute_concept_ids is None:
                self.fill_attribute_concept_ids(attribute_class_to_idx)
            concept_attr_indices = self.g_attribute_concept_ids[concept]
            
            pred_scores_for_object = attr_scores[object_index]
            ## Loss calculation
            concept_scores = pred_scores_for_object[concept_attr_indices]
            gt_class_index = torch.tensor(concept_attrs.index(gt_class)).to(concept_scores.device)

            # concept_loss += self.ce_loss(concept_scores, gt_class_index)
            temp_target = torch.zeros_like(concept_scores)
            temp_target[gt_class_index] = 1
            concept_loss += self.bce_loss(concept_scores, temp_target)
            ## Accuracy calculation
            this_acc = float(concept_scores.argmax().item() == gt_class_index.item())
            concept_accs.append(this_acc)
            gt_concepts_to_accs[gt_class].append(this_acc)
            concepts_to_pred_concepts[gt_class_name].append(this_acc)
        ### Box
        
        concept = "box"
        gt_class_name = "box"
        gt_class = gt_class_name + '_Object'
        concept_attr_indices = self.g_attribute_concept_ids[concept]
        pred_scores_for_object = attr_scores[object_index]
        ## Loss calculation
        concept_scores = pred_scores_for_object[concept_attr_indices]
        gt_class_value = 1.0 if object_info["object"]["shape"] == "box" else 0.0
        gt_class_value = torch.tensor(gt_class_value).to(concept_scores.device)
        concept_loss += self.bce_loss(concept_scores[0], gt_class_value)

        ## Accuracy calculation
        pred = concept_scores[0] >= 0.5
        this_acc = float(pred.item() == gt_class_value.item())
        concept_accs.append(this_acc)
        gt_concepts_to_accs[gt_class].append(this_acc)
        concepts_to_pred_concepts[gt_class_name].append(this_acc)

        ### Size
        size_loss = 0
        size_loss_counter = 0
        for concept in ["small", "big"]:
            gt_class_name = concept
            gt_class = gt_class_name + '_Object'
            concept_attr_indices = self.g_attribute_concept_ids[concept]
            concept_scores = attr_scores[:,concept_attr_indices].squeeze()
            ref_obj_score = concept_scores[object_index]
            ref_obj_size = int(object_info["object"]["size"])
            gt = ref_obj_size / 4
            if concept == "small":
                gt = 1 - gt
            
            size_loss += self._mse_loss(ref_obj_score, torch.tensor(gt).to(ref_obj_score.device))
            
            ## Accuracy calculation
            this_acc = 1 if (abs(ref_obj_score.item() - gt) < 0.2) else 0
            concept_accs.append(this_acc)
            gt_concepts_to_accs[gt_class].append(this_acc)
            concepts_to_pred_concepts[gt_class_name].append(this_acc)
            size_loss_counter += 1
        ## normalize the size loss
        concept_loss += size_loss / (2 * size_loss_counter)


        return concept_loss, gt_concepts_to_accs, concept_accs
    
    @staticmethod
    def _get_same_attrs(placed_objs, attr, object_info):
        if attr == "col": attr = "column"
        key = 'object' if attr not in [ "row", "column"] else 'position'
        return [obj_enum[0] for obj_enum in enumerate(placed_objs) if obj_enum[1][key][attr] == object_info[key][attr]]
    
    @staticmethod
    def _get_inside(placed_objs, object_info):
        obj_indices = []
        row_range = range(int(object_info['position']['row']), int(object_info['position']['row']) + int(object_info['object']['size']))
        col_range = range(int(object_info['position']['column']), int(object_info['position']['column']) + int(object_info['object']['size']))
        
        for obj_index, obj in enumerate(placed_objs):
            if int(obj['position']['row']) in row_range and int(obj['position']['column']) in col_range:
                obj_indices.append(obj_index)
        return obj_indices
    
    def spatial_relation_loss(self, object_info, placed_objs ,attribute_class_to_idx, all_scores, object_index, concepts_to_pred_concepts):
        concept_loss = 0
        concept_accs = []
        gt_concepts_to_accs = collections.defaultdict(list)
        for relation in ["same_color","same_row", "same_shape", "same_size", "same_col", "inside"]:
            
            if relation == "inside":
                if object_info["object"]["shape"] == "box":
                    common_objects = self._get_inside(placed_objs, object_info)
                else:
                    common_objects = []
            else:
                common_objects = self._get_same_attrs(placed_objs, relation.split('_')[1], object_info)
            gt = torch.zeros(len(placed_objs)).to(all_scores.device)
            
            gt[common_objects] = 1
            gt[object_index] = 0 
            relation_cocept_name = relation + '_Object_Object'
            relation_index = attribute_class_to_idx[relation_cocept_name]
            object_relation_scores = all_scores[object_index,:,relation_index]
            if self.normalize:
                object_relation_scores = torch.sigmoid(object_relation_scores)
            
            # Create a copy of the tensor before modifying it
            object_relation_scores_copy = object_relation_scores.clone()
            object_relation_scores_copy[object_index] = torch.sigmoid(object_relation_scores[object_index])

            # Use the modified copy for further operations
            object_relation_scores = object_relation_scores_copy


            #### The value for common_objects shuold be high the rest should be low
            concept_loss += self.bce_loss(object_relation_scores, gt)
            ## Calculate accuracy
            pred = object_relation_scores >= 0.5
            this_acc = float(((pred == gt).sum()/len(pred)).item())
            concept_accs.append(this_acc)
            concepts_to_pred_concepts[relation_cocept_name].append(this_acc)
            gt_concepts_to_accs[relation_cocept_name].append(this_acc)          
        return concept_loss, gt_concepts_to_accs, concept_accs


    def forward(self, feed_dict, f_sng, outputs, left_model, normalize=False):
        monitors = dict()
        attribute_class_to_idx = left_model.attribute_embedding.concept2id
        relation_class_to_idx = left_model.relation_embedding.concept2id

        accs, losses = [], []
        concepts_to_accs = []
        concepts_to_pred_concepts = collections.defaultdict(list)
        for index in range(len(feed_dict['input_command'])): # batch size
            situation_dict = json.loads(feed_dict['situation_dict'][index])
            attibute_scores = outputs["groundings"][index].compute_all_similarity("attribute")
            relation_scores = outputs["groundings"][index].compute_all_similarity("relation")
            if normalize:
                attibute_scores = torch.sigmoid(attibute_scores)
                relation_scores = torch.sigmoid(relation_scores)
            placed_objs = situation_dict['placed_objects'].values()
            placed_objs = sorted(placed_objs, key=lambda obj: int(obj["position"]["row"]) * 6 + int(obj["position"]["column"]))


            for object_index, (object_info) in enumerate(placed_objs):
                attr_concept_loss, attr_gt_concepts_to_accs, attr_concept_accs = self.obj_attribute_loss(object_info,placed_objs, attribute_class_to_idx, attibute_scores, object_index, concepts_to_pred_concepts)
                losses.append(attr_concept_loss)
                concepts_to_accs.append(attr_gt_concepts_to_accs)
                accs.extend(attr_concept_accs)                

                rel_concept_loss, rel_gt_concepts_to_accs, rel_concept_accs = self.spatial_relation_loss(object_info, placed_objs, relation_class_to_idx, relation_scores, object_index, concepts_to_pred_concepts)
                losses.append(rel_concept_loss)
                accs.extend(rel_concept_accs)
                concepts_to_accs.append(rel_gt_concepts_to_accs)


                                    
        avg_acc = sum(accs) / len(accs) if len(accs) != 0 else 0.0
        avg_loss = sum(losses) / len(losses) if len(losses) != 0 else 0.0

        outputs['concepts_to_accs'] = concepts_to_accs
        outputs['concepts_to_pred_concepts'] = concepts_to_pred_concepts

        if self.training and self.add_supervision:
            monitors['loss/attrcls'] = avg_loss

        if self.training:
            monitors['acc/attrcls'] = avg_acc
            monitors['train/acc/attrcls'] = avg_acc
        else:
            monitors['validation/acc/attrcls'] = avg_acc
            for concept, concept_accs in concepts_to_pred_concepts.items():
                if f'acc/attrcls/{concept}' not in monitors:
                    monitors[f'acc/attrcls/{concept}'] = 0
                    monitors[f'acc/attrcls/{concept}/n'] = 0
                monitors[f'acc/attrcls/{concept}'] += sum(concept_accs)/len(concept_accs)
                monitors[f'acc/attrcls/{concept}/n'] += len(concept_accs)

        return monitors, outputs

class QALossV2(MultitaskLossBase):
    def __init__(self, output_vocab: Optional[Vocab] = None, add_supervision: bool = True):
        super().__init__()
        self.output_vocab = output_vocab
        self.add_supervision = add_supervision
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.bcew_loss = nn.BCEWithLogitsLoss()

    def forward(self, execution_results, groundtruth, question_types):
        cut_off_value = 0.2
        monitors, outputs = collections.defaultdict(list), dict()
        outputs['pred_answers'] = list()

        assert len(execution_results) == len(groundtruth)
        for result, gt_answer, question_type in zip(execution_results, groundtruth, question_types):
            if result is None or result.tensor.isnan().any():
                monitors['acc/success_exec'].append(0.0)
                monitors['acc/qa'].append(0.0)
                continue
            
            try:
                result_typename = result.dtype.typename
            # TODO(Joy Hsu @ 2023/10/04): remove exceptions for tensor result type.
            except:
                result_typename = result.dtype
            if result_typename == 'bool':
                try:
                    pred_answer = 'yes' if result.tensor.item() > cut_off_value else 'no'
                except:
                    monitors['acc/success_exec'].append(0.0)
                    monitors['acc/qa'].append(0.0)
                    print("Mismatch")
                    continue
            elif result_typename == 'int64':
                try:
                    pred_answer = str(int(result.tensor.round().item()))
                except:
                    monitors['acc/success_exec'].append(0.0)
                    monitors['acc/qa'].append(0.0)
                    print("Mismatch")
                    continue
            else:
                try:
                    pred_answer = self.output_vocab.idx2word[result.tensor.argmax().item()]
                except:
                    pred_answer = self.output_vocab.idx2word[result.argmax().item()]
            outputs['pred_answers'].append(pred_answer)

            if result_typename == 'bool':
                if isinstance(gt_answer, bool):
                    #TODO Change to work with config
                    # this_loss = self._bce_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    # this_loss = self._bce_prob_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    this_loss = self.bce_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    this_accuracy = bool(result.tensor.item() > cut_off_value) == gt_answer
                    # print(f'compute loss pred={result.tensor.sigmoid().item()} gt={gt_answer} loss={this_loss.item()}')
                else:
                    this_loss, this_accuracy = 10, False
            elif result_typename == 'int64':
                if isinstance(gt_answer, int):
                    this_loss = self._mse_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    this_accuracy = pred_answer == str(gt_answer)
                else:
                    this_loss, this_accuracy = 10, False
            elif result_typename == 'Object':
                this_loss, this_accuracy = 10, False
            else:
                if isinstance(gt_answer, str):
                    temp_results = result
                    if isinstance(result, TensorValue):
                        temp_results = result.tensor
                    
                    this_loss = self.ce_loss(temp_results, torch.tensor(self.output_vocab.word2idx[gt_answer]).to(temp_results.device))
                    this_accuracy = pred_answer == gt_answer
                else:
                    this_loss, this_accuracy = 10, False

            if self.training and self.add_supervision:
                monitors['loss/qa'].append(this_loss)

            monitors['acc/success_exec'].append(1.0)
            monitors['acc/qa'].append(this_accuracy)
            monitors['acc/qa_succ_exec'].append(this_accuracy)
            monitors[f'acc/qa/{question_type}'].append(this_accuracy)

        for k, vs in list(monitors.items()):
            monitors[k + '/n'] = len(vs)
            monitors[k] = sum(vs) / len(vs) if len(vs) != 0 else 0.0

        return monitors, outputs

class QALoss(MultitaskLossBase):
    def __init__(self, output_vocab: Optional[Vocab] = None, add_supervision: bool = True):
        super().__init__()
        self.output_vocab = output_vocab
        self.add_supervision = add_supervision

    def forward(self, execution_results, groundtruth, question_types):
        monitors, outputs = collections.defaultdict(list), dict()
        outputs['pred_answers'] = list()

        assert len(execution_results) == len(groundtruth)
        for result, gt_answer, question_type in zip(execution_results, groundtruth, question_types):
            if result is None:
                monitors['acc/success_exec'].append(0.0)
                monitors['acc/qa'].append(0.0)
                continue

            try:
                result_typename = result.dtype.typename
            # TODO(Joy Hsu @ 2023/10/04): remove exceptions for tensor result type.
            except:
                result_typename = result.dtype
            if result_typename == 'bool':
                try:
                    pred_answer = 'yes' if result.tensor.item() > 0 else 'no'
                except:
                    monitors['acc/success_exec'].append(0.0)
                    monitors['acc/qa'].append(0.0)
                    print("Mismatch")
                    continue
            elif result_typename == 'int64':
                try:
                    pred_answer = str(int(result.tensor.round().item()))
                except:
                    monitors['acc/success_exec'].append(0.0)
                    monitors['acc/qa'].append(0.0)
                    print("Mismatch")
                    continue
            else:
                try:
                    pred_answer = self.output_vocab.idx2word[result.tensor.argmax().item()]
                except:
                    pred_answer = self.output_vocab.idx2word[result.argmax().item()]
            outputs['pred_answers'].append(pred_answer)

            if result_typename == 'bool':
                if isinstance(gt_answer, bool):
                    this_loss = self._bce_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    this_accuracy = bool(result.tensor.item() > 0) == gt_answer
                    # print(f'compute loss pred={result.tensor.sigmoid().item()} gt={gt_answer} loss={this_loss.item()}')
                else:
                    this_loss, this_accuracy = 10, False
            elif result_typename == 'int64':
                if isinstance(gt_answer, int):
                    this_loss = self._mse_loss(result.tensor, torch.tensor(gt_answer).float().to(result.tensor.device))
                    this_accuracy = pred_answer == str(gt_answer)
                else:
                    this_loss, this_accuracy = 10, False
            elif result_typename == 'Object':
                this_loss, this_accuracy = 10, False
            else:
                if isinstance(gt_answer, str):
                    try:
                        this_loss = self._xent_loss(result.tensor, torch.tensor(self.output_vocab.word2idx[gt_answer]).to(result.tensor.device))
                    except:
                        this_loss = self._xent_loss(result, torch.tensor(self.output_vocab.word2idx[gt_answer]).to(result.device))
                    this_accuracy = pred_answer == gt_answer
                else:
                    this_loss, this_accuracy = 10, False

            if self.training and self.add_supervision:
                monitors['loss/qa'].append(this_loss)

            monitors['acc/success_exec'].append(1.0)
            monitors['acc/qa'].append(this_accuracy)
            monitors['acc/qa_succ_exec'].append(this_accuracy)
            monitors[f'acc/qa/{question_type}'].append(this_accuracy)

        for k, vs in list(monitors.items()):
            monitors[k + '/n'] = len(vs)
            monitors[k] = sum(vs) / len(vs) if len(vs) != 0 else 0.0

        return monitors, outputs


class CLEVRConceptSupervisionLoss(MultitaskLossBase):
    def __init__(self, attribute_concepts: Dict[str, List[str]], relational_concepts: Dict[str, List[str]], add_supervision: bool = False):
        super().__init__()
        self.add_supervision = add_supervision
        self.attribute_concepts = attribute_concepts
        self.relational_concepts = relational_concepts

    def forward(self, groundings: Sequence[NCOneTimeComputingGrounding], feed_dict: jacinle.GView):
        monitors, outputs = collections.defaultdict(list), dict()

        attribute_classification_scores = collections.defaultdict(list)
        attribute_relation_scores = collections.defaultdict(list)
        relation_scores = collections.defaultdict(list)

        for grounding in groundings:
            for attr_name, concepts in self.attribute_concepts.items():
                this_attribute_classification = grounding.compute_similarities_batch('attribute', [f'{c}_Object' for c in concepts])
                this_attribute_relation = grounding.compute_similarity('relation', f'same_{attr_name}_Object_Object')

                attribute_classification_scores[attr_name].append(this_attribute_classification)
                attribute_relation_scores[attr_name].append(this_attribute_relation.reshape(-1))

            for attr_name, concepts in self.relational_concepts.items():
                this_relation = grounding.compute_similarities_batch('relation', [f'{c}_Object_Object' for c in concepts])
                relation_scores[attr_name].append(this_relation.reshape(-1, this_relation.shape[-1]))

        for k, v in attribute_classification_scores.items():
            attribute_classification_scores[k] = torch.concat(v, dim=0)
        for k, v in attribute_relation_scores.items():
            attribute_relation_scores[k] = torch.concat(v, dim=0)
        for k, v in relation_scores.items():
            relation_scores[k] = torch.concat(v, dim=0)

        loss = 0
        for k, v in attribute_classification_scores.items():
            if f'attribute_{k}' not in feed_dict:
                continue
            accuracy = (v.argmax(dim=-1) == feed_dict[f'attribute_{k}']).float().mean()
            monitors[f'acc/attrcls/{k}'] = accuracy.item()
            monitors[f'acc/attrcls/{k}/n'] = v.shape[0]
            if self.training and self.add_supervision:
                monitors[f'loss/attrcls/{k}'] = self._sigmoid_xent_loss(v, feed_dict[f'attribute_{k}']).mean()
                monitors[f'loss/attrcls/{k}/n'] = v.shape[0]
                loss += monitors[f'loss/attrcls/{k}']
        for k, v in attribute_relation_scores.items():
            if f'attribute_relation_{k}' not in feed_dict:
                continue
            accuracy = ((v > 0) == feed_dict[f'attribute_relation_{k}']).float().mean()
            monitors[f'acc/attrrel/{k}'] = accuracy.item()
            monitors[f'acc/attrrel/{k}/n'] = v.shape[0]
            if self.training and self.add_supervision:
                monitors[f'loss/attrrel/{k}'] = self._bce_loss(v, feed_dict[f'attribute_relation_{k}'].float()).mean()
                monitors[f'loss/attrrel/{k}/n'] = v.shape[0]
                loss += monitors[f'loss/attrrel/{k}']
        for k, v in relation_scores.items():
            if f'relation_{k}' not in feed_dict:
                continue
            accuracy = ((v > 0) == feed_dict[f'relation_{k}']).float().mean()
            monitors[f'acc/rel/{k}'] = accuracy.item()
            monitors[f'acc/rel/{k}/n'] = v.shape[0]
            if self.training and self.add_supervision:
                monitors[f'loss/rel/{k}'] = self._bce_loss(v, feed_dict[f'relation_{k}']).mean()
                monitors[f'loss/rel/{k}/n'] = v.shape[0]
                loss += monitors[f'loss/rel/{k}']

        if self.training and self.add_supervision:
            monitors['loss/concept_supervision'] = loss

        return monitors, outputs


class PickPlaceLoss(MultitaskLossBase):
    def __init__(self):
        super().__init__()

    def cross_entropy_with_logits(self, pred, labels, reduction='sum'):
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def forward(self, execution_results, inp_img, p0, p0_theta, p1, p1_theta, pick_attn, goal_transport):
        assert len(execution_results)==1 and len(inp_img)==1 # Assume batch size 1
        outputs, monitors = dict(), dict()

        # Pick loss
        pick_loc, place_output = execution_results[0]
        inp_img = np.array(inp_img[0])
        gt_pick_center = np.array(p0[0])
        gt_pick_theta = np.array(p0_theta[0])

        # Get label.
        theta_i = gt_pick_theta / (2 * np.pi / pick_attn.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % pick_attn.n_rotations

        label_size = inp_img.shape[:2] + (pick_attn.n_rotations,)
        label = np.zeros(label_size)
        label[gt_pick_center[0], gt_pick_center[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=pick_loc.device)

        pick_loc = pick_loc.permute(2, 0, 1)
        pick_loc = pick_loc.reshape(1, np.prod(pick_loc.shape))

        # Get loss.
        pick_loss = self.cross_entropy_with_logits(pick_loc, label)


        # Place loss
        gt_place_center = np.array(p1[0])
        gt_place_theta = np.array(p1_theta[0])

        itheta = gt_place_theta / (2 * np.pi / goal_transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % goal_transport.n_rotations

        # Get one-hot pixel label map.
        label_size = inp_img.shape[:2] + (goal_transport.n_rotations,)
        label = np.zeros(label_size)
        label[gt_place_center[0], gt_place_center[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=place_output.device)
        place_output = place_output.reshape(1, np.prod(place_output.shape))

        place_loss = self.cross_entropy_with_logits(place_output, label)
        goal_transport.iters += 1 # Check

        loss = pick_loss + place_loss

        if self.training:
            monitors['loss/pickplace'] = loss

        return monitors, outputs
