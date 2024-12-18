#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trainval.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/26/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.
import sys
import gc
import time
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda as cuda

import json
import pickle
import numpy as np

import jacinle.io as io
from jacinle.cli.argument import JacArgumentParser
from jacinle.cli.keyboard import yes_or_no
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar
from jaclearn.mldash import MLDashClient

import jactorch
from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float

logger = get_logger(__file__)

parser = JacArgumentParser(description='')
parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--expr', default='default', metavar='S', help='experiment name')
parser.add_argument('--config', type='kv', nargs='*', metavar='CFG', help='extra config')

# training hyperparameters
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
# parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='initial learning rate')
parser.add_argument('--lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset')
parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient')
parser.add_argument('--validation-interval', type=int, default=1, metavar='N', help='validation inverval (epochs)')
parser.add_argument('--validation-visualize', type='bool', default='yes', metavar='BOOL', help='whether to visualize the results in validation')

# finetuning and snapshot
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')
parser.add_argument('--save-interval', type=int, default=10, metavar='N', help='model save interval (epochs)')
parser.add_argument('--save-best-on', type=str, default=None, help='save the best model based on a metric')

parser.add_argument('--curriculum', type=str, default='none', choices=('none', 'scene', 'program', 'all'))

# evaluation only
parser.add_argument('--evaluate', action='store_true', help='evaluate the performance of the model and exit')
parser.add_argument('--evaluate-visualization-only', action='store_true', help='only useful when --evaluate is true; exit after finishing visualization')
parser.add_argument('--evaluate-on-train', action='store_true', help='evaluate on the training set')

# evaluation on custom datasets
parser.add_argument('--evaluate-custom', choices=['ref', 'puzzle', 'rpm'], help='evaluate the performance on downstream transfer datasets and exit')

# data related
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--validation-data-dir', required=False, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--val-split', default="dev", required=False, help='validation split name')
parser.add_argument('--data-parses', type='checked_file', nargs='+', default=None, metavar='FILE', help='data parses json file')
parser.add_argument('--data-concept-match', type='checked_file', default=None, metavar='FILE', help='extra data that matches concepts in parses')
parser.add_argument('--data-tvsplit', type=float, default=0.8, metavar='N', help='train/val split ratio')
parser.add_argument('--data-retain', type=float, default=1, metavar='N', help='train/val split ratio')
parser.add_argument('--data-workers', type=int, default=16, metavar='N', help='the num of workers that input training data')

# optional additional data path
parser.add_argument('--data-json', required=False, type='checked_file', metavar='FILE', help='scenes json file; by default this will be automatically set by args.data_dir')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=True, metavar='B', help='use tensorboard or not')
parser.add_argument('--debug', action='store_true', help='entering the debug mode, suppressing all logs to disk')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()

# filenames
args.series_name = 'reascan'
args.desc_name = escape_desc_name(args.desc)

if not args.evaluate:
    args.run_name = 'trainval-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    
else:
    args.run_name = 'val-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
if args.run_name is None:
    args.run_name = "" 
if args.evaluate_visualization_only:
    assert args.evaluate, 'You can only use --evaluate-visualization-only when --evaluate is true.'
if args.evaluate_on_train:
    assert args.evaluate, 'You can only use --evaluate-on-train when --evaluate is true.'

desc = load_source(args.desc)

# NB(Jiayuan Mao @ 02/15): compatible with the old version.
if hasattr(desc, 'configs'):
    configs = desc.configs
else:
    from jacinle.config.environ_v2 import configs

if args.config is not None:
    from jacinle.config.environ_v2 import set_configs
    with set_configs():
        for c in args.config:
            c.apply(configs)

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

mldash = MLDashClient('dumps')
# from memory_profiler import profile

# @profile
def main():
    # directories
    if not args.debug:
        args.dump_dir = ensure_path(osp.join('dumps', args.series_name, args.desc_name, args.expr, args.run_name))
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.vis_dir = ensure_path(osp.join(args.dump_dir, 'visualizations'))
        args.meta_file = osp.join(args.dump_dir, 'metainfo.json')
        args.log_file = osp.join(args.dump_dir, 'log.log')
        args.meter_file = osp.join(args.dump_dir, 'meter.json')

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
        else:
            args.tb_dir = None

    if not args.debug:
        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

    if args.debug and args.use_tb:
        logger.warning('Disabling the tensorboard in the debug mode.')
        args.use_tb = False
    if args.debug:
        torch.set_anomaly_enabled(True)

    if args.evaluate and args.use_tb:
        logger.warning('Disabling the tensorboard in the evaluation mode.')
        args.use_tb = False

    if args.data_json is None:
        args.data_json = osp.join(args.data_dir, 'ReaSCAN-compositional/train.json')

    validation_split = args.val_split
    if args.validation_data_dir is not None:
        # args.validation_data_json = osp.join(args.validation_data_dir, 'ReaSCAN-compositional/dev_comp_3500.json')
        # if args.validation_data_dir and not args.validation_data_dir.endswith('.json'):
        #     args.validation_data_json = osp.join(args.validation_data_dir, 'ReaSCAN-compositional/dev.json')
        #     validation_split = "dev"
        # else:
        #     
        args.validation_data_json = desc.get_json_file_name(args.validation_data_dir, validation_split)
    
    all_parses = dict()
    if args.data_parses is not None:
        for filename in args.data_parses:
            assert osp.isfile(filename), f'File {filename} does not exist.'

            logger.info('Loading parses from {}.'.format(filename))
            if filename.endswith('.p'):
                content = io.load_pkl(filename)
            else:
                content = io.load(filename)
            all_parses.update(content)
            
    from left.domain import create_domain_from_parsing
    domain = create_domain_from_parsing(all_parses)


    from jacinle.config.g import g
    g.concept_mapping = None

    logger.critical('Total parsed sentences: {}.'.format(len(all_parses)))
    logger.critical('Domain: {}'.format(domain))
    logger.info('Number of types: {}'.format(len(domain.types)))
    logger.info('Number of functions: {}'.format(len(domain.functions)))

    logger.critical('Loading the dataset.')
    transform_images = configs.model.concept_embedding != "clip"
    # data_set_args = {
    #     "data_json": args.data_json,
    #     "all_parses": all_parses,
    #     "split": "train",
    #     "load_image": configs.load_image,
    #     "specify_objects": configs.specify_objects,
    #     "transform_images": transform_images,
    #     "load_image_from_disk": configs.load_image_from_disk,
    # }
    # train_dataset = desc.ReaSCANDataset("train", **data_set_args)  # create a dataset with 1024 samples
    # validation_dataset = desc.ReaSCANDataset(data_set_args)
    # if args.data_retain < 1: 
    #     train_dataset.data = train_dataset.data[:int(len(train_dataset.data) * args.data_retain)]
    
    if args.validation_data_dir is None:
        if not args.evaluate or args.evaluate_on_train:
            train_dataset = desc.ReaSCANDataset(args.data_json,all_parses,split="train",load_image=configs.load_image, specify_objects=configs.specify_objects,transform_images=transform_images,load_image_from_disk=configs.load_image_from_disk)  # create a dataset with 1024 samples
        validation_dataset = desc.ReaSCANDataset(args.data_json,all_parses,split="val",load_image=configs.load_image, specify_objects=configs.specify_objects,transform_images=transform_images,load_image_from_disk=configs.load_image_from_disk)
    else:
        if args.debug:
            if not args.evaluate or args.evaluate_on_train:
                train_dataset = desc.ReaSCANDataset("train",args.data_json,all_parses,load_image=configs.load_image, specify_objects=configs.specify_objects,transform_images=transform_images,load_image_from_disk=configs.load_image_from_disk, dataset_size=100,)
            validation_dataset = desc.ReaSCANDataset(validation_split,args.validation_data_json,all_parses,load_image=configs.load_image, specify_objects=configs.specify_objects,transform_images=transform_images,load_image_from_disk=configs.load_image_from_disk, dataset_size=100)

        else:
            if not args.evaluate or args.evaluate_on_train:
                train_dataset = desc.ReaSCANDataset("train", args.data_json,all_parses,load_image=configs.load_image, specify_objects=configs.specify_objects,transform_images=transform_images,load_image_from_disk=configs.load_image_from_disk)  # create a dataset with 1024 samples
            validation_dataset = desc.ReaSCANDataset(validation_split ,args.validation_data_json,all_parses,load_image=configs.load_image, specify_objects=configs.specify_objects,transform_images=transform_images,load_image_from_disk=configs.load_image_from_disk)
            if args.data_retain < 1:
                ## shuffle the data before retaining
                train_dataset.data = train_dataset.data.sample(frac=args.data_retain, random_state=1)
    if not args.evaluate or args.evaluate_on_train:
        print(len(train_dataset),len(validation_dataset))
    else:
        print(len(validation_dataset))
    

    from left.data.referit3d.vocabulary import Vocabulary
    
    vocab = Vocabulary()
    # TODO(Jiayuan Mao @ 04/23): build the model.
    logger.critical('Building the model.')
    
    model = desc.make_model(args, domain, all_parses, vocab)

    if args.use_gpu:
        model.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            # Set user_scattered because we will add a multi GPU wrapper to the dataloader. See below.
            model = JacDataParallel(model, device_ids=args.gpus, user_scattered=True).cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = True

    if hasattr(desc, 'make_optimizer'):
        logger.critical('Building customized optimizer.')
        optimizer = desc.make_optimizer(model, args.lr)
    else:
        from jactorch.optim import AdamW
        trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = AdamW(trainable_parameters, args.lr, weight_decay=configs.train.weight_decay)

    if args.acc_grad > 1:
        from jactorch.optim import AccumGrad
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))

    trainer = TrainerEnv(model, optimizer)

    parent_meta_file = None
    if args.resume:
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            args.start_epoch = extra['epoch']
            logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        raw = trainer.load_weights(args.load)
        if raw is not None:
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))
            parent_meta_file = raw['extra']['meta_file']

    if args.use_tb:
        from jactorch.train.tb import TBLogger, TBGroupMeters
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
        logger.critical('Writing tensorboard logs to: "{}".'.format(args.tb_dir))
    else:
        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    if not args.debug:
        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

        logger.critical('Initializing MLDash.')
        mldash.init(
            desc_name=args.series_name + '/' + args.desc_name,
            expr_name=args.expr,
            run_name=args.run_name,
            args=args,
            highlight_args=parser,
            configs=configs,
        )
        mldash.update(metainfo_file=args.meta_file, log_file=args.log_file, meter_file=args.meter_file, tb_dir=args.tb_dir)

        if parent_meta_file is not None:
            try:
                parent_run = io.load(parent_meta_file)['args']['run_name']
                logger.critical('Setting parent run: {}.'.format(parent_run))
                if args.evaluate:
                    mldash.update_parent(parent_run, is_master=False)
                else:
                    mldash.update_parent(parent_run, is_master=True)
            except:
                logger.exception('Exception occurred during loading metainfo.')

    if args.embed:
        from IPython import embed; embed()

    if hasattr(desc, 'customize_trainer'):
        desc.customize_trainer(trainer)

    logger.critical('Building the data loader.')
    from jactorch.data.dataloader import JacDataLoader
    if not args.evaluate or args.evaluate_on_train:
        train_dataloader = JacDataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.data_workers,pin_memory=True)
    validation_dataloader = JacDataLoader(validation_dataset,batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.data_workers,pin_memory=True)
    if args.use_gpu and args.gpu_parallel:
        from jactorch.data.dataloader import JacDataLoaderMultiGPUWrapper
        train_dataloader = JacDataLoaderMultiGPUWrapper(train_dataloader, args.gpus)
        validation_dataloader = JacDataLoaderMultiGPUWrapper(validation_dataloader, args.gpus)

    undefined_configs = configs.find_undefined_values('configs')
    if len(undefined_configs) > 0:
        logger.warning('Undefined configs: {}'.format(undefined_configs))
        if not yes_or_no('Continue the script?', default='no'):
            return

    if args.evaluate_custom is not None:
        epoch = 0
        model.eval()

        validate_epoch_custom(epoch, trainer, validation_dataloader, meters)

        if not args.debug:
            meters.dump(args.meter_file)

        if not args.debug:
            mldash.log_metric('epoch', epoch, desc=False, expr=False)
            for key, value in meters.items():
                if key.startswith('loss') or key.startswith('validation/loss'):
                    mldash.log_metric_min(key, value.avg)
            for key, value in meters.items():
                if key.startswith('acc') or key.startswith('validation/acc'):
                    mldash.log_metric_max(key, value.avg)

        logger.critical(meters.format_simple('Epoch = {}'.format(epoch), compressed=False))
        return

    if args.evaluate:
        epoch = 0

        model.eval()
        if args.evaluate_on_train:
            validate_epoch(epoch, trainer, train_dataloader, meters)
        else:
            validate_epoch(epoch, trainer, validation_dataloader, meters)

        if not args.debug:
            meters.dump(args.meter_file)

        if not args.debug:
            mldash.log_metric('epoch', epoch, desc=False, expr=False)
            for key, value in meters.items():
                if key.startswith('loss') or key.startswith('validation/loss'):
                    mldash.log_metric_min(key, value.avg)
            for key, value in meters.items():
                if key.startswith('acc') or key.startswith('validation/acc'):
                    mldash.log_metric_max(key, value.avg)

        logger.critical(meters.format_simple('Epoch = {}'.format(epoch), compressed=False))
        return
    best_val_metric = 0
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        if args.curriculum != 'none':
            train_dataset = get_curriculum_dataset(epoch, train_dataset)
            train_dataloader = JacDataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.data_workers, pin_memory=True)

            # train_dataloader = this_train_dataset.make_dataloader(args.batch_size, shuffle=True, drop_last=True, nr_workers=args.data_workers)
            # NB(Jiayuan Mao @ 2023/03/21): always use the full validation dataset.
            # validation_dataloader = this_validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)
        meters.reset()

        model.train()
        
        train_epoch(epoch, trainer, train_dataloader, meters)
        gc.collect()

        if args.validation_interval > 0 and epoch % args.validation_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                validate_epoch(epoch, trainer, validation_dataloader, meters)

        latest_parses = model.parses
        if not args.debug:
            fname = osp.join(args.dump_dir, 'latest_parses.pkl')
            io.dump(fname, latest_parses)
            logger.critical(f'Latest parses saved to "{fname}".')

        if not args.debug:
            meters.dump(args.meter_file)

        if not args.debug:
            mldash.log_metric('epoch', epoch, desc=False, expr=False)
            for key, value in meters.items():
                if key.startswith('loss') or key.startswith('validation/loss'):
                    mldash.log_metric_min(key, value.avg)
            for key, value in meters.items():
                if key.startswith('acc') or key.startswith('validation/acc'):
                    mldash.log_metric_max(key, value.avg)

        logger.critical(meters.format_simple('Epoch = {}'.format(epoch), compressed=False))

        if not args.debug:
            if epoch % args.save_interval == 0:
                fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
                trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))
            if args.save_best_on is not None:
                val_metric = float(meters[args.save_best_on].avg)
                print("val_metric", val_metric)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    fname = osp.join(args.ckpt_dir, 'best.pth')
                    trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))


g_curriculum_strategy = [
    (0, 40),
    (10, 50),
    (20, 80),
    (30, 90),
    (40, 200),
    (50, 1e9), # Last step
    (1e9, None)
]



def get_curriculum_dataset(epoch, train_dataset):
    for si, s in enumerate(g_curriculum_strategy):
        if g_curriculum_strategy[si][0] < epoch <= g_curriculum_strategy[si + 1][0]:
            max_program_size = s[1]
            before_size = len(train_dataset)
            train_dataset = train_dataset.filter_program_size_raw(max_program_size)
            after_size = len(train_dataset)
            logger.critical('Building the data loader. Curriculum = {}, before_size: {} after_size = {}.'.format(s[1], before_size, after_size))
            break
    return train_dataset

def train_epoch(epoch, trainer, train_dataloader, meters):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_dataloader)

    meters.update(epoch=epoch)

    trainer.trigger_event('epoch:before', trainer, epoch)
    train_iter = iter(train_dataloader)

    end = time.time()
    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict)
            del output_dict
            del extra_info
            step_time = time.time() - end; end = time.time()

            # TODO(Jiayuan Mao @ 04/23): normalize the loss/monitors by adding n=xxx if applicable.
            meters.update(loss=loss)
            update_meters(meters, monitors)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            # TODO(Jiayuan Mao @ 04/23): customize the logger.
            pbar.set_description(meters.format_simple(
                'Epoch {}'.format(epoch),
                {k: v for k, v in meters.val.items() if not k.startswith('validation') and k.count('/') <= 1},
                compressed=True
            ), refresh=False)
            pbar.update()

            end = time.time()
            torch.cuda.empty_cache() # clear memory

    trainer.trigger_event('epoch:after', trainer, epoch)


@jactorch.no_grad_func
def validate_epoch(epoch, trainer, val_dataloader, meters):
    end = time.time()
    total_fails = 0
    run_visualizer = False
    # if args.evaluate and not args.debug:
    #     run_visualizer = True

    import matplotlib.pyplot as plt

    from PIL import Image
    from jaclearn.visualize.html_table import HTMLTableColumnDesc, HTMLTableVisualizer
    from jaclearn.visualize.box import vis_bboxes
    from concepts.dsl.tensor_value import TensorValue

    if run_visualizer:
        visualizer = HTMLTableVisualizer(osp.join(args.vis_dir, 'evaluation'), 'Evaluation')
        visualizer.begin_html()
        visualizer_index = 0
        visualizer_total = 30

    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        for feed_dict in val_dataloader:
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            output_dict, extra_info = trainer.evaluate(feed_dict)
            for x in output_dict["results"]:
                if None in x:
                    total_fails += 1
            # TODO(Jiayuan Mao @ 04/26): compute the monitoring values.
            monitors = as_float(output_dict['monitors'])
            monitors['validation/fail'] = total_fails 
            step_time = time.time() - end; end = time.time()

            # TODO(Jiayuan Mao @ 04/23): normalize the loss/other metrics by adding n=xxx if applicable.
            update_meters(meters, monitors, prefix='validation/')
            meters.update({'time/data': data_time, 'time/step': step_time})

            if run_visualizer and visualizer_index < visualizer_total:
                for i in range(len(feed_dict['question_index'])):
                    if args.validation_data_dir is None:
                        image_filename = osp.join(args.data_image_root, feed_dict['image_filename'][i])
                    else:
                        image_filename = osp.join(args.validation_data_image_root, feed_dict['image_filename'][i])
                    image = Image.open(image_filename)

                    with visualizer.table(f'Question #{feed_dict["question_index"][i]}', [
                        HTMLTableColumnDesc('image', 'Image', 'figure', {'width': '600px'}),
                        HTMLTableColumnDesc('question', 'Question', 'text', {'width': '200px'}),
                        HTMLTableColumnDesc('answer', 'Answer', 'text'),
                        HTMLTableColumnDesc('prediction', 'Prediction', 'text'),
                        HTMLTableColumnDesc('program', 'Program', 'code', {'width': '600px'}),
                    ]):
                        fig, ax = vis_bboxes(image, feed_dict['objects_raw'][i], 'Object', add_text=False)
                        visualizer.row(**{
                            'image': fig,
                            'question': feed_dict['question_raw'][i],
                            'answer': feed_dict['answer'][i],
                            'prediction': output_dict['pred_answers'][i],
                            'program': str(output_dict['parsings'][i]),
                        })
                        plt.close()

                    with visualizer.table(f'Question #{feed_dict["question_index"][i]} (Program)', [
                        HTMLTableColumnDesc('id', 'ID', 'text', {'width': '50px'}),
                        HTMLTableColumnDesc('visualization', 'Visualization', 'figure', {'width': '600px'}),
                        HTMLTableColumnDesc('program_and_output', 'Program and Output', 'code', {'width': '600px'}),
                    ]):
                        for j, (program, output) in enumerate(output_dict['execution_traces'][i]):
                            if isinstance(output, TensorValue) and output.dtype.typename in ('bool', 'Object') and len(output.batch_variables) == 1 and output.tensor.shape[0] == len(feed_dict['objects_raw'][i]):
                                fig, ax = vis_bboxes(image, feed_dict['objects_raw'][i], 's:', add_text=True, legends=[str(round(x, 3)) for x in output.tensor.detach().cpu().tolist()])
                                visualizer.row(**{
                                    'id': j,
                                    'visualization': fig,
                                    'program_and_output': str(program) + '\n\n' + str(output),
                                })
                                plt.close()

                    print('Visualized', visualizer_index)
                    visualizer_index += 1
                    if visualizer_index >= visualizer_total:
                        break
            else:
                if args.evaluate_visualization_only:
                    break

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {} (validation)'.format(epoch),
                {k: v for k, v in meters.avg.items() if k.startswith('validation/validation') and k.count('/') <= 3},
                compressed=True
            ), refresh=False)
            pbar.update()

            end = time.time()

    if run_visualizer:
        visualizer.end_html()
        link = '<a href="viewer://{}", target="_blank">{}</a>'.format(visualizer.visdir, visualizer.visdir)
        mldash.update(run_description=link)


def update_meters(meters, monitors, prefix: str = None):
    for k in list(monitors.keys()):
        if k + '/n' in monitors:
            meters.update({k: monitors[k]}, n=monitors[k + '/n'], prefix=prefix)
            del monitors[k]
            del monitors[k + '/n']

    meters.update(monitors, prefix=prefix)


@jactorch.no_grad_func
def validate_epoch_custom(epoch, trainer, val_dataloader, meters):
    end = time.time()

    run_visualizer = False
    if args.evaluate and not args.debug:
        # run_visualizer = True
        run_visualizer = False
    if args.validation_visualize is False:
        run_visualizer = False

    import matplotlib.pyplot as plt

    from PIL import Image
    from jaclearn.visualize.html_table import HTMLTableColumnDesc, HTMLTableVisualizer
    from jaclearn.visualize.box import vis_bboxes
    from concepts.dsl.tensor_value import TensorValue

    if run_visualizer:
        visualizer = HTMLTableVisualizer(osp.join(args.vis_dir, 'evaluation'), 'Evaluation')
        visualizer.begin_html()
        visualizer_index = 0
        visualizer_total = 30

    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        for feed_dict in val_dataloader:
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            output_dict, extra_info = trainer.evaluate(feed_dict)

            # TODO(Jiayuan Mao @ 04/26): compute the monitoring values.
            monitors = as_float(output_dict['monitors'])
            step_time = time.time() - end; end = time.time()

            # TODO(Jiayuan Mao @ 04/23): normalize the loss/other metrics by adding n=xxx if applicable.
            update_meters(meters, monitors, prefix='validation/')
            meters.update({'time/data': data_time, 'time/step': step_time})

            if run_visualizer and visualizer_index < visualizer_total:
                for i in range(len(feed_dict['question_index'])):
                    if args.validation_data_dir is None:
                        image_filename = osp.join(args.data_image_root, feed_dict['image_filename'][i])
                    else:
                        image_filename = osp.join(args.validation_data_image_root, feed_dict['image_filename'][i])
                    image = Image.open(image_filename)

                    with visualizer.table(f'Question #{feed_dict["question_index"][i]}', [
                        HTMLTableColumnDesc('image', 'Image', 'figure', {'width': '600px'}),
                        HTMLTableColumnDesc('question', 'Question', 'text', {'width': '200px'}),
                        HTMLTableColumnDesc('answer', 'Answer', 'text'),
                        HTMLTableColumnDesc('prediction', 'Prediction', 'text'),
                        HTMLTableColumnDesc('program', 'Program', 'code', {'width': '600px'}),
                    ]):
                        fig, ax = vis_bboxes(image, feed_dict['objects_raw'][i], 'Object', add_text=False)
                        visualizer.row(**{
                            'image': fig,
                            'question': feed_dict['question_raw'][i],
                            'answer': feed_dict['answer'][i],
                            'prediction': output_dict['pred_answers'][i],
                            'program': str(output_dict['parsings'][i]),
                        })
                        plt.close()

                    with visualizer.table(f'Question #{feed_dict["question_index"][i]} (Program)', [
                        HTMLTableColumnDesc('id', 'ID', 'text', {'width': '50px'}),
                        HTMLTableColumnDesc('visualization', 'Visualization', 'figure', {'width': '600px'}),
                        HTMLTableColumnDesc('program_and_output', 'Program and Output', 'code', {'width': '600px'}),
                    ]):
                        for j, (program, output) in enumerate(output_dict['execution_traces'][i]):
                            if isinstance(output, TensorValue) and output.dtype.typename in ('bool', 'Object') and len(output.batch_variables) == 1 and output.tensor.shape[0] == len(feed_dict['objects_raw'][i]):
                                fig, ax = vis_bboxes(image, feed_dict['objects_raw'][i], 's:', add_text=True, legends=[str(round(x, 3)) for x in output.tensor.detach().cpu().tolist()])
                                visualizer.row(**{
                                    'id': j,
                                    'visualization': fig,
                                    'program_and_output': str(program) + '\n\n' + str(output),
                                })
                                plt.close()

                    print('Visualized', visualizer_index)
                    visualizer_index += 1
                    if visualizer_index >= visualizer_total:
                        break
            else:
                if args.evaluate_visualization_only:
                    break

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {} (validation)'.format(epoch),
                {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                compressed=True
            ), refresh=False)
            pbar.update()

            end = time.time()

    if run_visualizer:
        visualizer.end_html()
        link = '<a href="viewer://{}", target="_blank">{}</a>'.format(visualizer.visdir, visualizer.visdir)
        mldash.update(run_description=link)


if __name__ == '__main__':
    main()

