#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil

from dataloader import get_dataloaders
from args import arg_parser
import models
from op_counter import measure_model
from deep_learning_power_measure.power_measure import experiment, parsers


driver = parsers.JsonParser("output_folder")
exp = experiment.Experiment(driver)

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

torch.manual_seed(args.seed)

def main():

    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224
    
    flops = torch.load(args.flops)  

    checkpoints = torch.load(args.checkpoints)

    model = getattr(models, args.arch)(args)
    model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(checkpoints['state_dict'])

    model.eval()

    train_loader = []
    val_loader = []
    test_loader = []

    T = torch.tensor([ 0.800e+00,  0.8000e+00,  0.8000e+00,  8.0e-01,  8.0e-01, 0.8e-00, -1.0000e+08])

#    for i in range(512):
#        args.batch_size=i+1
#        train_loader[i], val_loader[i], test_loader[i] = get_dataloaders(args)
    train_loader, val_loader, test_loader = get_dataloaders(args)


    ## START EXPERIMENT
    p, q = exp.measure_yourself(period=2) # measure consumption every 2 seconds

    with torch.no_grad():
        for j in range(50):
            for i, (input,target) in enumerate(test_loader):
                batch_flops=[]
                input = input.cuda()
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                output, batch_flops = model(input_var, T, flops)     

    q.put(experiment.STOP_MESSAGE)
    ## END EXPERIMENT

    ### EXPERIMENT RESULTS
    driver = parsers.JsonParser(args.output_folder)
    exp_result = experiment.ExpResults(driver)
    exp_result.print()

    return 

if __name__ == '__main__':
    main()
