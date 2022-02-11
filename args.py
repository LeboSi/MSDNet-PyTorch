import os
import glob
import time
import argparse

model_names = ['msdnet']

arg_parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')            
exp_group.add_argument('--flops', default='save/flops.pth', type=str)
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu', default=None, type=str, help='GPU available.')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

# model arch related
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: msdnet)')
arch_group.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')
arch_group.add_argument('--checkpoints', default=None, type=str,
                        metavar='C', help='path to model checkpoints')

# msdnet config
arch_group.add_argument('--nBlocks', type=int, default=1)
arch_group.add_argument('--nChannels', type=int, default=32)
arch_group.add_argument('--base', type=int,default=4)
arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
arch_group.add_argument('--step', type=int, default=1)
arch_group.add_argument('--growthRate', type=int, default=6)
arch_group.add_argument('--grFactor', default='1-2-4', type=str)
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
arch_group.add_argument('--bnFactor', default='1-2-4')
arch_group.add_argument('--bottleneck', default=True, type=bool)


# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')

#experiement related
exp_group = arg_parser.add_argument_group('experiments',
                                            'experiments setting')
exp_group.add_argument('--output_folder',
                    help='directory to save the energy consumption records',
                    default='measure_power', type=str)