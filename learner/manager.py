"""Managing the learner live cycle."""

import os
import time
import subprocess as sp
from tempfile import NamedTemporaryFile
import uuid
import logging

import torch

from utils.flags import with_flags
from models.builder import ModelBuilder


@with_flags
class OutputManager(object):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--output_id_tmpl', type=str, default='run',
                            help='A template string for id the run.')
        parser.add_argument('--output_tag', type=str, default='',
                            help='A string for tagging the run, used as the last part of output root directory.')
        parser.add_argument('--output_prefix', type=str, default='./results',
                            help='The root path for holding the outputs.')

    def get_output_dir(self):
        assert self.args is not None

        if not self.args.output_tag:
            self.args.output_tag = RUN_TAG

        output_dir = os.path.join(self.args.output_prefix, self.args.output_tag)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_output_path(self, filename):
        return os.path.join(self.get_output_dir(),
                            f'{self.args.output_id_tmpl}_{filename}'.format(self.args))

    def open_output(self, filename, mode='r'):
        return open(self.get_output_path(filename), mode=mode)


@with_flags
class GPUManager(object):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--gpu', default=0, type=int,
                            help='The id of the gpu to use.')

    def __init__(self):
        self.gpu = self.args.gpu

    def device_context(self):
        return torch.cuda.device(self.gpu)


@with_flags
class ModelManager(object):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--model_name', default='linear',
                            help='The name of the model to use.')

    def build(self, ishape, oshape):
        return self.build_by_name(self.args.model_name, ishape, oshape)

    def build_by_name(self, name, ishape, oshape):
        return ModelBuilder[name](ishape, oshape)


def cur_ts():
    return time.strftime('%Y%m%d-%H%M%S')


def cur_git_ver():
    return sp.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()


def mk_tag():
    return f'{cur_ts()}_{cur_git_ver()}_{uuid.uuid4()}'


RUN_TAG = mk_tag()
