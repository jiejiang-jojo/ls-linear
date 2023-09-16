"""The handler customize the learner or other process by inserting logic."""

import logging
import json
from argparse import Namespace
import subprocess as sp
from time import time

import pandas as pd
import numpy as np
import torch
import mlflow as mf

from learner.manager import OutputManager, RUN_TAG
from utils.flags import with_flags


class Registry(object):
    def __init__(self, handles):
        self.handles = handles
        self.handler_map = {pos: [] for pos in self.handles}

    def append(self, handler):
        for pos, action in handler.methods():
            if pos not in self.handler_map:
                raise ValueError(f'Unknown position {pos} in {self.__class__.__name__}')
            self.handler_map[pos].append(action)

    def invoke(self, ctx, position, **new_vars):
        for k, v in new_vars.items():
            setattr(ctx, k, v)
        for action in self.handler_map[position]:
            action(ctx)


class HandlerBase(object):
    """The base class will provide register to all methods starting with handle."""

    def methods(self):
        return [(k[len('handle_'):], getattr(self, k))
                for k in dir(self)
                if k.startswith('handle_')]


@with_flags
class MetadataLogger(HandlerBase):
    args = None

    @classmethod
    def register_flags(cls, parser):
        pass

    def __init__(self, out_mgr=OutputManager()):
        self.out = out_mgr

    def handle_before_train(self, ctx):
        assert self.args is not None
        with self.out.open_output('flags.json', mode='w') as fout:
            json.dump(vars(self.args), fout, indent=2)
        with self.out.open_output('repo.diff', mode='w') as fout:
            diff = sp.check_output(['git', 'diff', 'HEAD']).decode().strip()
            fout.write(diff)
            ctx.metadata_repo_diff_path = fout.name


@with_flags
class MLFlowLogger(HandlerBase):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--mlflow_log_by_steps', type=int, default=-1,
                            help='Whether to record the model at multiples of the given step.')

    def __init__(self):
        pass

    def handle_before_train(self, ctx):
        assert self.args is not None
        mf.start_run(run_name=RUN_TAG)
        mf.log_params(vars(self.args))
        mf.set_tag('is_train', 'True')
        try:
            # broken at https://github.com/mlflow/mlflow/issues/7819
            mf.log_artifact(ctx.metadata_repo_diff_path)
        except Exception as err:
            logging.warning(f'{err}')

    def handle_after_train_step(self, ctx):
        if self.args.mlflow_log_by_steps > 0 and ctx.train_total_step % self.args.mlflow_log_by_steps == 0:
            mf.log_metric(f'loss/train_every_{self.args.mlflow_log_by_steps}_steps', ctx.train_loss, step=ctx.train_total_step)

    def handle_after_train_epoch(self, ctx):
        mf.log_metric('loss/train_epoch', ctx.epoch_train_loss, step=ctx.train_total_step)

    def handle_before_evaluate(self, ctx):
        if getattr(ctx, 'evaluate_only', False):
            mf.set_tag('evaluate_only', 'true')

    def handle_after_evaluate(self, ctx):
        mf.log_metric('loss/validation', ctx.eval_loss, step=ctx.train_total_step)
        for name, value in ctx.eval_metrics.items():
            mf.log_metric(f'{name}/validation', value, step=ctx.train_total_step)
            mf.log_metric(f'{name}/validation_best', ctx.best_metrics[name], step=ctx.train_total_step)
        if hasattr(ctx, 'best_model_path'):
            mf.set_tag('best_model_path', ctx.best_model_path)

    def handle_after_train(self, ctx):
        mf.end_run()


@with_flags
class CheckpointModel(HandlerBase):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--checkpoint_name_tmpl', type=str, default='step_{train_total_step}_loss_{model_loss:.4f}.model',
                            help='The template filename for saved models.')
        parser.add_argument('--checkpoint_by_steps', type=int, default=-1,
                            help='Whether to record the model at multiples of the given step.')
        parser.add_argument('--checkpoint_use_loss', type=bool, default=True,
                            help='If true only keep the model with the best loss.')

    def __init__(self, validation_dataloader=None, out_mgr=OutputManager()):
        self.out = out_mgr
        self.best_loss = None
        self.validation_dataloader = validation_dataloader

    def maybe_save_model(self, model, ctx):
        if not self.args.checkpoint_use_loss:
            torch.save(model.state_dict(),
                       self.out.get_output_path(self.args.checkpoint_name_tmpl.format(**vars(ctx))))
            return

        if self.validation_dataloader is not None:
            ctx.learner.evaluate(self.validation_dataloader, ctx)
            model_loss = ctx.eval_loss
        else:
            model_loss = ctx.train_loss

        if self.best_loss is not None and model_loss >= self.best_loss:
            return
        model_path = self.out.get_output_path(self.args.checkpoint_name_tmpl.format(model_loss=model_loss, **vars(ctx)))
        torch.save(model.state_dict(), model_path)
        self.best_loss = model_loss
        ctx.best_model_path = model_path

    def handle_after_train_step(self, ctx):
        if self.args.checkpoint_by_steps > 0 and ctx.train_total_step % self.args.checkpoint_by_steps == 0:
            self.maybe_save_model(ctx.learner.model, ctx)

    def handle_after_train_epoch(self, ctx):
        self.maybe_save_model(ctx.learner.model, ctx)


@with_flags
class PredictionOutputSaving(HandlerBase):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--prediction_output_saving_name_tmpl', type=str, default='prediction.{format}',
                            help='The template filename for saving predictions.')
        parser.add_argument('--prediction_output_saving_format', type=str, default='csv',
                            help='The format for saving the predictions. (csv, npy)')
        parser.add_argument('--prediction_target_saving_name_tmpl', type=str, default='prediction_target.{format}',
                            help='The template filename for saving prediction targets.')
        parser.add_argument('--prediction_target_saving_format', type=str, default='csv',
                            help='The format for saving the prediction targets. (csv, npy)')

    def __init__(self, evaluation_dataloader=None, out_mgr=OutputManager()):
        self.out = out_mgr

    def handle_after_evaluate(self, ctx):
        if ctx.is_train:
            return
        prediction_output_path = self.out.get_output_path(
                self.args.prediction_output_saving_name_tmpl.format(**dict(vars(ctx), format=self.args.prediction_output_saving_format)))
        outputs = torch.cat(ctx.eval_all_outputs)
        if prediction_output_path.endswith('.npy'):
            np.save(prediction_output_path, outputs.to('cpu'))
        elif prediction_output_path.endswith('.csv'):
            pd.DataFrame(outputs.squeeze().to('cpu')).to_csv(prediction_output_path, index=False, header=False)

        prediction_target_path = self.out.get_output_path(
                self.args.prediction_target_saving_name_tmpl.format(**dict(vars(ctx), format=self.args.prediction_output_saving_format)))
        targets = torch.cat(ctx.eval_all_targets)
        if prediction_target_path.endswith('.npy'):
            np.save(prediction_target_path, targets.to('cpu'))
        elif prediction_target_path.endswith('.csv'):
            pd.DataFrame(targets.squeeze().to('cpu')).to_csv(prediction_target_path, index=False, header=False)


@with_flags
class LoadModel(HandlerBase):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--load_model_path', type=str, default=None,
                            help='The path to the model to load.')

    def __init__(self, model_path=None):
        self.model_path = model_path
        if self.model_path is None:
            self.model_path = self.args.load_model_path

    def handle_before_evaluate(self, ctx):
        ctx.learner.model.load_state_dict(torch.load(self.model_path))


def title(title: str, width: int=50) -> str:
    tl = len(title)
    if tl > width - 4:
        return title
    bars = (width - tl - 2) // 2
    return '-' * bars + f' {title} ' + '-' * bars


def duration(secs, limit=2):
    line = ''
    for s, u in (
        (86400, 'd'),
        (3600, 'h'),
        (60, 'm'),
        (1, 's'),
        (0.001, 'ms')):
        if secs > s:
            line += f'{int(secs/s):d}{u}'
            secs = secs % s
            limit -= 1
            if limit == 0 or abs(secs - 0) < 1e-4:
                return line
    return line


class EpochReporter(HandlerBase):
    def __init__(self):
        self.header_printed = False

    def handle_before_train(self, ctx):
        logging.info(title('Train Started'))
        ctx.best_metrics = {}

    def handle_before_train_epoch(self, ctx):
        ctx.epoch_train_loss_acc = 0
        ctx.epoch_train_steps = 0
        ctx.epoch_start_time = time()

    def handle_after_train_step(self, ctx):
        ctx.epoch_train_loss_acc += ctx.train_loss.item()
        ctx.epoch_train_steps += 1

    def handle_after_train_epoch(self, ctx):
        ctx.epoch_train_loss = ctx.epoch_train_loss_acc / ctx.epoch_train_steps
        ctx.epoch_duration = time() - ctx.epoch_start_time

    def handle_after_evaluate(self, ctx):
        self.log_loss(ctx)
        for name, value in ctx.eval_metrics.items():
            if ctx.best_metrics.get(name) is None or ctx.best_metrics[name] > value:
                ctx.best_metrics[name] = value

    def handle_after_train(self, ctx):
        logging.info(title('Train End'))

    def log_loss(self, ctx):
        if not self.header_printed:
            logging.info('epoch        train_loss        valid_loss        duration')
            self.header_printed = True
        logging.info(f'{ctx.train_epoch:5d}        {ctx.epoch_train_loss:10.6f}        {ctx.eval_loss:10.6f}        {duration(ctx.epoch_duration)}')


class TestAfterTrain(HandlerBase):
    def __init__(self, test_ds_loader):
        self.test_ds_loader = test_ds_loader

    def handle_after_train(self, ctx):
        ctx.is_train = False
        logging.info(title('Evaluate on the Best w/ Test data'))
        ctx.learner.model.load_state_dict(torch.load(ctx.best_model_path))
        ctx.learner.evaluate(self.test_ds_loader, ctx)
