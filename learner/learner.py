"""The learning framework."""

import mlflow
import torch
from argparse import Namespace

from learner import handlers
from learner.metrics import MSE, MAE
from utils.flags import with_flags


class TrainStepLimit(Exception):
    pass


@with_flags
class Learner(object):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--learner_train_epochs', type=int, default=1,
                            help='The number of epochs to run trainig.')
        parser.add_argument('--learner_train_steps', type=int, default=-1,
                            help='The number of steps to run trainig.')
        parser.add_argument('--learner_train_lr', type=float, default=1e-4,
                            help='The number of steps to run trainig.')

    def __init__(self, model,
                 loss_fn=torch.nn.MSELoss(), metrics=(MSE(), MAE()),
                 optimizer=torch.optim.Adam):
        mlflow.autolog()

        self.model = model.to('cuda')
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer(self.model.parameters(), lr=self.args.learner_train_lr)
        self.handle_reg = handlers.Registry(self.HANDLES)

    def step_train(self, minibatch, step, ctx):
        ctx.is_train = True
        handle = self.handle_reg.invoke
        if ctx.train_total_step > self.args.learner_train_steps > 0:
            raise TrainStepLimit()  # Breaking when the train steps are set and the current the total steps reaches it.

        self.model.train(True)
        inputs, targets = minibatch
        handle(ctx, 'before_train_step', train_step=step, train_inputs=inputs, train_targets=targets)

        outputs = self.model(ctx.train_inputs.to('cuda'))
        handle(ctx, 'before_train_loss', train_outputs=outputs)

        loss = self.loss_fn(ctx.train_outputs, ctx.train_targets.to('cuda'))
        # if self.args.model_name=='lts':
        #     loss = loss + 0.00001 * torch.sum(self.model.mstd_linear.weight**2)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        ctx.train_total_step += 1
        handle(ctx, 'after_train_step', train_loss=loss)

    def train(self, dataloader, ctx=None):
        if ctx is None:
            ctx = Namespace(is_train=True, learner=self, train_total_step=0)
        handle = self.handle_reg.invoke

        try:
            handle(ctx, 'before_train', learner=self)
            self.optimizer.zero_grad()
            for epoch in range(self.args.learner_train_epochs):
                handle(ctx, 'before_train_epoch', train_epoch=epoch)
                for step, minibatch in enumerate(dataloader):
                    self.step_train(minibatch, step, ctx)
                handle(ctx, 'after_train_epoch')
        except TrainStepLimit:
            pass
        handle(ctx, 'after_train')

    def evaluate(self, dataloader, ctx=None):
        if ctx is None:
            ctx = Namespace(learner=self, train_total_step=0)
        ctx.is_train = False
        self.model.train(False)
        handle = self.handle_reg.invoke

        all_outputs = []
        all_targets = []
        handle(ctx, 'before_evaluate', eval_all_outputs=all_outputs, eval_all_targets=all_targets)
        with torch.no_grad():
            for step, minibatch in enumerate(dataloader):
                inputs, targets = minibatch
                handle(ctx, 'before_evaluate_step', eval_step=step, eval_inputs=inputs, eval_targets=targets)

                outputs = self.model(ctx.eval_inputs.to('cuda'))
                all_outputs.append(outputs)
                all_targets.append(targets.to('cuda'))
                handle(ctx, 'before_evaluate_loss', eval_outputs=outputs, eval_all_outputs=all_outputs, eval_all_targets=all_targets)

            all_outputs = torch.cat(ctx.eval_all_outputs)
            all_targets = torch.cat(ctx.eval_all_targets)
            loss = self.loss_fn(all_outputs, all_targets)
            handle(ctx, 'after_evaluate_loss', eval_loss=loss)

            metrics = {m.name: m(all_outputs, all_targets) for m in self.metrics}
            handle(ctx, 'after_evaluate_step', eval_metrics=metrics)

        handle(ctx, 'after_evaluate')

    def register(self, handler):
        self.handle_reg.append(handler)

    HANDLES = [
            'before_train',
            'before_train_epoch',
            'before_train_step',
            'before_train_loss',
            'after_train_step',
            'after_train_epoch',
            'after_train',
            'before_evaluate',
            'before_evaluate_step',
            'before_evaluate_loss',
            'after_evaluate_loss',
            'after_evaluate_step',
            'after_evaluate',
            ]
