"""Run a simple exp."""

import logging
import torch
from torch.utils.data import DataLoader

from learner.data import SplittedDataset, SlideWindowDataset, GasWellDataset
from learner.handlers import EpochReporter, MetadataLogger, MLFlowLogger, CheckpointModel, LoadModel, PredictionOutputSaving, TestAfterTrain
from learner.learner import Learner
from learner.manager import ModelManager, GPUManager, OutputManager

from utils.flags import global_parse, with_flags
from argparse import Namespace

@with_flags
class Experiment(object):
    args = None

    @classmethod
    def register_flags(cls, parser):
        parser.add_argument('--exp_name', default='',
                            help='A name for this experiment. (Not used)')
        parser.add_argument('--model_id', default='0',
                            help='A model_id for this experiment. (Not used)')
        parser.add_argument('--exp_train', default=False, action='store_true',
                            help='Whether to run the train process.')
        parser.add_argument('--exp_batch_size', default=64, type=int,
                            help='The batch size for training and evaluating.')
        parser.add_argument('--exp_prediction_split', default='test',
                            help='The data split for evaluating.')
        parser.add_argument('--exp_test_after_train', default=True, action='store_false',
                            help='Whether to run a test on the best model (by validation) after training.')

    def __init__(self):
        self.is_train = self.args.exp_train
        self.name = self.args.exp_name
        self.model_mgr = ModelManager()
        self.gpu_mgr = GPUManager()

    def run(self):
        with self.gpu_mgr.device_context():
            if self.is_train:
                self.train()
                return
            self.prediction()

    def train(self):
        train_ds = SlideWindowDataset(
                SplittedDataset(
                    GasWellDataset(), split_name='train'
                    )
                )
        train_ds_loader = DataLoader(train_ds,
                                     batch_size=self.args.exp_batch_size,
                                     shuffle=True,
                                     num_workers=2)

        validation_ds = SlideWindowDataset(
                SplittedDataset(
                    GasWellDataset(), split_name='validation'
                    )
                )
        validation_ds_loader = DataLoader(validation_ds,
                                          batch_size=self.args.exp_batch_size,
                                          shuffle=False,
                                          num_workers=2)
        model = self.model_mgr.build(train_ds.input_shape(), train_ds.output_shape())

        learner = Learner(model)
        learner.register(MetadataLogger())
        learner.register(EpochReporter())
        learner.register(MLFlowLogger())
        learner.register(CheckpointModel(validation_dataloader=validation_ds_loader))

        if self.args.exp_test_after_train:
            learner.register(PredictionOutputSaving())
            test_ds = SlideWindowDataset(
                    SplittedDataset(
                        GasWellDataset(), split_name=self.args.exp_prediction_split
                        )
                    )
            test_ds_loader = DataLoader(test_ds, batch_size=self.args.exp_batch_size, shuffle=False, num_workers=2)
            learner.register(TestAfterTrain(test_ds_loader))

        learner.train(train_ds_loader)

    def prediction(self):
        test_ds = SlideWindowDataset(
                SplittedDataset(
                    GasWellDataset(), split_name=self.args.exp_prediction_split
                    )
                )
        test_ds_loader = DataLoader(test_ds, batch_size=self.args.exp_batch_size, shuffle=False, num_workers=2)

        model = self.model_mgr.build(test_ds.input_shape(), test_ds.output_shape())

        learner = Learner(model)
        learner.register(LoadModel())
        learner.register(MLFlowLogger())
        learner.register(PredictionOutputSaving())

        learner.evaluate(test_ds_loader)


if __name__ == "__main__":
    global_parse()
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.handlers.WatchedFileHandler(OutputManager().get_output_path('log.INFO')),
                                  logging.StreamHandler()],
                        format='%(asctime)s %(levelname)s %(pathname)s:%(lineno)s [%(threadName)s] | %(message)s',
                        datefmt='%m-%d %H:%M')

    exp = Experiment()
    exp.run()
