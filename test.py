import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--config", default=None, help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--ckpt", default=None, help="The directory to models")

    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Evaluator:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)

        # dataloaders
        self.test_loader = get_test_loader(args=self.args, config=self.config)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            features = self.encoder(data)
            scores = self.classifier(features)
            correct += calculate_correct(scores, labels)
        return correct

    def do_testing(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)

        self.encoder.eval()
        self.classifier.eval()
        if self.args.ckpt is not None:
            state_dict = torch.load(self.args.ckpt, map_location=lambda storage, loc: storage)
            encoder_state = state_dict["encoder_state_dict"]
            classifier_state = state_dict["classifier_state_dict"]
            self.encoder.load_state_dict(encoder_state)
            self.classifier.load_state_dict(classifier_state)

        with torch.no_grad():
            total = len(self.test_loader.dataset)
            class_correct = self.do_eval(self.test_loader)
            class_acc = float(class_correct) / total
            self.logger.log_test(f'Test accuracy', {'class': class_acc})


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(args, config, device)
    evaluator.do_testing()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()