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

    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Trainer:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)

        # teacher networks
        self.encoder_teacher = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier_teacher = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
        preprocess_teacher(self.encoder, self.encoder_teacher)
        preprocess_teacher(self.classifier, self.classifier_teacher)

        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])

        # dataloaders
        self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
        self.val_loader = get_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()

        # turn on train mode
        self.encoder.train()
        self.classifier.train()
        self.encoder_teacher.train()
        self.classifier_teacher.train()

        for it, (batch, label, domain) in enumerate(self.train_loader):

            # preprocessing
            batch = torch.cat(batch, dim=0).to(self.device)
            label = torch.cat(label, dim=0).to(self.device)
            # domain = torch.cat(domain, dim=0).to(self.device)

            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()

            # forward
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            features = self.encoder(batch)
            scores = self.classifier(features)
            with torch.no_grad():
                features_teacher = self.encoder_teacher(batch)
                scores_teacher = self.classifier_teacher(features_teacher)

            assert batch.size(0) % 2 == 0
            split_idx = int(batch.size(0) / 2)
            scores_ori, scores_aug = torch.split(scores, split_idx)
            scores_ori_tea, scores_aug_tea = torch.split(scores_teacher, split_idx)
            scores_ori_tea, scores_aug_tea = scores_ori_tea.detach(), scores_aug_tea.detach()
            labels_ori, labels_aug = torch.split(label, split_idx)
            assert scores_ori.size(0) == scores_aug.size(0)

            # classification loss for original data
            loss_cls = criterion(scores_ori, labels_ori)
            loss_dict["main"] = loss_cls.item()
            correct_dict["main"] = calculate_correct(scores_ori, labels_ori)
            num_samples_dict["main"] = int(scores.size(0) / 2)

            # classification loss for augmented data
            loss_aug = criterion(scores_aug, labels_aug)
            loss_dict["aug"] = loss_aug.item()
            correct_dict["aug"] = calculate_correct(scores_aug, labels_aug)
            num_samples_dict["aug"] = int(scores.size(0) / 2)

            # calculate probability
            p_ori, p_aug = F.softmax(scores_ori / self.config["T"], dim=1), F.softmax(scores_aug / self.config["T"], dim=1)
            p_ori_tea, p_aug_tea = F.softmax(scores_ori_tea / self.config["T"], dim=1), F.softmax(scores_aug_tea / self.config["T"], dim=1)

            # use KLD for consistency loss
            loss_ori_tea = F.kl_div(p_aug.log(), p_ori_tea, reduction='batchmean')
            loss_aug_tea = F.kl_div(p_ori.log(), p_aug_tea, reduction='batchmean')

            # get consistency weight
            const_weight = get_current_consistency_weight(epoch=self.current_epoch,
                                                          weight=self.config["lam_const"],
                                                          rampup_length=self.config["warmup_epoch"],
                                                          rampup_type=self.config["warmup_type"])

            # calculate total loss
            total_loss = 0.5 * loss_cls + 0.5 * loss_aug + \
                         const_weight * loss_ori_tea + const_weight * loss_aug_tea

            loss_dict["ori_tea"] = loss_ori_tea.item()
            loss_dict["aug_tea"] = loss_aug_tea.item()
            loss_dict["total"] = total_loss.item()

            # backward
            total_loss.backward()

            # update
            self.encoder_optim.step()
            self.classifier_optim.step()
            self.global_step += 1

            # update teachers
            warm_update_teacher(self.encoder, self.encoder_teacher, self.config["teacher_momentum"], self.global_step)
            warm_update_teacher(self.classifier, self.classifier_teacher, self.config["teacher_momentum"], self.global_step)

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()
        self.encoder_teacher.eval()
        self.classifier_teacher.eval()

        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})
                self.results[phase][self.current_epoch] = class_acc

            # save from best val
            if self.results['val'][self.current_epoch] >= self.best_val_acc:
                self.best_val_acc = self.results['val'][self.current_epoch]
                self.best_val_epoch = self.current_epoch + 1
                self.logger.save_best_model(self.encoder, self.classifier, self.best_val_acc)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            features = self.encoder(data)
            scores = self.classifier(features)
            correct += calculate_correct(scores, labels)
        return correct


    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

        self.best_val_acc = 0
        self.best_val_epoch = 0

        for self.current_epoch in range(self.epochs):

            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()

            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
            self._do_epoch()
            self.logger.finish_epoch()

        # save from best val
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_val_acc, self.best_val_epoch - 1)

        return self.logger


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, config, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()