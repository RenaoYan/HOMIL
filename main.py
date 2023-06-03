import os
import torch
import argparse
from util.utils import set_seed, M2_updating, E_step, M1_updating, extract_feature
from util.preprocess import build_dataset
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='HDC main parameters')

    # general params.
    parser.add_argument('--experiment_name', type=str, default='7cls_5x_HOMIL', help='experiment name')
    parser.add_argument('--MIL_model', type=str, default='ABMIL', choices=['ABMIL', 'CLAM_SB', 'CLAM_MB'],
                        help='MIL model to use')
    parser.add_argument('--device_ids', type=str, default=1, help='gpu devices for training')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--fold', type=int, default=1, help='fold number')
    parser.add_argument('--num_classes', type=int, default=2, help='classification number')
    parser.add_argument('--rounds', type=int, default=5, help='rounds to train')
    parser.add_argument('--continue_round', type=int, default=0, help='round to continue')

    # M2 params.
    parser.add_argument('--M2_epochs', type=int, default=400, help='M2 epochs to train')
    parser.add_argument('--M2_patience', type=int, default=20, help='M2 epochs to early stop')
    parser.add_argument('--M2_max_lr', type=float, default=1e-3, help='M2 max learning rate')
    parser.add_argument('--M2_min_lr', type=float, default=1e-4, help='M2 min learning rate')
    parser.add_argument('--M2_model_dir', type=str, default='', help='dir to save M2 model')

    # feature augmentation params.
    parser.add_argument('--feat_aug', action='store_true', help='use feat augmentation')
    parser.add_argument('--feat_aug_method', type=str, default='dynamic', choices=['dynamic', 'static'],
                        help='method to adjust feat augmentation')
    parser.add_argument('--feat_aug_warmup_round', type=int, default=1, help='warmup round to use feat augmentation')
    parser.add_argument('--discard_rate', type=float, default=0, help='initial discard rate')
    parser.add_argument('--discard_rate_th', type=float, default=0.7, help='threshold of discard rate')
    parser.add_argument('--mixup_num', type=int, default=0, help='initial random mixup number')
    parser.add_argument('--mixup_num_th', type=int, default=10, help='threshold of random mixup number')

    # joint learning params.
    parser.add_argument('--joint', action='store_true', help='use joint learning')
    parser.add_argument('--fixed_feat_dir', type=str, default='', help='fixed feat to concat')
    parser.add_argument('--joint_warmup_epochs', type=int, default=200, help='warmup epochs to joint learning')
    parser.add_argument('--joint_patience', type=int, default=20, help='warmup early stop epoch')
    parser.add_argument('--dropout_rate', type=float, default=0.9, help='rate for dropout')

    # E params.
    parser.add_argument('--K0', type=int, default=2, help='initial top-K to select auxiliary dataset')
    parser.add_argument('--label_correction', action='store_true', help='use label correction')
    parser.add_argument('--label_correction_th', type=float, default=0.7, help='prob threshold to correct label')

    # M1 params.
    parser.add_argument('--M1_epochs', type=int, default=400, help='M1 epochs to train')
    parser.add_argument('--M1_patience', type=int, default=20, help='M1 early stop epoch')
    parser.add_argument('--M1_lr', type=float, default=5e-5, help='M1 learning rate')
    parser.add_argument('--M1_batch_size', type=int, default=64, help='M1 batch size')
    parser.add_argument('--M1_model_dir', type=str, default='', help='dir to save M1 models')

    # dir params.
    parser.add_argument('--csv_dir', type=str, default='', help='csv dir to load data')
    parser.add_argument('--patch_dir', type=str, default='', help='train/val patch dir')
    parser.add_argument('--test_patch_dir', type=str, default='', help='test patch dir')
    parser.add_argument('--feat_dir', type=str, default='', help='train/val/test feat dir')
    parser.add_argument('--coord_dir', type=str, default='', help='train/val/test coord dir')
    parser.add_argument('--log_dir', type=str, default='', help='tensorboard dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # set device
    args.device = torch.device('cuda:{}'.format(args.device_ids))
    print('Using GPU ID: {}'.format(args.device_ids))
    # set random seed
    set_seed(args.seed)
    print('Using Random Seed: {}'.format(str(args.seed)))

    # set tensorboard
    args.log_dir = os.path.join(args.log_dir, args.experiment_name, str(args.fold))
    os.makedirs(args.log_dir, exist_ok=True)
    args.writer = SummaryWriter(args.log_dir)
    print('Set Tensorboard: {}'.format(args.ts_dir))

    # set checkpoint dirs
    args.M2_model_dir = os.path.join(args.M2_model_dir, args.experiment_name, 'M2')
    os.makedirs(args.M2_model_dir, exist_ok=True)
    args.M1_model_dir = os.path.join(args.M1_model_dir, args.experiment_name, 'M1')
    os.makedirs(args.M1_model_dir, exist_ok=True)

    # set feat dirs
    args.pretrained_feat_dir = os.path.join(args.feat_dir, 'feat0')
    args.feat_dir = os.path.join(args.feat_dir, 'feat_{}'.format(args.experiment_name))
    args = args.build_dataset(args)

    args.round_id = args.continue_round
    extract_feature(vars(args))  # extract feature
    obj = M2_updating(vars(args))  # M2 round 0
    for round_id in range(args.continue_round + 1, args.rounds):
        new_obj = E_step(vars(args), obj)  # E_step
        args.round_id = round_id
        M1_updating(vars(args), new_obj)  # M1 updating
        extract_feature(vars(args))  # extract feature
        obj = M2_updating(vars(args))  # M2 updating
