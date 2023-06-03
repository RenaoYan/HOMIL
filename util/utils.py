import os
import sys
import time
import math
import h5py
import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
from util.metrics import draw_metrics
from torch.utils.data import DataLoader
from util.preprocess import set_transforms
from dataset.dataset import M2Dataset, M1Dataset, Extract_Feat_Dataset
from util.train_utils import m2_train_epoch, m2_pred, m2_patch_pred, m1_train_epoch, m1_pred, feat_extraction
from models.MIL_models import ABMIL, Feat_Classifier, CLAM_SB, CLAM_MB, Joint_ABMIL, Joint_Feat_Classifier, Aux_Model


def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)
    torch.backends.cudnn.deterministic = True
    sys.setrecursionlimit(10000)


class EarlyStopping:
    def __init__(self, model_path, patience=7, warmup_epoch=0, verbose=False):
        self.patience = patience
        self.warmup_epoch = warmup_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.model_path = model_path

    def __call__(self, epoch, val_loss, model, val_acc=None):
        flag = False
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            flag = True
        if val_acc is not None:
            if self.best_acc is None or val_acc >= self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, model, status='acc')
                self.counter = 0
                flag = True
        if flag:
            return self.counter
        self.counter += 1
        print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
        if self.counter >= self.patience and epoch > self.warmup_epoch:
            self.early_stop = True
        return self.counter

    def save_checkpoint(self, score, model, status='loss'):
        """Saves model when validation loss or validation acc decrease."""
        if status == 'loss':
            pre_score = self.val_loss_min
            self.val_loss_min = score
        else:
            pre_score = self.val_acc_max
            self.val_acc_max = score
        torch.save(model.state_dict(), self.model_path)
        if self.verbose:
            print('Valid {} ({} --> {}).  Saving model ...{}'.format(status, pre_score, score, self.model_path))


def adjust_feat_aug(args, mixup_rate=2, discard_factor=0.1):
    mixup_num, discard_rate = args.mixup_num, args.discard_rate
    if args.feat_aug_method == 'dynamic':
        if args.round_id >= args.feat_aug_warmup_round:
            mixup_num = min(args.mixup_num_th,
                            pow(mixup_rate, args.mixup_num + args.round_id - args.feat_aug_warmup_round))
            discard_rate = min(args.discard_rate_th,
                               discard_rate + discard_factor * (args.round_id - args.feat_aug_warmup_round))
    return mixup_num, discard_rate


def M2_updating(args):
    print('----------------M2_updating starts---------------')
    start = time.time()
    device = args.device
    ts_writer = args.writer
    round_id = args.round_id
    MIL_model = args.MIL_model
    num_classes = args.num_classes
    lr = args.M2_max_lr
    min_lr = args.M2_min_lr
    dataset = args.dataset
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['val'], dataset['test']
    feat_dir = args.pretrained_feat_dir if round_id == 0 else args.feat_dir
    joint = args.joint
    fixed_feat_dir = args.fixed_feat_dir if joint else None
    joint_warmup_epochs = args.joint_warmup_epochs if joint else 0
    dropout_rate = args.dropout_rate if joint else 0
    mixup_num, discard_rate = adjust_feat_aug(args, mixup_rate=2, discard_factor=0.1)

    train_dset = M2Dataset(train_dataset, feat_dir, discard_rate, mixup_num, fixed_feat_dir=fixed_feat_dir)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
    train_dset1 = M2Dataset(train_dataset, feat_dir, fixed_feat_dir=fixed_feat_dir)
    train_loader1 = DataLoader(train_dset1, batch_size=1, shuffle=False, num_workers=0)
    val_dset = M2Dataset(val_dataset, feat_dir, fixed_feat_dir=fixed_feat_dir)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)
    test_dset = M2Dataset(test_dataset, feat_dir, fixed_feat_dir=fixed_feat_dir)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.M2_model_dir, exist_ok=True)
    if joint:
        M2_model_dir = os.path.join(args.M2_model_dir, 'joint_{}_model_{}.pth'.format(MIL_model, round_id))
        if 'ABMIL' in MIL_model:
            model = Joint_ABMIL(n_classes=num_classes)
        else:
            raise NotImplementedError
    else:
        M2_model_dir = os.path.join(args.M2_model_dir, '{}_model_{}.pth'.format(MIL_model, round_id))
        if 'ABMIL' in MIL_model:
            model = ABMIL(n_classes=num_classes)
        elif 'CLAM_SB' in MIL_model:
            model = CLAM_SB(size_arg="small", k_sample=8, n_classes=num_classes, instance_loss_fn=criterion)
        elif 'CLAM_MB' in MIL_model:
            model = CLAM_MB(size_arg="small", k_sample=8, n_classes=num_classes, instance_loss_fn=criterion)
        else:
            raise NotImplementedError
    model = model.to(device)
    warmup_epoch = 0
    if not os.path.exists(M2_model_dir):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        if joint:
            warmup_early_stopping = EarlyStopping(model_path=M2_model_dir, patience=args.joint_patience, verbose=True)
            for warmup_epoch in range(1, joint_warmup_epochs + 1):
                m2_train_epoch(round_id, warmup_epoch, model, optimizer, train_loader, criterion, device, num_classes,
                               MIL_model, dropout_rate=1, joint=True)
                loss, acc, _, _, _, _ = m2_pred(round_id, model, val_loader, criterion, device, num_classes, MIL_model,
                                                True)
                counter = warmup_early_stopping(warmup_epoch, loss, model, acc)
                if warmup_early_stopping.early_stop:
                    print('Early Stop!')
                    break
                # adjust learning rate
                if counter > 0 and counter % 7 == 0 and lr > min_lr:
                    lr = lr / 3 if lr / 3 >= min_lr else min_lr
                    for params in optimizer.param_groups:
                        params['lr'] = lr
            model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'))
            loss, acc, auc, mat, _, f1 = m2_pred(round_id, model, train_loader1, criterion, device, num_classes,
                                                 MIL_model, True)
            draw_metrics(ts_writer, 'Train_WarmUp', num_classes, loss, acc, auc, mat, f1, round_id)
            loss, acc, auc, mat, _, f1 = m2_pred(round_id, model, val_loader, criterion, device, num_classes,
                                                 MIL_model, True)
            draw_metrics(ts_writer, 'Val_WarmUp', num_classes, loss, acc, auc, mat, f1, round_id)
            loss, acc, auc, mat, _, f1 = m2_pred(round_id, model, test_loader, criterion, device, num_classes,
                                                 MIL_model, True)
            draw_metrics(ts_writer, 'Test_WarmUp', num_classes, loss, acc, auc, mat, f1, round_id)

            model = Joint_ABMIL(n_classes=num_classes, dropout=0.5).to(device)
            model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'))
            lr = args.M2_max_lr
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        early_stopping = EarlyStopping(model_path=M2_model_dir, patience=args.M2_patience, verbose=True)
        for m2_epoch in range(warmup_epoch + 1, warmup_epoch + args.M2_epochs + 1):
            m2_train_epoch(round_id, warmup_epoch, model, optimizer, train_loader, criterion, device, num_classes,
                           MIL_model, dropout_rate=dropout_rate, joint=joint)
            loss, acc, _, _, _, _ = m2_pred(round_id, model, val_loader, criterion, device, num_classes, MIL_model,
                                            joint)
            counter = early_stopping(m2_epoch, loss, model, acc)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
            # adjust learning rate
            if counter > 0 and counter % 7 == 0 and lr > min_lr:
                lr = lr / 3 if lr / 3 >= min_lr else min_lr
                for params in optimizer.param_groups:
                    params['lr'] = lr
    model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'))
    loss, acc, auc, mat, train_attns, f1 = m2_pred(round_id, model, train_loader1, criterion, device, num_classes,
                                                   MIL_model, joint, 'Train')
    draw_metrics(ts_writer, 'Train', num_classes, loss, acc, auc, mat, f1, round_id)
    loss, acc, auc, mat, val_attns, f1 = m2_pred(round_id, model, val_loader, criterion, device, num_classes,
                                                 MIL_model, joint, 'Val')
    draw_metrics(ts_writer, 'Val', num_classes, loss, acc, auc, mat, f1, round_id)
    loss, acc, auc, mat, test_attns, f1 = m2_pred(round_id, model, test_loader, criterion, device, num_classes,
                                                  MIL_model, joint, 'Test')
    draw_metrics(ts_writer, 'Test', num_classes, loss, acc, auc, mat, f1, round_id)

    if joint:
        patch_model = Joint_Feat_Classifier(n_classes=num_classes).to(device)
    else:
        patch_model = Feat_Classifier(n_classes=num_classes).to(device)
    patch_model.load_state_dict(torch.load(M2_model_dir, map_location='cpu'), strict=False)
    train_probs = m2_patch_pred(patch_model, train_loader1, device, joint)
    val_probs = m2_patch_pred(patch_model, val_loader, device, joint)
    test_probs = m2_patch_pred(patch_model, test_loader, device, joint)

    obj = {'train_attns': train_attns, 'train_probs': train_probs,
           'val_attns': val_attns, 'val_probs': val_probs,
           'test_attns': test_attns, 'test_probs': test_probs}

    if args.label_correction and round_id > 0:
        patch_model = Feat_Classifier(args.num_classes)
        if args.joint:
            M1_model_dir = os.path.join(args.M1_model_dir, 'joint_encoder_{}.pth'.format(round_id))
        else:
            M1_model_dir = os.path.join(args.M1_model_dir, 'encoder_{}.pth'.format(round_id))
        patch_model.load_state_dict(torch.load(M1_model_dir, map_location='cpu'), strict=False)
        patch_model.to(device)
        train_aux_probs = m2_patch_pred(patch_model, train_loader1, device, False)
        val_aux_probs = m2_patch_pred(patch_model, val_loader, device, False)
        test_aux_probs = m2_patch_pred(patch_model, test_loader, device, False)
        obj.update({'train_aux_probs': train_aux_probs,
                    'val_aux_probs': val_aux_probs,
                    'test_aux_probs': test_aux_probs})
    end = time.time()
    print('M2 use time: ', end - start)

    return obj


def E_step(args, obj):
    dataset = args.dataset
    round_id = args.round_id
    coord_dir = args.coord_dir
    K0 = args.K0

    new_obj = {}
    print('------------------E stage starts-----------------')
    start = time.time()

    for item in dataset:
        dset_patch = {}
        attns_name = item + '_attns'
        probs_name = item + '_probs'
        attns = obj[attns_name]
        probs = obj[probs_name]

        aux_probs = None
        if args.label_correction and round_id > 0:
            aux_probs_name = item + '_aux_probs'
            aux_probs = obj[aux_probs_name]

        slide_to_label = dataset[item]
        with tqdm(total=len(attns)) as pbar:
            for slide_id, attn in attns.items():
                slide_label = slide_to_label[slide_id]

                attn = torch.from_numpy(attn).squeeze(0)
                prob = torch.from_numpy(probs[slide_id])
                prob = torch.transpose(prob, 1, 0)
                prob = prob[slide_label]
                attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
                score = prob * attn

                slide_patch_num = len(score)
                if round_id == 0:
                    K = int(min(K0, slide_patch_num / 3))
                else:
                    K = int(min(math.ceil((round_id + 1) * K0 * math.log10(slide_patch_num)),
                                slide_patch_num / 3))

                h5py_path = os.path.join(coord_dir, slide_id + '.h5')
                file = h5py.File(h5py_path, 'r')
                coord_dset = file['coords']
                coords = np.array(coord_dset[:])
                file.close()

                # positive Top-K tumor patches
                _, ptopk_id = torch.topk(score, k=K, dim=0)
                ptopk_coords = coords[ptopk_id.numpy()].tolist()
                select_coords = ptopk_coords
                label = [slide_label] * K
                idx = ptopk_id.tolist()

                # negative Top-K tumor patches
                _, ntopk_id = torch.topk(-score, k=K, dim=0)
                ntopk_coords = coords[ntopk_id.numpy()].tolist()
                select_coords = select_coords + ntopk_coords
                label = label + [0] * K
                idx = idx + ntopk_id.tolist()
                if aux_probs is not None:
                    resumed_select_coords = select_coords.copy()
                    resumed_label = label.copy()
                    resumed_idx = idx.copy()
                    aux_prob = torch.from_numpy(aux_probs[slide_id])
                    aux_prob = torch.transpose(aux_prob[idx, :], 1, 0)
                    _, pred_label = torch.max(aux_prob, dim=1)
                    wrong_idx = np.where(pred_label.numpy() != slide_label)
                    wrong_probs, _ = torch.max(aux_prob[wrong_idx], dim=1)

                    label_correction_th = args.label_correction_th if 'train' in item else 0.9
                    for i in range(len(wrong_probs)):
                        if wrong_probs[i] > label_correction_th:
                            wrong_coord = resumed_select_coords[i]
                            select_coords.remove(wrong_coord)
                            wrong_label = resumed_label[i]
                            label.remove(wrong_label)
                            wrong_idx = resumed_idx[i]
                            idx.remove(wrong_idx)
                dset_patch[slide_id] = {'coords': select_coords, 'labels': label, 'idx': idx}
                pbar.set_description(item)
                pbar.update(1)
            dset_name = item + '_dset_patch'
            new_set = {dset_name: dset_patch}
            new_obj = {**new_obj, **new_set}

    end = time.time()
    print('E use time: ', end - start)
    return new_obj


def M1_updating(args, new_obj):
    print('----------------M1_updating starts---------------')
    start = time.time()
    device = args.device
    ts_writer = args.writer
    round_id = args.round_id
    patch_dir = args.patch_dir
    num_classes = args.num_classes
    batch_size = args.M1_batch_size
    train_dset_patch = new_obj['train_dset_patch']
    val_dset_patch = new_obj['val_dset_patch']
    test_dset_patch = new_obj['test_dset_patch']
    train_dset = M1Dataset(split=train_dset_patch, patch_dir=patch_dir, transform=set_transforms(True))
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    train_dset1 = M1Dataset(split=train_dset_patch, patch_dir=patch_dir, transform=set_transforms(False))
    train_loader1 = DataLoader(train_dset1, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    val_dset = M1Dataset(split=val_dset_patch, patch_dir=patch_dir, transform=set_transforms(False))
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    test_dset = M1Dataset(split=test_dset_patch, patch_dir=patch_dir, transform=set_transforms(False))
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    os.makedirs(args.M1_model_dir, exist_ok=True)
    if args.joint:
        M1_model_dir = os.path.join(args.M1_model_dir, 'joint_encoder_{}.pth'.format(round_id))
        pre_M1_model_dir = os.path.join(args.M1_model_dir, 'joint_encoder_{}.pth'.format(round_id - 1))
    else:
        M1_model_dir = os.path.join(args.M1_model_dir, 'encoder_{}.pth'.format(round_id))
        pre_M1_model_dir = os.path.join(args.M1_model_dir, 'encoder_{}.pth'.format(round_id - 1))
    model = Aux_Model(num_classes)
    model = model.to(device)
    if not os.path.exists(M1_model_dir):
        if os.path.exists(pre_M1_model_dir):
            model.load_state_dict(torch.load(pre_M1_model_dir, map_location='cpu'))
        # optimization
        optimizer = torch.optim.Adam(model.parameters(), lr=args.M1_lr, weight_decay=5e-4)
        early_stopping = EarlyStopping(model_path=M1_model_dir, patience=args.M1_patience, verbose=True)

        for m1_epoch in range(1, args.M1_epochs + 1):
            m1_train_epoch(round_id, m1_epoch, model, optimizer, train_loader, criterion, device, num_classes)
            val_loss, val_acc, _, _ = m1_pred(round_id, model, val_loader, criterion, device, num_classes, status='Val')
            early_stopping(m1_epoch, val_loss, model, val_acc)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
    model.load_state_dict(torch.load(M1_model_dir, map_location='cpu'))
    loss, acc, auc, f1 = m1_pred(round_id, model, train_loader1, criterion, device, num_classes, status='Train')
    draw_metrics(ts_writer, 'Train', num_classes, loss, acc, auc, None, f1, round_id)
    loss, acc, auc, f1 = m1_pred(round_id, model, val_loader, criterion, device, num_classes, status='Val')
    draw_metrics(ts_writer, 'Val', num_classes, loss, acc, auc, None, f1, round_id)
    loss, acc, auc, f1 = m1_pred(round_id, model, test_loader, criterion, device, num_classes, status='Test')
    draw_metrics(ts_writer, 'Test', num_classes, loss, acc, auc, None, f1, round_id)

    end = time.time()
    print('M1 use time: ', end - start)


def extract_feature(args):
    print('-------------feature extracting starts------------')
    start = time.time()
    device = args.device
    data_eset = args.data_eset
    round_id = args.round_id
    M1_model = Aux_Model(args.num_classes)
    if args.joint:
        M1_model_dir = os.path.join(args.M1_model_dir, 'joint_encoder_{}.pth'.format(round_id))
    else:
        M1_model_dir = os.path.join(args.M1_model_dir, 'encoder_{}.pth'.format(round_id))
    if os.path.exists(M1_model_dir):
        M1_model.load_state_dict(torch.load(M1_model_dir, map_location='cpu'))
        print('loading checkpoints from ', M1_model_dir)
    else:
        print('using checkpoints from ImageNet')
    model = M1_model.to(device)

    coord_dir = args.coord_dir
    os.makedirs(coord_dir, exist_ok=True)
    feat_dir = args.pretrained_feat_dir if round_id == 0 else args.feat_dir
    os.makedirs(feat_dir, exist_ok=True)
    patch_dir = args.patch_dir
    test_patch_dir = args.test_patch_dir

    slide_dict = {}
    for data_name in data_eset:
        paths = sorted(data_eset[data_name])
        for path in paths:
            if 'test' in data_name:
                slide_dict.update({os.path.join(test_patch_dir, path): 'test'})
            elif 'val' in data_name:
                slide_dict.update({os.path.join(patch_dir, path): 'val'})
            else:
                slide_dict.update({os.path.join(patch_dir, path): 'train'})
    slide_paths = sorted(slide_dict.keys())
    with tqdm(total=len(slide_paths)) as pbar:
        for i, slide_path in enumerate(slide_paths):
            slide_name = os.path.basename(slide_path).split('.')[0]
            coord_dir = os.path.join(args.coord_dir, slide_name + '.h5')
            feat_path = os.path.join(feat_dir, '{}.pt'.format(slide_name))
            if os.path.exists(feat_path) and round_id <= 1:
                pbar.update(1)
                continue

            if slide_dict.get(slide_path) == 'test':
                transform = set_transforms(is_train=False)
            else:
                transform = set_transforms(is_train=True)
            dset = Extract_Feat_Dataset(slide_path, transform=transform)
            loader = DataLoader(dset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False)
            features, coords = feat_extraction(model, loader, device)
            if not os.path.exists(coord_dir):
                f = h5py.File(coord_dir, 'w')
                f["coords"] = coords
                f.close()
            torch.save(features, feat_path)

            pbar.set_description('Round: {}, WSI: {}, with {} patches'.format(round_id, slide_name, len(dset)))
            pbar.update(1)

    end = time.time()
    print('feature extracting use time: ', end - start)
