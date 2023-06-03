import torch
from tqdm import tqdm
from torch.nn import functional as F
from util.metrics import calculate_metrics


def m2_train_epoch(round_id, m2_epoch, model, optimizer, loader, criterion, device, num_classes, model_suffix='ABMIL',
                   dropout_rate=0, joint=False):
    model.train()
    attns = {}
    loss_all = 0.
    logits = torch.Tensor()
    targets = torch.Tensor()
    drop_layer = torch.nn.Dropout(dropout_rate)
    with tqdm(total=len(loader)) as pbar:
        for _, sample in enumerate(loader):
            optimizer.zero_grad()
            slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
            feat = feat.to(device)
            target = target.to(device)
            if joint:
                fixed_feat = sample['fixed_feat']
                fixed_feat = fixed_feat.to(device)
                fixed_feat = drop_layer(fixed_feat) if dropout_rate > 0 else fixed_feat
                logit, attn, _ = model(feat, fixed_feat)
                loss = criterion(logit, target.long())
            else:
                if 'ABMIL' in model_suffix:
                    logit, attn = model(feat)
                    loss = criterion(logit, target.long())
                elif 'CLAM' in model_suffix:
                    bag_weight = 0.7
                    logit, attn, instance_dict = model(feat, target, instance_eval=True)
                    instance_loss = instance_dict['instance_loss']
                    loss = bag_weight * criterion(logit, target.long()) + (1 - bag_weight) * instance_loss
                else:
                    raise NotImplementedError

            # calculate metrics
            attns[slide_id[0]] = attn
            logits = torch.cat((logits, logit.detach().cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)
            acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)
            loss_all += loss.detach().item() * len(target)

            # loss backward
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            pbar.set_description('[Round:{}, M2 Epoch:{}] lr:{:.4f}, loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                 .format(round_id, m2_epoch, lr, loss_all / len(targets), acc, roc_auc, f1))
            pbar.update(1)
    return loss_all / len(targets), acc, roc_auc, f1, attns


def m2_pred(round_id, model, loader, criterion, device, num_classes, model_suffix='ABMIL', joint=False, status='Val'):
    model.eval()
    attns = {}
    loss_all = 0.
    logits = torch.Tensor()
    targets = torch.Tensor()
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, sample in enumerate(loader):
                slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
                feat = feat.to(device)
                target = target.to(device)
                if joint:
                    fixed_feat = sample['fixed_feat']
                    fixed_feat = fixed_feat.to(device)
                    logit, attn, _ = model(feat, fixed_feat)
                else:
                    if 'ABMIL' in model_suffix:
                        logit, attn = model(feat)
                    elif 'CLAM' in model_suffix:
                        logit, attn, _ = model(feat, target)
                    else:
                        raise NotImplementedError

                # calculate metrics
                attns[slide_id[0]] = attn
                loss = criterion(logit, target.long())
                logits = torch.cat((logits, logit.detach().cpu()), dim=0)
                targets = torch.cat((targets, target.cpu()), dim=0)
                loss_all += loss.item() * len(target)
                acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)

                pbar.set_description('[{} Round:{}] loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                     .format(status, round_id, loss_all / len(targets), acc, roc_auc, f1))
                pbar.update(1)
    acc, f1, roc_auc, mat = calculate_metrics(logits, targets, num_classes, confusion_mat=True)
    print(mat)
    return loss_all / len(targets), acc, roc_auc, mat, attns, f1


def m2_patch_pred(model, loader, device, joint=False):
    model.eval()
    instance_probs = {}
    with torch.no_grad():
        for _, sample in enumerate(loader):
            slide_id, feat = sample['slide_id'], sample['feat']
            feat = feat.to(device)
            if joint:
                fixed_feat = sample['fixed_feat']
                fixed_feat = fixed_feat.to(device)
                logit = model(feat, fixed_feat)
            else:
                logit = model(feat)
            probs = F.softmax(logit, dim=1)
            instance_probs[slide_id[0]] = probs.detach().cpu().numpy()
    torch.cuda.empty_cache()
    return instance_probs


def m1_train_epoch(round_id, m1_epoch, model, optimizer, loader, criterion, device, num_classes):
    model.train()
    logits = torch.Tensor()
    targets = torch.Tensor()
    loss_all = 0.
    with tqdm(total=len(loader)) as pbar:
        for i, (img, target) in enumerate(loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)
            logit = model(img)

            loss = criterion(logit, target.long())
            logits = torch.cat((logits, logit.detach().cpu()), dim=0)
            targets = torch.cat((targets, target.cpu()), dim=0)
            loss_all += loss.detach().item() * len(target)

            loss.backward()
            optimizer.step()
            acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)
            pbar.set_description('Round: {}, M1 Epoch:{}, loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                 .format(round_id, m1_epoch, loss_all / len(targets), acc, roc_auc, f1))
            pbar.update(1)
    return loss_all / len(targets), acc, roc_auc, f1


def m1_pred(round_id, model, loader, criterion, device, num_classes, status='Val'):
    model.eval()
    logits = torch.Tensor()
    targets = torch.Tensor()
    loss_all = 0.
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, (img, target) in enumerate(loader):
                img = img.to(device)
                target = target.to(device)
                logit = model(img)
                loss = criterion(logit, target.long()).item()

                logits = torch.cat((logits, logit.cpu()), dim=0)
                targets = torch.cat((targets, target.cpu()), dim=0)
                loss_all += loss * len(target)

                acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)
                pbar.set_description('[{}:{}] loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                     .format(status, round_id, loss_all / len(targets), acc, roc_auc, f1))
                pbar.update(1)
    return loss_all / len(targets), acc, roc_auc, f1


# feature extraction
def feat_extraction(model, loader, device):
    model.eval()
    features = torch.Tensor()
    coords = torch.Tensor()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img = data['image']
            img = img.to(device)
            coord = data['coords']
            feature = model(img)
            features = torch.cat((features, feature.cpu()), dim=0)
            coords = torch.cat((coords, coord), dim=0)
    return features, coords
