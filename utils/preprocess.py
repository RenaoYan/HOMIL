import os
import pandas as pd
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def prepare_data(df, case_id, label_dict=None):
    df_case_id = df['case_id'].tolist()
    df_slide_id = df['slide_id'].tolist()
    df_label = df['label'].tolist()

    slide_id = []
    label = []
    for case_id_ in case_id:
        idx = df_case_id.index(case_id_)
        slide_id.append(df_slide_id[idx])
        label_ = df_label[idx]
        if label_dict is None:
            label.append(int(label_))
        else:
            label.append(label_dict[label_])
    return slide_id, label


def return_splits(csv_path, label_dict=None, label_csv=None):
    assert os.path.exists(csv_path)
    split_df = pd.read_csv(csv_path)
    train_id = split_df['train'].dropna().tolist()
    val_id = split_df['val'].dropna().tolist()
    test_id = split_df['test'].dropna().tolist()
    if label_csv is None:
        train_label = split_df['train_label'].dropna().tolist()
        train_label = list(map(int, train_label))
        val_label = split_df['val_label'].dropna().tolist()
        val_label = list(map(int, val_label))
        test_label = split_df['test_label'].dropna().tolist()
        test_label = list(map(int, test_label))
    else:
        df = pd.read_csv(label_csv)
        train_id, train_label = prepare_data(df, train_id, label_dict)
        val_id, val_label = prepare_data(df, val_id, label_dict)
        test_id, test_label = prepare_data(df, test_id, label_dict)

    train_split = dict(zip(train_id, train_label))
    val_split = dict(zip(val_id, val_label))
    test_split = dict(zip(test_id, test_label))

    return train_split, val_split, test_split


def build_dataset(args):
    csv_path = os.path.join(args.csv_dir, 'Fold_{}.csv'.format(args.fold))  # dir to save label
    train_dataset, val_dataset, test_dataset = return_splits(csv_path=csv_path)
    args.dataset = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    train_eset, val_eset, test_eset = train_dataset.keys(), val_dataset.keys(), test_dataset.keys()
    args.data_eset = {'train_eset': train_eset, 'val_eset': val_eset, 'test_eset': test_eset}
    return args


def set_transforms(is_train=True):
    if is_train:
        t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)])
        t.transforms.append(transforms.ToTensor())
        t.transforms.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    return t
