import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

label_dict = {'non-tumor': 0, 'lymphoma': 1, 'glioma': 2, 'meningioma': 3, 'ade': 4, 'neuro': 5, 'squamous': 6}


def get_args():
    parser = argparse.ArgumentParser(description='Build K-Fold Dataset')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--n_splits', type=int, default=5, help='the split number of KFold')
    parser.add_argument('--csv_dir', type=str, default='',
                        help='csv file of the train/val dataset')
    parser.add_argument('--test_csv_dir', type=str, default='',
                        help='csv file of the test dataset')
    parser.add_argument('--save_dir', type=str, default='./csv', help='dir to save split dataset')
    args = parser.parse_args()
    return args


def count_label_distribution(labels):
    for label in label_dict.values():
        num = np.sum(labels == label)
        print('{}: {}'.format(label, num))


def split_dataset(args, data_col='slide_id', label_col='label', verbose=True):
    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.csv_dir, encoding='gbk')
    slides = np.array(df[data_col])
    targets = np.array(df[label_col])

    labels = np.array([])
    for label in targets:
        labels = np.append(labels, label_dict.get(label))

    if os.path.exists(args.test_csv_dir):
        df = pd.read_csv(args.test_csv_dir)
        test_slides = np.array(df[data_col])
        test_targets = np.array(df[label_col])

        test_labels = np.array([])
        for label in test_targets:
            test_labels = np.append(test_labels, label_dict.get(label))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for k, (train_index, val_index) in enumerate(skf.split(slides, labels)):
        train_slides, val_slides = slides[train_index], slides[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
        if verbose:
            print('Fold-{}, total num={}, Train Data Distribution:'.format(k, len(train_labels)))
            count_label_distribution(train_labels)
            print('Fold-{}, total num={}, Val Data Distribution:'.format(k, len(val_labels)))
            count_label_distribution(val_labels)
        data_dict = {'train': train_slides, 'train_label': train_labels, 'val': val_slides, 'val_label': val_labels}
        if os.path.exists(args.test_csv_dir):
            data_dict.update({'test': test_slides, 'test_label': test_labels})
        df_save = pd.DataFrame(pd.DataFrame.from_dict(data_dict, orient='index').values.T,
                               columns=list(data_dict.keys()))
        df_save.to_csv(os.path.join(args.save_dir, 'Fold_{}.csv'.format(k)))


if __name__ == "__main__":
    args = get_args()
    split_dataset(args)
