# HOMIL

Code for "Hierarchical Optimized Multiple Instance Learning for Cerebral Tumor Diagnosis on Multiple Magnification Histological
Images"

## Installation

To clone all files:

```
git clone https://github.com/RenaoYan/HOMIL.git
```

To install Python dependencies:

```
pip install -r requirements.txt
```

## Preparation

A dataset with all slides in one folder, assuming that digitized whole slide image data in well known standard formats:

```
DATA_DIR/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

A csv file with all labeled slide names (with or without file extension):

```
CSV_DIR/CSV_FILE.csv
		├── slide_id,label
		├── slide_1.svs,tumor
		├── slide_2.svs,non-tumor
		└── ...
```

## Preprocessing

To build 256x256 patches from slides at a specific magnification m1 (e.g. 5x), where the 0th layer magnification is M (
e.g. 40x):

```
python build_patch.py --wsi_dir DATA_DIR --csv_dir CSV_DIR/CSV_FILE.csv --save_dir RESULTS_DATA_DIR --mag_to_cut 5 --magnification 40 --patch_w 256 --patch_h 256
```

The above command will use the OTSU algorithm (as default) to build patches from every slide in DATA_DIR using default
parameters and generate the following folder structure at the specified RESULTS_DATA_DIR:

```
RESULTS_DATA_DIR/
	└── 5x (magnification m1)
		├── slide_1
			├── thumbnail
				└── thumbnail.png
			├── x1_y1.png
			├── x2_y2.png
			└── ...
		├── slide_2
			├── thumbnail
				└── thumbnail.png
			├── x1_y1.png
			├── x2_y2.png
			└── ...
		└── ...
```

To build K-fold dataset (e.g. K = 5), splitting into train and validation datasets:

```
python build_kfold_dataset.py --csv_dir CSV_DIR/CSV_FILE.csv --save_dir RESULTS_CSV_DIR --n_splits 5
```

If a test dataset is needed:

```
python build_kfold_dataset.py --csv_dir CSV_DIR/CSV_FILE.csv --save_dir RESULTS_CSV_DIR --n_splits 5 --test_csv_dir CSV_DIRECTORY/TEST_CSV_FILE.csv
```

The above command will generate K fold csv files with the following folder structure at the specified RESULTS_CSV_DIR:

```
RESULTS_CSV_DIR/
	├── Fold_0.csv
	├── Fold_1.csv
	├── ...
	└── Fold_K-1.csv
```

## Use HOMIL to Learn Data with One Magnification

To train HOMIL with a specific magnification m1 (e.g. 5x) for a C-classification task (e.g. 7):

```
python main.py --experiment_name HOMIL_5x --num_classes 7 --save_topK --patch_dir RESULTS_DATA_DIR/5x --csv_dir RESULTS_CSV_DIR --ckpt_dir ./ckpt --data_dir ./data/5x --ts_dir ./logger
```

The list of training parameters is as follows:

- `experiment_name`: name of this training.
- `num_classes`: classification number of the task.
- `MIL_model`: which attention based MIL model to use, such as ABMIL, CLAM_SB, CLAM_MB.
- `device_ids`: use gpu to speed up training.
- `seed`: random seed to reproduce the results.
- `fold`: which fold csv to train/valid.
- `rounds`: training round, 5 is enough in general.
- `continue_round`: continue round to resume training.
- `feat_aug`: use feature augmentation or not (default no).
- `feat_aug_warmup_round`: use feature augmentation after the warmup round.
- `label_correction`: use label correction or not (default no).
- `save_topK`: save positive and negative top-K patch coords (pROI and nROI) or not (default no).
- `ckpt_dir`: save checkpoint of models.
- `data_dir`: save features and coords.
- `patch_dir`: directory of train/validation patches.
- `test`: use test dataset or not (default no).
- `test_patch_dir`: directory of test patches.
- `ts_dir`: save tensorboard.

The above command will generate the following folder structure:

```
MAIN_DIR/
	├── ckpt
		└── HOMIL_5x (experiment name)
			├── M1
				├── encoder_1.pth
				├── encoder_2.pth
				└── ...
			└── M2
				├── ABMIL_model_0.pth (MIL model)
				├── ABMIL_model_1.pth
				└── ...
	├── data
		└── 5x (magnification m1)
			├── coord (coords of all patches)
				├── slide_1.h5
				├── slide_2.h5
				└── ...
			├── topk_coord (coords of pROI and nROI patches)
				└── HOMIL_5x (experiment name)
					├── round_0
						├── slide_1.h5
						├── slide_2.h5
						└── ...
					├── round_1
						├── slide_1.h5
						├── slide_2.h5
						└── ...
					└── ...
			├── feat0 (weight of encoder inherited from pretrained ImageNet)
				├── slide_1.pt
				├── slide_2.pt
				└── ...
			└── feat (trainable weight of encoder)
				└── HOMIL_5x (experiment name)
					├── round_0
						├── slide_1.pt
						├── slide_2.pt
						└── ...
					├── round_1
						├── slide_1.pt
						├── slide_2.pt
						└── ...
					└── ...
	└── logger
		└── HOMIL_5x (experiment name)
			└── 0 (fold k)
```

## Use HOMIL to Learn Data with Two Magnification

To train HOMIL with another specific magnification m2 (e.g. 20x) for a C-classification task (e.g. 7), the coords of
pROI and nROI should be generated by the best model (e.g. best round = 3), then the corresponding patches at m2 can be
cut from pROIs and nROIs:

```
python build_patch.py --use_coord --csv_dir ./data/topk_coord/HOMIL_5x/round_3 --coord_mag 5 --mag_to_cut 20 --magnification 40 --screen_out --wsi_dir DATA_DIR --csv_dir CSV_DIR/CSV_FILE.csv --save_dir RESULTS_DATA_DIR --patch_w 256 --patch_h 256
```

The above command will build patches from every slide in DATA_DIR using the saved coords and generate the following
folder structure at the specified RESULTS_DATA_DIR:

```
RESULTS_DATA_DIR/
	├── 5x (magnification m1)
		├── slide_1
		├── slide_2
		└── ...
	└── 20x (magnification m2)
		├── slide_1
		├── slide_2
		└── ...
```

To jointly train HOMIL with two magnifications m2 and m1 (well-trained) for a C-classification task:

```
python main.py --experiment_name HOMIL_20x --num_classes 7 --joint --fixed_feat_dir ./data/feat/HOMIL_5x/round_3 --patch_dir RESULTS_DATA_DIR/20x --csv_dir RESULTS_CSV_DIR --ckpt_dir ./ckpt --data_dir ./data --ts_dir ./logger
```

The above command will generate the following folder structure:

```
MAIN_DIR/
	├── ckpt
		├── HOMIL_5x
			├── M1
			└── M2
		└── HOMIL_20x
			├── M1
			└── M2
	├── data
		├── 5x (magnification m1)
			├── coord
			├── topk_coord
			├── feat0
			└── feat
		└── 20x (magnification m2)
			├── coord
			├── topk_coord
			├── feat0
			└── feat
	└── logger
		├── HOMIL_5x
		└── HOMIL_20x
```
## Citation
If you find our work useful in your research or if you use parts of this code please consider citing our paper:
```
@ARTICLE{10899821,
  author={Zhu, Lianghui and Yan, Renao and Guan, Tian and Zhang, Fenfen and Guo, Linlang and He, Qiming and Shi, Shanshan and Shi, Huijuan and He, Yonghong and Han, Anjia},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Hierarchically Optimized Multiple Instance Learning With Multi-Magnification Pathological Images for Cerebral Tumor Diagnosis}, 
  year={2025},
  pages={1-13},
  doi={10.1109/JBHI.2025.3544612}}
```
