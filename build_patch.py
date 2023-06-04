import os
import cv2
import time
import h5py
import argparse
import openslide
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Extracting the feature')
    # dir params.
    parser.add_argument('--use_coord', action='store_true', help='use coord or autocut')
    parser.add_argument('--wsi_dir', type=str,
                        default='./',
                        help='dir of wsi')
    parser.add_argument('--csv_dir', type=str,
                        default='./',
                        help='dir for wsi label')
    parser.add_argument('--coord_dir', type=str,
                        default='./data/coord',
                        help='dir to save patches')
    parser.add_argument('--save_dir', type=str,
                        default='./data/patch',
                        help='dir to save patches')

    # build patch params.
    parser.add_argument('--patch_w', type=int, default=256, help='the width of patch')
    parser.add_argument('--patch_h', type=int, default=256, help='the height of patch')
    parser.add_argument('--slide_format', type=str, default='svs', help='the format of the whole slide images')
    parser.add_argument('--magnification', type=int, default=40, help='magnification of the scanner: 40x, 80x, ...')
    parser.add_argument('--mag_to_cut', type=float, default=20, help='magnification to cut patch: 5x, 20x, 40x, ...')
    parser.add_argument('--blank_th', type=int, default=240, help='the threshold of rgb to screen out blank patch')
    parser.add_argument('--blank_var_th', type=int, default=500, help='the threshold of rgb to screen out blank patch')
    parser.add_argument('--black_th', type=int, default=15, help='the threshold of rgb to screen out black patch')

    # autocut params.
    parser.add_argument('--use_otsu', action='store_false', help='use otsu threshold or not')
    parser.add_argument('--overlap_w', type=int, default=0, help='the overlap width of patch')
    parser.add_argument('--overlap_h', type=int, default=0, help='the overlap height of patch')
    parser.add_argument('--thumbnail_level', type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help='the top level to catch thumbnail images from slide')
    parser.add_argument('--kernel_size', type=int, default=5, help='the kernel size of close and open opts for mask')
    parser.add_argument('--blank_rate_th', type=float, default=0.8, help='cut patches with a lower blank rate')
    parser.add_argument('--black_rate_th', type=float, default=0.2, help='cut patches with a lower black rate')

    # cut from coord params.
    parser.add_argument('--coord_mag', type=float, default=5, help='previous cut magnification: 5x, 20x, 40x, ...')
    parser.add_argument('--screen_out', action='store_true', help='use screen_out or not')

    args = parser.parse_args()
    return args


def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    ret1, th1 = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)
    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(th1), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    _image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    image_open = (_image_open / 255.0).astype(np.uint8)
    return image_open


class Slide2Patch():
    def __init__(self, args):
        # general params.
        self.patch_w = args.patch_w
        self.patch_h = args.patch_h
        self.zoomscale = args.mag_to_cut / args.magnification
        self.x_size = int(self.patch_w / self.zoomscale)
        self.y_size = int(self.patch_h / self.zoomscale)
        self.x_overlap = int(args.overlap_w / self.zoomscale)
        self.y_overlap = int(args.overlap_h / self.zoomscale)
        self.save_dir = os.path.join(args.save_dir, '{}x'.format(int(args.mag_to_cut)))
        self.blank_th = args.blank_th
        self.blank_var_th = args.blank_var_th
        self.black_th = args.black_th

        # autocut params.
        self.use_otsu = args.use_otsu
        self.thumbnail_level = args.thumbnail_level
        self.kernel_size = args.kernel_size
        self.blank_rate_th = args.blank_rate_th

        # cut from coord params.
        self.mag_rate = int(args.mag_to_cut / args.coord_mag)
        self.screen_out = args.screen_out

    def auto_cut(self, data_path):
        slide = openslide.open_slide(data_path)
        _thumbnail_level = slide.level_count - self.thumbnail_level
        _thumbnail = np.array(slide.get_thumbnail(slide.level_dimensions[_thumbnail_level]))
        thumbnail = cv2.cvtColor(_thumbnail, cv2.COLOR_BGRA2BGR)
        bg_mask = get_bg_mask(thumbnail, kernel_size=self.kernel_size)
        marked_img = thumbnail.copy()

        x_size, y_size = self.x_size, self.y_size
        x_overlap, y_overlap = self.x_overlap, self.y_overlap
        x_step, y_step = x_size - x_overlap, y_size - y_overlap
        WSI_level = slide.get_best_level_for_downsample(1 / self.zoomscale)
        x_offset = int(x_size / pow(2, WSI_level))
        y_offset = int(y_size / pow(2, WSI_level))
        slide_x, slide_y = slide.level_dimensions[0]
        thumbnail_save_dir = os.path.join(self.save_dir, os.path.basename(data_path).split('.')[0], 'thumbnail')
        os.makedirs(thumbnail_save_dir, exist_ok=True)
        save_dir = os.path.join(self.save_dir, os.path.basename(data_path))

        for i in range(int(np.floor((slide_x - x_size) / x_step + 1))):
            for j in range(int(np.floor((slide_y - y_size) / y_step + 1))):
                mask_start_x = int(np.floor(i * x_step / slide_x * bg_mask.shape[1]))
                mask_start_y = int(np.floor(j * y_step / slide_y * bg_mask.shape[0]))
                mask_end_x = int(np.ceil((i * x_step + x_size) / slide_x * bg_mask.shape[1]))
                mask_end_y = int(np.ceil((j * y_step + y_size) / slide_y * bg_mask.shape[0]))
                mask = bg_mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
                x_start, y_start = i * x_step, j * y_step

                if self.use_otsu:
                    if (np.sum(mask == 0) / mask.size) < self.blank_rate_th:
                        img = slide.read_region((x_start, y_start), WSI_level, (x_offset, y_offset))
                        img = img.convert('RGB')
                        img.thumbnail((self.patch_w, self.patch_h))
                        img_arr = np.array(img.convert('L'))
                        if (np.sum(img < self.black_th) / img_arr.size) < args.black_rate_th:
                            cv2.rectangle(marked_img, (mask_start_x, mask_start_y),
                                          (mask_end_x, mask_end_y), (255, 0, 0), 2)
                            patch_save_dir = os.path.join(save_dir, '{}_{}.png'.format(x_start, y_start))
                            img.save(patch_save_dir)
                else:
                    img = slide.read_region((x_start, y_start), WSI_level, (x_offset, y_offset))
                    img = img.convert('RGB')
                    img.thumbnail((self.patch_w, self.patch_h))
                    img_arr = np.array(img)
                    img_RGB_mean = np.mean(img_arr[:, :, :])
                    img_RGB_var = np.var(img_arr[:, :, :])
                    if img_RGB_mean < self.blank_th and img_RGB_var > self.blank_var_th:
                        img_arr = np.array(img.convert('L'))
                        if (np.sum(img < self.black_th) / img_arr.size) < args.black_rate_th:
                            cv2.rectangle(marked_img, (mask_start_x, mask_start_y),
                                          (mask_end_x, mask_end_y), (255, 0, 0), 2)
                            patch_save_dir = os.path.join(save_dir, '{}_{}.png'.format(x_start, y_start))
                            img.save(patch_save_dir)
        marked_img.save(os.path.join(thumbnail_save_dir, 'thumbnail.png'))

    def cut_from_coords(self, data_path, coords):
        slide = openslide.open_slide(data_path)

        WSI_level = slide.get_best_level_for_downsample(1 / self.zoomscale)
        x_offset = int(self.x_size / pow(2, WSI_level))
        y_offset = int(self.y_size / pow(2, WSI_level))
        mag_rate = self.mag_rate
        save_dir = os.path.join(self.save_dir, os.path.basename(data_path).split('.')[0])
        os.makedirs(save_dir, exist_ok=True)

        for coord in coords:
            x_start, y_start = int(coord[0]), int(coord[1])
            for i in range(mag_rate):
                for j in range(mag_rate):
                    img_x = int(x_start + i * self.x_size)
                    img_y = int(y_start + j * self.y_size)
                    img = slide.read_region((img_x, img_y), WSI_level, (x_offset, y_offset))
                    img = img.convert('RGB')
                    img.thumbnail((self.patch_w, self.patch_h))
                    img_arr = np.array(img)
                    img_RGB_mean = np.mean(img_arr[:, :, :])
                    img_RGB_var = np.var(img_arr[:, :, :])

                    if img_RGB_mean < self.blank_th and img_RGB_var > self.blank_var_th:
                        patch_save_dir = os.path.join(save_dir, '{}_{}.png'.format(img_x, img_y))
                        img.save(patch_save_dir)


if __name__ == "__main__":
    args = get_args()
    print('-------------patch building starts------------')
    start = time.time()
    Auto_Build = Slide2Patch(args)
    csv_file = pd.read_csv(args.csv_dir)
    slide_list = csv_file['slide_id'].dropna().tolist()
    for slide_id in slide_list:
        slide_path = os.path.join(args.wsi_dir, '{}.{}'.format(slide_id.split('.')[0], args.slide_format))
        if os.path.exists(slide_path):
            if args.use_coord:
                h5py_path = os.path.join(args.coord_dir, '{}.h5'.format(slide_id.split('.')[0]))
                file = h5py.File(h5py_path, 'r')
                coord_dset = file['coords']
                coords = np.array(coord_dset[:])
                file.close()
                Auto_Build.cut_from_coords(slide_path, coords)
            else:
                Auto_Build.auto_cut(slide_path)
        else:
            print(slide_path, 'do not exist!')
    end = time.time()
    print('finish building patch, using time: ', end - start)
