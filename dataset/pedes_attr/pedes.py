import glob
import os
import pickle
import json
import numpy as np
import math
import time
import random
import torch
import torch.utils.data as data
from PIL import Image
import hashlib
import cv2
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
# torch.multiprocessing.set_start_method('spawn', force=True)
import random
# from tools.function import get_pkl_rootpath
cv2.setNumThreads(1)
color_2val = {'red':0, 
             'orange':20, 
             'yellow':40, 
             'green':60, 
             'blue':120, 
             'purple':140, 
             'pink':160}
color_skip = set(['mix', 'white', 'black', 'grey', 'brown'])
vest_color = set(['green', 'yellow', 'orange'])
val_2color = {v:k for k,v in color_2val.items()}
color_set = list(color_2val.keys())
id_2attr = json.load(open('id_2attr.json', 'r'))
attr_2id = {v:int(k) for k,v in id_2attr.items()}

# def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
#     # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
#     # paths, flip the order so normal is run on CPU if this becomes a problem
#     # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
#     if per_pixel:
#         return torch.empty(patch_size, dtype=dtype, device=device).normal_()
#     elif rand_color:
#         return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
#     else:
#         return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

def _get_pixels_numpy(per_pixel, rand_color, patch_size, dtype=np.float32):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return np.empty(patch_size, dtype=dtype)
    elif rand_color:
        return np.empty((patch_size[2], 1, 1), dtype=dtype)
    else:
        return torch.zeros((patch_size[2], 1, 1), dtype=dtype)

class PedesAttr(data.Dataset):
    
    def __init__(self, cfg, split, transform=None, target_transform=None, idx=None, testing=None, known_labels=None):

        assert cfg.DATASET.NAME in ['PETA', 'PA100k', 'RAP', 'RAP2', 'PPAR'], \
            f'dataset name {cfg.DATASET.NAME} is not exist'

        # data_path = get_pkl_rootpath(cfg.DATASET.NAME, cfg.DATASET.ZERO_SHOT)
        # data_path = 'data/PPAR/dataset_all_20230913_batch8_full.pkl'
        if cfg.DATASET.PATH:
            data_path = cfg.DATASET.PATH
        print("which pickle", data_path)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name

        attr_label = dataset_info.label
        attr_label[attr_label == 2] = 0
        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)
        self.label_vector = dataset_info.attr_vectors.astype(np.float32)
        self.label_word = dataset_info.attr_words
        # self.words = self.label_word.tolist()
        self.testing = testing
        self.known_labels = known_labels
        
        if 'label_idx' not in dataset_info.keys():
            print(' this is for zero shot split')
            assert cfg.DATASET.ZERO_SHOT
            self.eval_attr_num = self.attr_num
        else:
            self.eval_attr_idx = dataset_info.label_idx.eval
            self.eval_attr_num = len(self.eval_attr_idx)

            assert cfg.DATASET.LABEL in ['all', 'eval', 'color'], f'key word {cfg.DATASET.LABEL} error'
            if cfg.DATASET.LABEL == 'eval':
                attr_label = attr_label[:, self.eval_attr_idx]
                self.attr_id = [self.attr_id[i] for i in self.eval_attr_idx]
                self.attr_num = len(self.attr_id)
            elif cfg.DATASET.LABEL == 'color':
                attr_label = attr_label[:, self.eval_attr_idx + dataset_info.label_idx.color]
                self.attr_id = [self.attr_id[i] for i in self.eval_attr_idx + dataset_info.label_idx.color]
                self.attr_num = len(self.attr_id)

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'
        self.split = split
        self.dataset = cfg.DATASET.NAME
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.root_path = dataset_info.root

        if self.target_transform:
            self.attr_num = len(self.target_transform)
            print(f'{split} target_label: {self.target_transform}')
        else:
            self.attr_num = len(self.attr_id)
            print(f'{split} target_label: all')

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0

        if idx is not None:
            self.img_idx = idx

        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]  # [:, [0, 12]]

        # augmentation configs
        self.split_body = [0.2, 0.55]
        self.height = cfg.DATASET.HEIGHT
        self.width = cfg.DATASET.WIDTH
        self.prob_jitter = 0.2
        self.prob_re = 0.1
        self.random_erasing = RandomErasing(
                probability=self.prob_re,
                mode='pixel',
                max_count=1,
                num_splits=0,
                device='cuda',
            )
        
        self.transform_a_train = A.Compose([
            A.Resize(height=self.height, width=self.width),
            A.ImageCompression(p=0.1),
            A.Spatter(p=0.1),
            A.GaussNoise(p=0.2),
            A.MedianBlur(p=0.1),
            A.PiecewiseAffine(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.PixelDropout(p=0.2),
            A.RandomFog(p=0.1),
            A.RandomRain(p=0.1),
            A.RandomSnow(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.transform_a_val = A.Compose([
            A.Resize(height=self.height, width=self.width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # def auto_padding(self, img):
    #     h,w,c = img.shape
    #     ratio = h / w
    #     target_ratio = self.height / self.width
    #     # 如果人员过方，增加上下的border
    #     if ratio < target_ratio:
    #         vertical_border = int(w * target_ratio - h)
    #         img = cv2.copyMakeBorder(img, vertical_border//2 , vertical_border//2, 0, 0, 
    #                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     # 如果人员过高，增加左右的border
    #     else:
    #         horizontal_border = int(h / target_ratio - w)
    #         img = cv2.copyMakeBorder(img, 0, 0, horizontal_border//2 , horizontal_border//2, 
    #                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     return img
    
    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        gt_label = gt_label.astype(np.float32)
        imgpath = os.path.join(self.root_path, imgname)

        imgpath = imgpath.replace('/aiprime/', '/public/191-aiprime/')
        img = np.asarray(Image.open(imgpath))
        # 启用自动padding，调节到指定高宽比，通常是2
        # img = self.auto_padding(img)
        # img = resizeAndPad(img, 
        #                     (self.height, self.width), 
        #                     [random.randint(0,255) for _ in range(3)])
        # 如果是训练时 才进行colorjitter, 60%的几率发生color jitter
        if self.split == 'train':
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # img, gt_label = self.transform_color_jitter(img, gt_label)
            # img, gt_label = self.random_erasing(img, gt_label)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            
            
            # cv2.imwrite('tmp.jpg', img)
            img = self.transform_a_train(image=img)['image']
        # 如果不是训练，使用默认的transform
        else:
            img = self.transform_a_val(image=img)['image']

        # labels, mask = [], []
        # if self.l2l:
        #     labels = torch.Tensor(gt_label).float()
        #     unk_mask_indices = get_unk_mask_indices(img, self.testing, self.attr_num, self.known_labels)
        #     mask = labels.clone()
        #     mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        return img, gt_label, imgname, self.label_vector, []

    def __len__(self):
        return len(self.img_id)

    def autoAdjustments(self, img, border=0.2):    
        h,w,c = img.shape
        border_x = int(w * border)
        border_y = int(h * border)
        center_img = img[border_y : h-border_y, border_x : w-border_x, :]
        alow = center_img.min()
        ahigh = center_img.max()
        # ahigh = min(128, ahigh)
        amax = 255
        amin = 0
        # calculate alpha, beta
        alpha = ((amax - amin) / (ahigh - alow))
        beta = amin - alow * alpha
        # perform the operation g(x,y)= α * f(x,y)+ β
        new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return [new_img, alpha, beta]

    def shift_2color(self, img, shift, s_jitter=20, v_jitter=20):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        
        s = s.astype(np.uint16)
        shift_s = np.clip((s + random.randint(0-s_jitter, s_jitter)), 0, 255)
        shift_s = shift_s.astype(np.uint8)

        h = h.astype(np.uint16)
        shift_h = (h + shift) % 180
        shift_h = shift_h.astype(np.uint8)

        v = v.astype(np.uint16)
        shift_v = np.clip((v + random.randint(0-v_jitter, v_jitter)), 0, 255)
        shift_v = shift_v.astype(np.uint8)

        shift_hsv = cv2.merge([shift_h, shift_s, shift_v])
        shift_img = cv2.cvtColor(shift_hsv, cv2.COLOR_HSV2BGR)

        return shift_img

    def cal_h_dis(self, h1, h2):
        if h1 < h2:
            return h2- h1
        else:
            return h2 + 180 - h1

    def transform_color_jitter(self, img, label, verbose=False):
        if random.random() > self.prob_jitter:
            return img, label
        t1 = time.time()
        trans_flag = 1
        # 只有站着的人判断
        if label[attr_2id["overall_person_posture|stand"]] == 0:
            trans_flag = 0
        # 只有完整的人才判断
        if label[attr_2id["overall_is_incomplete"]] == 1:
            trans_flag = 0

        if not trans_flag:
            return img, label
        
        if verbose:
            print('entering color finder')
        body_trans = {}
        body_color = {}

        for part in ['head_top', 'body_upper', 'body_lower']:
            body_trans[part] = 1
            for c in ['mix', 'white', 'black', 'grey', 'brown', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']:
                if label[attr_2id[f"{part}_color|{c}"]] == 1:
                    body_color[part] = c
                    if c in color_skip:
                        body_trans[part] = 0
                    break
        if verbose:
            print('entering body cutter')
        img_his, _, _ = self.autoAdjustments(img)
        
        img_h, img_w, _ = img_his.shape
        h_head_upperbody = int(self.split_body[0] * img_h)
        h_upper_lowerbody = int(self.split_body[1] * img_h)

        head =  img_his[:h_head_upperbody, :, :]
        upper = img_his[h_head_upperbody:h_upper_lowerbody, :, :]
        lower = img_his[h_upper_lowerbody:, :, :]

        # 获取三个部位的颜色
        # 确定每个部位的trans flag
        if verbose:
            print('entering color shifting')

        if body_trans['head_top'] and 'head_top' in body_color:
            head_color = body_color['head_top']
            head_h = color_2val[head_color]
            target_head_color = random.choice(color_set)
            target_head_h = color_2val[target_head_color]
            head_diff = self.cal_h_dis(head_h, target_head_h)
            head = self.shift_2color(head, head_diff)

            label[attr_2id[f'head_top_color|{head_color}']] = 0
            label[attr_2id[f'head_top_color|{target_head_color}']] = 1
        
        if body_trans['body_upper'] and 'body_upper' in body_color:
            upper_color = body_color['body_upper']
            upper_h = color_2val[upper_color]
            target_upper_color = random.choice(color_set)
            target_upper_h = color_2val[target_upper_color]
            upper_diff = self.cal_h_dis(upper_h, target_upper_h)
            upper = self.shift_2color(upper, upper_diff)

            label[attr_2id[f'body_upper_color|{upper_color}']] = 0
            label[attr_2id[f'body_upper_color|{target_upper_color}']] = 1

            # 如果上半身穿着反光背心，但是目标颜色不在反光背心的颜色范围内，置为0
            for vc in vest_color:
                label[attr_2id[f'body_upper_reflective_vest|{vc}']] = 0
            if target_upper_color in vest_color:
                label[attr_2id[f'body_upper_reflective_vest|{target_upper_color}']] = 1

            

        if body_trans['body_lower'] and 'body_lower' in body_color:
            lower_color = body_color['body_lower']
            lower_h = color_2val[lower_color]
            target_lower_color = random.choice(color_set)
            target_lower_h = color_2val[target_lower_color]
            lower_diff = self.cal_h_dis(lower_h, target_lower_h)
            lower = self.shift_2color(lower, lower_diff)

            label[attr_2id[f'body_lower_color|{lower_color}']] = 0
            label[attr_2id[f'body_lower_color|{target_lower_color}']] = 1
        
        # 工服不作处理，按样式区分
        # if label[attr_2id[f'body_upper_wearing|uniform']] == 1 and label[attr_2id[f'body_lower_wearing|uniform']] == 1:
        #     if upper_color == lower_color:
        #         if target_lower_color != target_upper_color:

    
        
        shift_person = np.concatenate((head, upper, lower), axis=0)
        t2 = time.time()
        if verbose:
            print('entering concatenate')
            print(img.shape, label.shape)
            print(f'colorjitter_time: {t2-t1}')
        return shift_person, label

class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.4,
            min_area=0.6,
            max_area=0.95,
            min_aspect=0.3,
            max_aspect=None,
            mode='pixel',
            min_count=1,
            max_count=None,
            num_splits=0,
            device='cuda',
            split_body=[0.2, 0.55]
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device
        self.split_body = split_body

    def _erase(self, img, img_h, img_w, chan, dtype):
        area = img_h * img_w
        count = 1
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count

                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[top:top + h, left:left + w, :] = _get_pixels_numpy(
                        self.per_pixel,
                        self.rand_color,
                        (h, w, chan),
                        dtype=dtype
                    )
                    break

    def __call__(self, input, label):
        trans_flag = 1
        if label[attr_2id["overall_person_posture|stand"]] == 0:
            trans_flag = 0
        if label[attr_2id["overall_is_incomplete"]] == 1:
            trans_flag = 0

        if not trans_flag:
            return input, label
        
        img_h, img_w, c = input.shape
        h_head_upperbody = int(self.split_body[0] * img_h)
        h_upper_lowerbody = int(self.split_body[1] * img_h)
        # print(input.shape)
        head =  input[:h_head_upperbody, :, :]
        upper = input[h_head_upperbody:h_upper_lowerbody, :, :]
        lower = input[h_upper_lowerbody:, :, :]
        # print(head.shape, upper.shape, lower.shape)

        head_block = random.random() > self.probability
        if head_block:
            self._erase(head, *head.shape, head.dtype)
            # head_h, head_w, head_c = head.shape
            # self._erase(head, head_c, head_h, head_w, head.dtype)
            label[attr_2id["overall_person_action_phone_calling"]] == 0
            label[attr_2id["overall_person_action_cigarette_smoking"]] == 0

            label[attr_2id["head_top_blocked"]] == 1
            label[attr_2id["head_top_helmet"]] == 0
            label[attr_2id["head_top_hat"]] == 0
            label[attr_2id["head_top_color|mix"]] = 0
            label[attr_2id["head_top_color|red"]] = 0
            label[attr_2id["head_top_color|orange"]] = 0
            label[attr_2id["head_top_color|yellow"]] = 0
            label[attr_2id["head_top_color|green"]] = 0
            label[attr_2id["head_top_color|blue"]] = 0
            label[attr_2id["head_top_color|purple"]] = 0
            label[attr_2id["head_top_color|pink"]] = 0
            label[attr_2id["head_top_color|black"]] = 0
            label[attr_2id["head_top_color|white"]] = 0
            label[attr_2id["head_top_color|grey"]] = 0
            label[attr_2id["head_top_color|brown"]] = 0

            label[attr_2id["head_face_blocked"]] == 1
            label[attr_2id["head_face_glasses"]] = 0
            label[attr_2id["head_face_goggles"]] = 0
            label[attr_2id["head_face_safety_mask|welding_mask"]] = 0
            label[attr_2id["head_face_safety_mask|transparent_mask"]] = 0

        upper_block = random.random() > self.probability
        if upper_block:
            self._erase(upper, *upper.shape, upper.dtype)
            # upper_h, upper_w, upper_c = upper.shape
            # self._erase(upper, upper_c, upper_h, upper_w, upper.dtype)
            label[attr_2id["overall_person_action_cigarette_holding"]] = 0
            label[attr_2id["overall_person_action_phone_playing"]] = 0
            # label[attr_2id["overall_person_posture|stand"]] = 0
            # label[attr_2id["overall_person_posture|sit"]] = 0
            # label[attr_2id["overall_person_posture|ride_bicycle"]] = 0
            # label[attr_2id["overall_person_posture|push_bicycle"]] = 0
            # label[attr_2id["overall_person_posture|bow"]] = 0
            # label[attr_2id["overall_person_posture|crouch"]] = 0
            # label[attr_2id["overall_person_posture|fall"]] = 0
            # label[attr_2id["overall_person_posture|jump"]] = 0
            # label[attr_2id["overall_person_posture|climb"]] = 0
            label[attr_2id["overall_person_orientation|front"]] = 0
            label[attr_2id["overall_person_orientation|side"]] = 0
            label[attr_2id["overall_person_orientation|back"]] = 0

            label[attr_2id["body_upper_blocked"]] = 1
            label[attr_2id["body_upper_safety_belt"]] = 0
            label[attr_2id["body_upper_wearing|cloth"]] = 0
            label[attr_2id["body_upper_wearing|uniform"]] = 0
            label[attr_2id["body_upper_lifejacket|neck"]] = 0
            label[attr_2id["body_upper_lifejacket|belt"]] = 0
            label[attr_2id["body_upper_lifejacket|vest"]] = 0
            label[attr_2id["body_upper_short_sleeve"]] = 0
            label[attr_2id["body_upper_reflective_vest|green"]] = 0
            label[attr_2id["body_upper_reflective_vest|yellow"]] = 0
            label[attr_2id["body_upper_reflective_vest|orange"]] = 0
            label[attr_2id["body_upper_color|mix"]] = 0
            label[attr_2id["body_upper_color|red"]] = 0
            label[attr_2id["body_upper_color|orange"]] = 0
            label[attr_2id["body_upper_color|yellow"]] = 0
            label[attr_2id["body_upper_color|green"]] = 0
            label[attr_2id["body_upper_color|blue"]] = 0
            label[attr_2id["body_upper_color|purple"]] = 0
            label[attr_2id["body_upper_color|pink"]] = 0
            label[attr_2id["body_upper_color|black"]] = 0
            label[attr_2id["body_upper_color|white"]] = 0
            label[attr_2id["body_upper_color|grey"]] = 0
            label[attr_2id["body_upper_color|brown"]] = 0

        lower_block = random.random() > self.probability
        if lower_block:
            self._erase(lower, *lower.shape, lower.dtype)
            # lower_h, lower_w, lower_c = upper.shape
            # self._erase(lower, lower_c, lower_h, lower_w, lower.dtype)
            label[attr_2id["body_lower_blocked"]] = 1
            label[attr_2id["body_lower_wearing|cloth"]] = 0
            label[attr_2id["body_lower_wearing|uniform"]] = 0
            label[attr_2id["body_lower_short_trousers"]] = 0
            label[attr_2id["body_lower_color|mix"]] = 0
            label[attr_2id["body_lower_color|red"]] = 0
            label[attr_2id["body_lower_color|orange"]] = 0
            label[attr_2id["body_lower_color|yellow"]] = 0
            label[attr_2id["body_lower_color|green"]] = 0
            label[attr_2id["body_lower_color|blue"]] = 0
            label[attr_2id["body_lower_color|purple"]] = 0
            label[attr_2id["body_lower_color|pink"]] = 0
            label[attr_2id["body_lower_color|black"]] = 0
            label[attr_2id["body_lower_color|white"]] = 0
            label[attr_2id["body_lower_color|grey"]] = 0
            label[attr_2id["body_lower_color|brown"]] = 0       

        if head_block or upper_block or lower_block:
            label[attr_2id["overall_is_bitrate_error"]] = 1
            label[attr_2id["overall_is_incomplete"]] = 1

        return input, label

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs


def get_unk_mask_indices(image,testing,num_labels,known_labels,epoch=1):
    
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels>0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices



def resizeAndPad(img, size, padColor):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    # if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
    #     padColor = [padColor]*3

    # scale and pad
    # print(padColor)
    # print([new_w, new_h, pad_top, pad_bot, pad_left, pad_right])
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, 
                                    pad_top, 
                                    pad_bot, 
                                    pad_left, 
                                    pad_right, 
                                    borderType=cv2.BORDER_CONSTANT, 
                                    value=padColor)


    
    return scaled_img

def tmp():
    from pathlib import Path
    from tqdm import tqdm
    images = glob.glob('/public/191-aiprime/weimin.xiao/dataset/person_attr/pseudo/zhyz_uniform_20240417/images/*')
    for img in tqdm(images):
        
        name = Path(img).stem.split('.')[0] + '.jpg'
        img = cv2.imread(img)
        scaled_img = resizeAndPad(img,(224,224),114)
        cv2.imwrite(f'/public/191-aiprime/weimin.xiao/dataset/person_attr/pseudo/zhyz_uniform_20240417/scaled_img/{name}',scaled_img)


# tmp()