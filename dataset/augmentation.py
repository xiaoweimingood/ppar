import random
import torch
import numpy as np
import math
import json
import torchvision.transforms as T
from PIL import Image

from dataset.autoaug import AutoAugment
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

class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__



def get_transform(cfg):
    height = cfg.DATASET.HEIGHT
    width = cfg.DATASET.WIDTH
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    random_erasing = RandomErasing(
                probability=0.5,
                mode='pixel',
                max_count=1,
                num_splits=0,
                device='cuda',
            )


    if cfg.DATASET.TYPE == 'pedes':

        train_transform = T.Compose([
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

        valid_transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            normalize
        ])



    elif cfg.DATASET.TYPE == 'multi_label':

        valid_transform = T.Compose([
            T.Resize([height, width]),
            T.ToTensor(),
            normalize,
        ])

        if cfg.TRAIN.DATAAUG.TYPE == 'autoaug':
            train_transform = T.Compose([
                T.RandomApply([AutoAugment()], p=cfg.TRAIN.DATAAUG.AUTOAUG_PROB),
                T.Resize((height, width), interpolation=3),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

        else:
            train_transform = T.Compose([
                T.Resize((height + 64, width + 64)),
                MultiScaleCrop(height, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
    else:

        assert False, 'xxxxxx'

    return train_transform, valid_transform

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

    def _erase(self, img, chan, img_h, img_w, dtype):
        print('entering erasing')
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
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
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
        
        c, img_h, img_w = input.shape
        h_head_upperbody = int(self.split_body[0] * img_h)
        h_upper_lowerbody = int(self.split_body[1] * img_h)
        print(input.shape)
        head =  input[:, :h_head_upperbody, :]
        upper = input[:, h_head_upperbody:h_upper_lowerbody, :]
        lower = input[:, h_upper_lowerbody:, :]
        print(head.shape, upper.shape, lower.shape)

        head_block = random.random() > self.probability
        if head_block:
            print(random.random())
            self._erase(head, *head.size(), head.dtype)
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
            print(random.random())
            self._erase(upper, *upper.size(), upper.dtype)
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
            print(random.random())
            self._erase(lower, *lower.size(), lower.dtype)
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

        return input, label

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs
