import argparse
import json
import os
from os import path as osp
from copy import deepcopy
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import glob
import cv2
from datetime import datetime
from dataset.augmentation import get_transform
from dataset.multi_label.coco import COCO14
from models.model_factory import build_backbone, build_classifier
from loguru import logger
from PIL import Image
import numpy as np
import torch

from tqdm import tqdm
from pathlib import Path
from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from models import FeatClassifier
from tools.function import get_reload_weight
from tools.utils import set_seed, str2bool
from ppar_utils import *

img_suf = set(['.jpg', '.JPG', '.png'])

def main(cfg, args, save_fmt='class'): # or class
    set_seed(42)

    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    # model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    _, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
                                target_transform=cfg.DATASET.TARGETTRANSFORM)
    valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                            target_transform=cfg.DATASET.TARGETTRANSFORM)

    valid_idx = train_set.eval_attr_idx
    logger.warning(f'backbone {cfg.BACKBONE.TYPE}, head {cfg.CLASSIFIER.NAME}')
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=train_set.attr_num,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    attr_json = json.load(open('attributes_1_0_6.json'))
    attr_dict = gen_candidate(attr_json)
    attr_2id = gen_template(attr_json)
    id_2attr = {v:k for k,v in attr_2id.items()}
    print(attr_dict)
    print(attr_2id)
    model_path = 'weights'

    pth = 'ppar_batch8_internimage_b_224_best_acc.pth'

    model = FeatClassifier(backbone, classifier)
    model = get_reload_weight(model_path, model, pth=pth)
    model = model.cuda()
    model.eval()

    batch_size = 16

    img_dir_list = {'lifejacket':[
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230202_nc4_shfb/labeled',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20221230_nc4_csyk',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20221121_nc4_spc',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230131_nc4_shfb',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230201_nc4_shfb',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230104_nc4_prev_ppe',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230130_nc4_hardcase',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230202_history_nc4_shfb/',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230228_hardcase_before20230210/label',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230228_hardcase_before20230220',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230310_beihailianhua/label',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230417_bwty/label',
            # '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230504_nc4_bwty_batch4/label',
            '/public/191-aiprime/weimin.xiao/dataset/lifejacket/20230608_nc4_bhlh/label'

            # '/home/weimin.xiao/dataset/public_dataset/pa100k/0'
            ],
            'safetyharness':['/public/191-aiprime/weimin.xiao/dataset/safetybelt/belt_cls/prev_nc2_20230111',
                             '/public/191-aiprime/weimin.xiao/dataset/safetybelt/belt_cls/yzhx_nc2_20230412',
                             '/public/191-aiprime/weimin.xiao/dataset/safetybelt/belt_cls/russia_nc2_20230515',
                             '/public/191-aiprime/weimin.xiao/dataset/safetybelt/belt_cls/russia_nc2_20230629',
                             '/public/191-aiprime/weimin.xiao/dataset/safetybelt/belt_cls/russia_nc2_20230902'
                         ],
    }
    task = 'lifejacket'
    imgdir_list = img_dir_list[task]

    # imgdir_list = open('filelist/safetyharness_40k.txt', 'r').readlines()
    # imgdir_list = open(f'/public/191-aiprime/weimin.xiao/projects/timm/flist/{task}/val.txt', 'r').readlines()
    # imgdir_list = glob.glob('/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_dml/5label_intersection/*')
    folder = 'tjg_uniform_20240712'
    imgdir_list = glob.glob(f'/public/191-aiprime/weimin.xiao/dataset/person_attr/pseudo/{folder}/images/*.jpg')
    output_dir = f'/public/191-aiprime/weimin.xiao/dataset/person_attr/pseudo/{folder}'
    imgdir_list = [imgdir_list]


    for data_idx, img_dir in enumerate(imgdir_list):
        

        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M")
        output_result = {}
        os.makedirs(output_dir, exist_ok=True)
        # os.makedirs(output_dir + '/images', exist_ok=True)
        os.makedirs(output_dir + '/labels', exist_ok=True)

        # img_list = glob.glob(img_dir + '/**', recursive=True)
        img_list = img_dir
        img_list = [x for x in img_list if Path(x).suffix in img_suf]

        # new_img_list = []
        # for img_path in img_list:
        #     # 防止伪标签加入
        #     if 'unlabel' in img_path or 'prelabel' in img_path or 'PPAR' in img_path:
        #         continue
        #     if 'ignore' in img_path or 'unsure' in img_path or 'pseudo' in img_path:
        #         continue
        #     new_img_list.append(img_path)
        # img_list = new_img_list

        for i in tqdm(range(0, len(img_list), batch_size)):
            batch_imgs = []
            batch_imgs_size = []
            for j in range(i, min(i + batch_size, len(img_list))):
                img_path = img_list[j]
                

                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)
                h, w, _ = img.shape
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                if valid_tsfm is not None:
                    img = valid_tsfm(img)
                batch_imgs.append(img)
                batch_imgs_size.append([h,w])
            
            batch_imgs = torch.stack(batch_imgs)
            batch_imgs = batch_imgs.cuda()
            valid_logits, attns = model(batch_imgs)
            valid_probs = torch.sigmoid(valid_logits)
            valid_probs = valid_probs.detach().cpu().numpy().tolist()
            
            
            if save_fmt == 'raw':
                try:
                    for j in range(i, min(i + batch_size, len(img_list))):
                        img_path = img_list[j]
                        cur_result = valid_probs[j - i]
                        cur_result = [round(x, 4) for x in cur_result]
                        output_result[img_path] = cur_result
                except:
                    print(i, j, len(valid_probs))
                # break
            # valid_probs = valid_probs

            elif save_fmt == 'class':
                for j in range(i, min(i + batch_size, len(img_list))):
                    img_path = img_list[j]
                    img_name = str(Path(img_path).stem)
                    img_name_withsuf = str(Path(img_path).name)
                    person_res = {}
                    attr_name_list = []
                    cur_result = valid_probs[j - i]

                    for attr_idx, p in zip(valid_idx, cur_result):
                        attr = id_2attr[str(attr_idx)]
                        attr_name_split = attr.split('|')
                        if len(attr_name_split) == 2:
                            attr_name, state = attr_name_split
                        else:
                            attr_name = attr
                            state = 'true'
                        attr_name_list.append(attr_name)
                        if attr_name not in person_res:
                            person_res[attr_name] = ['null', 0.0]
                        if p > person_res[attr_name][1]:
                            person_res[attr_name] = [state, p]

                    if args.verbose:
                        print('-'*len(img_path))
                        print(img_path)
                        for attr_name, state in person_res.items():
                            print(f'{attr_name} \tp is {state}')
                    
                    person_attr = person_res_2dict(person_res, thresh=0.5)
                    # print(person_attr)
                    person_json = deepcopy(attr_json)

                    # new_img_path = osp.join(output_dir, 'images', img_name_withsuf)
                    # shutil.copy(img_path, new_img_path)
                    person_json['imagePath'] = img_name_withsuf
                    person_json['imageHeight'] = h
                    person_json['imageWidth'] = w
                    person_json['attributes'] = build_dict(person_attr)
                    output_json_path = osp.join(output_dir, 'labels', img_name+'.json')

                    with open(output_json_path, 'w') as f:
                        json.dump(person_json, f)
            else:
                raise NotImplementedError

        if save_fmt == 'raw':
            output_path = osp.join(output_dir, f'{task}_result_{data_idx}_{date_time}.json')
            with open(output_path, 'w') as f:
                json.dump(output_result, f)

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str, default="configs/pedes_baseline/ppar_internimage_infer.yaml",
    )
    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument("--verbose", type=str2bool, default="false")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args, save_fmt='class')