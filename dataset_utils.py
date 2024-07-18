import pickle
import os
import shutil
from PIL import Image
from copy import deepcopy
from pathlib import Path
from os import path as osp
import json
import glob
from tqdm import tqdm


def rename():
    dataset_pkl_path = f'data/PPAR/dataset_all.pkl'

    replace_dict = {
        '码头_九号码头球': 'matou_jiuhaomatouqiu',
        '模拟': 'moni',
    }

    f = open(dataset_pkl_path, 'rb')
    c = pickle.load(f)
    print('----------------------------')
    attr_list = sorted(c.attr_name)
    new_paths = []
    for i, img_name in enumerate(c.image_name):
        new_path = deepcopy(img_name)
        print(i)
        for k, v in replace_dict.items():
            new_path = new_path.replace(k, v)
            if new_path != img_name:
                print('replaced')
            try:
                print(new_path)
            except:
                cur_dir = os.path.dirname(new_path)
                cur_name = os.path.basename(new_path)

                cur_name = cur_name.split('_')[-1]
                new_path = os.path.join(cur_dir, cur_name)
                print(new_path)

        if 'spc' in new_path:
            new_path = new_path.split('-', 2)[-1]
            print(new_path)
        new_path = new_path.replace('/aiprime/weimin.xiao/dataset/person_attr',
                                    '/public/share/weimin.xiao/person_attr')
        new_paths.append(new_path)
        if i == 17:
            print(new_path)
    c.image_name = new_paths
    with open('dataset_all_zj.pkl', 'wb') as f:
        pickle.dump(c, f)


def relabel_fromtxt(task='helmet'):

    # target = f'/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch_p0.1_{task}/labels'
    target = '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_dml/labels'
    # ref_txt = f'filelist/{task}_40k.txt'
    ref_txt = f'/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_dml/{task}.txt'
    print(f'working on task {task}')
    label_map = {
        'safetyharness': {
            '0': False,
            '1': True
        },
        'lifejacket': {
            '0': False,
            '1': 'belt',
            '2': 'neck',
            '3': 'vest'
        },
        'smoke': {'0': False,
                  '1': True},

        'helmet': {'0': False,
                  '1': True},
        'incomplete': {'0': False,
                  '1': True},
        'lower_block': {'0': False,
                  '1': True},
        'upper_block': {'0': False,
                  '1': True},
    }
    txt = open(ref_txt).readlines()
    imgdir_list = [x.strip().rsplit(' ', 1) for x in txt]
    for img_path, img_label in tqdm(imgdir_list):
        img_name = str(Path(img_path).stem)
        json_path = osp.join(target, f'{img_name}.json')
        img_attr = json.load(open(json_path, 'r'))

        replace_label = label_map[task][img_label]

        if task == 'lifejacket':
            img_attr['attributes']['body_upper']['lifejacket'] = replace_label
            if replace_label:
                img_attr['attributes']['body_upper']['safety_belt'] = False

        elif task == 'safetyharness':
            img_attr['attributes']['body_upper']['safety_belt'] = replace_label
            if replace_label:
                img_attr['attributes']['body_upper']['lifejacket'] = False

        elif task == 'helmet':
            img_attr['attributes']['head_top']['helmet'] = replace_label
            if replace_label:
                img_attr['attributes']['head_top']['hat'] = False

        elif task == 'smoke':
            img_attr['attributes']['overall'].pop('cigarette_smoking', None)
            img_attr['attributes']['overall'].pop('cigarette_holding', None)
            img_attr['attributes']['overall']['person_action']['cigarette_smoking'] = replace_label
            if replace_label:
                img_attr['attributes']['overall']['person_action']['cigarette_holding'] = True

        elif task == 'incomplete':
            img_attr['attributes']['overall']['is_incomplete'] = replace_label

        elif task == 'lower_block':
            img_attr['attributes']['body_lower']['blocked'] = replace_label

        elif task == 'upper_block':
            img_attr['attributes']['body_upper']['blocked'] = replace_label

        with open(json_path, 'w') as f:
            json.dump(img_attr, f)


def compute_acc(
    pred_json='preds_safetyharness_v2/lifejacket_result_0_2023_09_14_15_00.json',
    gt_txt='/public/191-aiprime/weimin.xiao/projects/timm/flist/lifejacket/val.txt'
):

    label_map = {53: 2, 54: 1, 55: 3}
    # label_map = {50: 1}

    gt = open(gt_txt, 'r').readlines()
    gt = [x.strip().rsplit(' ', 1) for x in gt]
    gt = {k: v for k, v in gt}
    gt = {
        k.replace('/aiprime/', '/public/191-aiprime/'): v
        for k, v in gt.items()
    }

    pred = json.load(open(pred_json, 'r'))

    for thresh in range(5, 100, 5):
        t, f = 0, 0
        thresh /= 100
        for img_path, p in pred.items():
            max_idx = -1
            max_p = -1
            # for p_idx in range(50, 51):
            for p_idx in range(53, 56):
                if p[p_idx] > max_p:
                    max_p = p[p_idx]
                    max_idx = p_idx

            if max_p > thresh:
                cur_label = label_map[max_idx]
            else:
                cur_label = 0

            if int(cur_label) == int(gt[img_path]):
                t += 1
            else:
                f += 1

            # print(f'pred:{cur_label}, gt:{gt[img_path]}')
        acc = round(t / (t + f) * 100, 5)
        print(f'acc: {acc}, thresh: {thresh}')

def find_intersect(dir='/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_dml/5label',
                   out='/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_dml/5label_intersection'):
    labels = os.listdir(dir)
    img_list_all = []
    for label in labels:
        img_list = glob.glob(osp.join(dir, label, '*/*'))
        img_list_sure = [x for x in img_list if 'unsure' not in x]
        print(f'label {label} img num is {len(img_list_sure)}')
        img_list_all.append(img_list_sure)
    
    img_intersect_set = set([osp.basename(x) for x in img_list_all[0]])
    for img_list in img_list_all[1:]:
        img_name_set = set([osp.basename(x) for x in img_list])
        img_intersect_set = img_intersect_set & img_name_set

    print(len(img_intersect_set))
    
    os.makedirs(out, exist_ok=True)

    for label, img_list in zip(labels, img_list_all):
        label_txt = []
        for img_path in img_list:
            img_name = osp.basename(img_path)
            if img_name in img_intersect_set:
                new_path = osp.join(out, img_name)
                if not osp.exists(new_path):
                    shutil.copy(img_path, new_path)
                img_label = str(Path(img_path).parent.name)
                label_txt.append(f'{new_path} {img_label}')
        
        with open(f'{out}/{label}.txt', 'w') as f:
            for line in label_txt:
                f.writelines(f'{line}\n')
    
def compute_mse():
    out = []
    files = glob.glob('preds_safetyharness/*')
    with open('/public/191-aiprime/weimin.xiao/projects/timm/flist/safetyharness/train.txt')as f:
        f.read()
    for file in files:
        info = json.load(open('file', 'r'))
        for path, label in info.items():
            label[50]

# compute_acc()
relabel_fromtxt(task='smoke')
relabel_fromtxt(task='helmet')
relabel_fromtxt(task='incomplete')
relabel_fromtxt(task='lower_block')
relabel_fromtxt(task='upper_block')
# find_intersect()