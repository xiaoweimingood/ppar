from collections import defaultdict
import os
import json
import glob
from pathlib import Path
from os import path as osp
from copy import deepcopy
from tqdm import tqdm
bool_set = set(['True', 'False'])
no_false_set = set(['overall-person_posture', 
                    'overall-person_orientation',
                    'overall-image_lighting',
                    'overall-camera_pose'])

# 从86个预测标签生成带层级的字典
def build_dict(person_attr):
    final_out = {}
    cur = final_out
    for k,v in person_attr.items():
        cur = final_out
        splitted = k.split('-', 1)
        while len(splitted) != 1:
            key = splitted[0]
            if key not in cur:
                cur[key] = {}
            cur = cur[key]
            splitted = splitted[1].split('-', 1)

        else:
            cur[splitted[0]] = v if v not in bool_set else v == 'True'

    return final_out

# def build_dict(label_log_str):
#     final_out = {}
#     cur = final_out
#     for ll in label_log_str:
#         cur = final_out
#         splitted = ll.split('-', 1)
#         while len(splitted) != 1:
#             key = splitted[0]
#             if key not in cur:
#                 cur[key] = {}
#             cur = cur[key]
#             splitted = splitted[1].split('-', 1)

#         else:
#             _k, _v = splitted[0].split('|')
#             cur[_k] = _v

#     return final_out

# 输入为attribute的json
# 输出为id_2attr
def gen_template(template_label):
    attr = template_label['attributes']
    attr_all = recurse_find_label(attr)
    attr_map = {}
    print('attr map list:')
    for i, attr in enumerate(attr_all):
        print(i, '\t', attr)
        attr_map[attr] = str(i)
    return attr_map

# 输入为attribute的json
# 输出为attr字典，键为名称，值为可选的列表
def gen_candidate(attr_json):
    attr = attr_json['attributes']
    cand = recurse_find_cand(attr)
    attr_dict = defaultdict(list)
    for c in cand:
        attr_name, attr_v = c.split('|')
        attr_dict[attr_name].append(attr_v)
    return attr_dict

def recurse_find_cand(dic):
    return_info = []
    for k, v in dic.items():
        if isinstance(v, dict):
            return_info.extend([k + '-' + x for x in recurse_find_cand(v)])
        elif isinstance(v, list):
            return_info.extend([k + '|' + str(x) for x in v])
    return return_info

def recurse_find_label(dic):
    return_info = []
    for k, v in dic.items():
        if isinstance(v, dict):
            return_info.extend([k + '-' + x for x in recurse_find_label(v) ])
        elif isinstance(v, list):
            if len(v) > 2:
                if len(v) == 3 and 'unsure' in v:
                    return_info.extend([k])
                else:
                    # False状态其实就是其它所有标签置零，不会单独存在false
                    # unsure也不会作为状态单独存在，会平摊所有概率
                    return_info.extend([k + '|' + str(x) for x in v if str(x) != 'False' and str(x) != 'unsure' ])
            else:
                return_info.extend([k])
    return return_info

# thresh：若为true的conf大于该阈值，则是真的true，否则未false
def person_res_2dict(person_res, thresh=0.3, gt=None):
    output = {}
    for attr_name, state in person_res.items():
        if attr_name in no_false_set:
            output[attr_name] = state[0]
        elif 'color' in attr_name:
            if state[1] < thresh:
                output[attr_name] = 'unsure'
            else:
                output[attr_name] = state[0]
        else:
            if state[1] < thresh:
                output[attr_name] = False
            else:
                output[attr_name] = True if state[0] == 'true' else state[0]
    if gt:
        output[gt] = True
    return output

def merge_pseudo_label(pseudo_label_dir = '/home/weimin.xiao/dataset/public_dataset/peta/peta_release/labels_pseudo',
                        gt_label_dir = '/home/weimin.xiao/dataset/public_dataset/peta/peta_release/labels_ppar',
                        output_dir = '/home/weimin.xiao/dataset/public_dataset/peta/peta_release/labels_combined'):
    pseudo_json_list = glob.glob(pseudo_label_dir + '/*.json')
    gt_json_list = glob.glob(gt_label_dir + '/*.json')
    assert len(pseudo_json_list) == len(gt_json_list)

    os.makedirs(output_dir, exist_ok=True)

    for pseudo_json_path in tqdm(pseudo_json_list):
        pseudo_label = json.load(open(pseudo_json_path, 'r'))
        label_name = Path(pseudo_json_path).name

        gt_label = json.load(open(osp.join(gt_label_dir, label_name), 'r'))
        # cur = out_label['attributes']
        final_out = overwrite_gt_to_pseudo(pseudo_label['attributes'], gt_label['attributes'])
        pseudo_label['attributes'] = final_out
        out_path = osp.join(output_dir, label_name)
        with open(out_path, 'w') as f:
            json.dump(pseudo_label, f)
        # print(final_out)
        # break

def overwrite_gt_to_pseudo(pseudo, gt):
    for k,v in pseudo.items():
        if isinstance(v, dict):
            pseudo[k] = overwrite_gt_to_pseudo(pseudo[k], gt[k])
        else:
            if gt[k] != 'UNFILLED':
                # print(f'overwriting attr {k} from {pseudo[k]} to {gt[k]}')
                pseudo[k] = gt[k]
    return pseudo

if __name__ == '__main__':
    merge_pseudo_label()