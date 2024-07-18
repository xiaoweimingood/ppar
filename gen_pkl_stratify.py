from easydict import EasyDict
import json
import pickle
from glob import glob
import numpy as np
from pathlib import Path
import os
from os import path as osp
import random
from tqdm import tqdm
from collections import defaultdict
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
# from sentence_transformers import SentenceTransformer
root = '/public/191-aiprime/weimin.xiao/dataset/person_attr/pseudo'
save_dir = f'{root}/tjg_uniform_20240712'
out_name = save_dir.rsplit('/', 1)[1] + '.pkl'
os.makedirs(save_dir, exist_ok=True)

def main():
    dataset = EasyDict()
    dataset.description = 'PPAR'
    dataset.reorder = 'group_order'
    dataset.root = ''
    dataset_dir = [
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/aisutdio_train', 
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/spc',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch3_safetybelt',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch4',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch5_smoke',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch6_goggles',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch7',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch8_gt_20230616',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch8_shadow',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch8_vest',
                    # '/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch8_20230707',
    
                    save_dir
                   ]

    # init block
    img_dir =  [Path(x).joinpath('images') for x in dataset_dir]
    # labels_dir = dataset_dir.joinpath('labels')
    template_path = '/public/191-aiprime/weimin.xiao/dataset/person_attr/attributes_1_0_6.json'
    template_label = json.load(open(template_path, 'r'))
    candidate = gen_candidate(template_label)
    attr_2id = gen_template(template_label)
    id_2attr = {v:k for k,v in attr_2id.items()}
    with open('id_2attr.json', 'w') as f:
        json.dump(id_2attr, f)
    num_attr = len(attr_2id)
    print(f'num of labels is {num_attr}')
    attr_unsure_score = compute_attr_unsure_score(candidate)
    print(attr_unsure_score)

    # image_name block
    imgs = []
    for img_d in img_dir:
        cur_img_list = glob(osp.join(img_d, '*.*'))
        print(f'{len(cur_img_list)} images found at {img_d}')
        imgs.extend(cur_img_list)
    img_len = len(imgs)
    print(f'{img_len} images found at {img_dir}')
    
    random.seed(42)
    random.shuffle(imgs)
    img_names = [osp.basename(x) for x in imgs]
    dataset.image_name = imgs

    # partition block
    img_idxs = list(range(img_len))
    n_train = int(img_len * 0.9)
    dataset.partition = EasyDict()

    idx_train = np.array(img_idxs[:n_train])
    idx_val = np.array(img_idxs[n_train:])

    dataset.partition.train = idx_train
    dataset.partition.val = idx_val

    # attr_name block
    dataset.attr_name = list(attr_2id.keys())

    # label_idx block
    eval_idx = []
    color_idx = []

    eval_idx = list(id_2attr.keys())
    # for idx, attr in id_2attr.items():
        # if 'color' in attr:
        #     color_idx.append(idx)
        # else:
        #     eval_idx.append(idx)
    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = eval_idx
    dataset.label_idx.color = color_idx

    # label block
    all_label = []
    for img_path in tqdm(imgs):
        img_path = Path(img_path)
        img_name_withsuffix = img_path.name
        img_name = img_path.stem
        label_name = str(img_name) + '.json'
        labels_dir = Path(img_path).parent.parent.joinpath('labels')
        label_path = labels_dir.joinpath(label_name)
        label_info = json.load(open(label_path, 'r'))
        # print(label_path)
        label = label_info['attributes']
        
        label_onehot = [0] * num_attr

        label_withname = recurse_find_filled_label(label)
        for attr in label_withname:
            # print(attr)
            
            if attr not in attr_2id:
                # 若某项为false或者true,相关的项的label都置为0或1
                if attr.endswith('True'):
                    attr = attr[:-5]
                    label_onehot[attr_2id[attr]] = 1

                # 标签默认都为0， 跳过这一步
                elif attr.endswith('False'):
                    pass
                #     attr = attr[:-6]
                #     for k,v in attr_2id.items():
                #         if attr in k:
                #             label_onehot[v] = 0

                # 若某项为unsure，将其它选项置为均值
                elif attr.endswith('unsure'):
                    attr_name = attr[:-7]
                    for k,v in attr_2id.items():
                        if attr_name in k:
                            # unsure的置为0
                            label_onehot[v] = 0
                            # unsure的平摊
                            # label_onehot[v] = attr_unsure_score[attr_name]
                else:
                    print(f'illegal ending with string {attr} at img path {img_path}')
                    # raise Exception
            # 某项确实存在于字典中
            else:
                label_onehot[attr_2id[attr]] = 1

        all_label.append(label_onehot)


    img_idxs = np.array(img_idxs)
    all_label = np.array(all_label)
    # x_train, y_train, x_test, y_test = iterative_train_test_split(img_idxs, all_label, test_size=0.1)

    dataset.label = np.array(all_label)

    # weight block
    dataset.weight_train = np.mean(dataset.label[idx_train], axis=0).astype(np.float32)
    dataset.weight_val = np.mean(dataset.label[idx_val], axis=0).astype(np.float32)

    with open(osp.join(save_dir, out_name), 'wb') as f:
        pickle.dump(dataset, f)

def gen_candidate(template_label):
    attr = template_label['attributes']
    cand = recurse_find_cand(attr)
    attr_dict = defaultdict(list)
    for c in cand:
        attr_name, attr_v = c.split('|')
        attr_dict[attr_name].append(attr_v)
    return attr_dict

def gen_template(template_label):
    attr = template_label['attributes']
    attr_all = recurse_find_label(attr)
    attr_map = {}
    print('attr map list:')
    for i, attr in enumerate(attr_all):
        print(i, '\t', attr)
        attr_map[attr] = i
    return attr_map

def recurse_find_cand(dic):
    return_info = []
    for k, v in dic.items():
        if isinstance(v, dict):
            return_info.extend([k + '_' + x for x in recurse_find_cand(v)])
        elif isinstance(v, list):
            return_info.extend([k + '|' + str(x) for x in v])
    return return_info

def recurse_find_filled_label(dic):
    return_info = []
    for k, v in dic.items():
        if isinstance(v, dict):
            return_info.extend([k + '_' + x for x in recurse_find_filled_label(v)])
        elif isinstance(v, bool) or isinstance(v, str):
            return_info.extend([k + '|' + str(v)])
    return return_info

def recurse_find_label(dic):
    return_info = []
    for k, v in dic.items():
        if isinstance(v, dict):
            return_info.extend([k + '_' + x for x in recurse_find_label(v) ])
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

# 计算不确定的得分
def compute_attr_unsure_score(candidate):
    attr_unsure_score = {}
    for attr_name, attr_v_list in candidate.items():
        if 'unsure' in attr_v_list:
            attr_unsure_score[attr_name] = round(1 / (len(attr_v_list) - 1), 3)
    return attr_unsure_score



def modify_pkl(pkl_path=f'{save_dir}/{out_name}'):
    f = open(pkl_path, 'rb')
    c = pickle.load(f)
    
    all_label = c.label
    n_img = all_label.shape[0]
    img_idxs = np.array(list(range(n_img))).reshape(-1, 1)
    # print(img_idxs.shape, all_label.shape)
    x_train, y_train, x_test, y_test = iterative_train_test_split(img_idxs, all_label, test_size=0.1)
    idx_train = x_train.squeeze()
    idx_val = x_test.squeeze()
    print(x_train, x_test)

    c.partition.train = idx_train
    c.partition.val = idx_val

    with open(osp.join(save_dir, out_name), 'wb') as f:
        pickle.dump(c, f)

# def get_label_embeds(labels):
#     model = SentenceTransformer('all-mpnet-base-v2')
#     embeddings = model.encode(labels)
#     return embeddings

# def add_attr_vector_2pkl(pkl_path='data/PPAR/dataset_all_nopesudo.pkl'):
#     attr_words = [
#     "shadow",
#     "reflect",
#     "other",
#     "incomplete",
#     "bitrate error",
#     "crowd",
#     "cigarette smoking",
#     "cigarette holding",
#     "phone playing",
#     "phone calling",
#     "pose stand",
#     "pose sit",
#     "pose ride bicycle",
#     "pose push bicycle",
#     "pose bow",
#     "pose crouch",
#     "pose fall",
#     "pose jump",
#     "pose climb",
#     "orientation front",
#     "orientation side",
#     "orientation back",
#     "normal light",
#     "dark light",
#     "bright light",
#     "camera up to down",
#     "camera down to up",
#     "camera horizontal",
#     "camera top to bottom",
#     "head top blocked",
#     "helmet",
#     "hat",
#     "head top color mix",
#     "head top color red",
#     "head top color orange",
#     "head top color yellow",
#     "head top color green",
#     "head top color blue",
#     "head top color purple",
#     "head top color pink",
#     "head top color black",
#     "head top color white",
#     "head top color grey",
#     "head top color brown",
#     "face blocked",
#     "glasses",
#     "goggles",
#     "welding mask",
#     "transparent mask",
#     "upper blocked",
#     "safety belt",
#     "upper cloth",
#     "upper uniform",
#     "lifejacket neck",
#     "lifejacket belt",
#     "lifejacket vest",
#     "short sleeve",
#     "upper green reflective_vest",
#     "upper yellow reflective_vest",
#     "upper orange reflective_vest",
#     "upper color mix",
#     "upper color red",
#     "upper color orange",
#     "upper color yellow",
#     "upper color green",
#     "upper color blue",
#     "upper color purple",
#     "upper color pink",
#     "upper color black",
#     "upper color white",
#     "upper color grey",
#     "upper color brown",
#     "lower blocked",
#     "lower cloth",
#     "lower uniform",
#     "short trousers",
#     "lower color mix",
#     "lower color red",
#     "lower color orange",
#     "lower color yellow",
#     "lower color green",
#     "lower color blue",
#     "lower color purple",
#     "lower color pink",
#     "lower color black",
#     "lower color white",
#     "lower color grey",
#     "lower color brown",
#     ]

#     f = open(pkl_path, 'rb')
#     c = pickle.load(f)
#     c.attr_words = np.array(attr_words)
#     c.attr_vectors = get_label_embeds(attr_words)

#     with open(pkl_path, 'wb') as f:
#         pickle.dump(c, f)

def gen_test_pkl(pkl_path='data/PPAR/dataset_all.pkl'):
    f = open(pkl_path, 'rb')
    c = pickle.load(f)
    c.partition.train = c.partition.train[:400]
    c.partition.val = c.partition.val[:100]

    with open('data/PPAR/dataset_part.pkl', 'wb') as f:
        pickle.dump(c, f)

def rm_nolabel_img(json_dir='/public/191-aiprime/weimin.xiao/dataset/person_attr/ppar_batch8_gt_20230616/labels'):
    json_list = glob(json_dir+'/*.json')
    img_list = glob(str(Path(json_dir).parent.joinpath('images'))+'/*')
    img_set = set([Path(x).stem for x in img_list])
    json_set = set([Path(x).stem for x in json_list])

    extra_img = img_set - json_set
    print(f'{len(extra_img)} imgs need to be removed')
    for img_name in list(extra_img):
        img_path = Path(json_dir).parent.joinpath('images').joinpath(f'{img_name}.jpg')
        print(img_path)
        os.remove(img_path)
    print('images removed')
    # for json_path in json_list:

main()
# add_attr_vector_2pkl()
modify_pkl()
# gen_test_pkl()

# rm_nolabel_img()