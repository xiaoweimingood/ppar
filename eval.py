import argparse
import os


from dataset.augmentation import get_transform
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics
from models.base_block import FeatClassifier
from models.backbone import swin_transformer
from models.head import mldecoder

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool

set_seed(605)


def main(cfg, args):
    # exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    # model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    _, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    valid_set = PedesAttr(cfg=cfg,
                          split=cfg.DATASET.VAL_SPLIT,
                          transform=valid_tsfm,
                          target_transform=cfg.DATASET.TARGETTRANSFORM)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}')

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE,
                                        cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=88,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=cfg.CLASSIFIER.SCALE)

    model = FeatClassifier(backbone, classifier)

    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()

    model_dir = '/public/191-aiprime/weimin.xiao/projects/Rethinking_of_PAR/exp_result/PPAR/swinb.colorjitter.asl.adamw.batch6-1.mldecoder/img_model'
    pth = 'ckpt_max_2023-05-06_11-15-16.pth'
    model = get_reload_weight(model_dir, model, pth=pth)
    model = model.cuda()
    model.eval()
    preds_probs = []
    gt_list = []
    path_list = []

    attn_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            valid_logits, attns = model(imgs, gt_label)

            valid_probs = torch.sigmoid(valid_logits[0])

            path_list.extend(imgname)
            gt_list.append(gt_label.cpu().numpy())
            preds_probs.append(valid_probs.cpu().numpy())

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    if cfg.METRIC.TYPE == 'pedestrian':
        valid_result = get_pedestrian_metrics(gt_label, preds_probs)
        valid_map, _ = get_map_metrics(gt_label, preds_probs)

        print(
            f'Evaluation on test set, \n',
            'ma: {:.4f},  map: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'
            .format(valid_result.ma, valid_map, np.mean(valid_result.label_f1),
                    np.mean(valid_result.label_pos_recall),
                    np.mean(valid_result.label_neg_recall)),
            'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                valid_result.instance_acc, valid_result.instance_prec,
                valid_result.instance_recall, valid_result.instance_f1))

        # with open(os.path.join(model_dir, 'results_test_feat_best.pkl'),
        #           'wb+') as f:
        #     pickle.dump(
        #         [valid_result, gt_label, preds_probs, attn_list, path_list],
        #         f,
        #         protocol=4)


def argument_parser():
    parser = argparse.ArgumentParser(
        description="attribute recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        type=str,
        default='configs/pedes_baseline/ppar.yaml'
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)
