
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

import argparse
from yolox.data.datasets.kitti_common import get_label_annos
from yolox.data.datasets.kitti_eval import get_official_eval_result


parser = argparse.ArgumentParser("Script for KITTI 2D metrics evaluation.")
parser.add_argument("--split_path", type=str, default='datasets/KITTI/ImageSets/val.txt')
parser.add_argument("--label_dir", type=str, default='datasets/KITTI/training/label_2')
parser.add_argument("--pred_dir", type=str, default='predictions')
args = parser.parse_args()


def eval(args):
    ids = list()
    for line in open(args.split_path):
        ids.append(line.strip())

    print("==> Loading detections and GTs...")
    img_ids = [int(id) for id in ids]
    dt_annos = get_label_annos(args.pred_dir)
    gt_annos = get_label_annos(args.label_dir, img_ids)

    classes = ('Car',)
    test_id = {'Car': 0}

    results_str = ''
    results_dict = {}
    for category in classes:
        cls_str, cls_dict = get_official_eval_result(gt_annos, dt_annos, test_id[category])
        results_str += cls_str
        results_dict.update(cls_dict)
    print(results_str)
    print(results_dict)
    return results_str, results_dict

if __name__ == '__main__':
    eval(args)
