# Modified YOLOX for 2D object detection on KITTI

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Performance](#performance)
5. [Model_info](#model_info)
6. [Acknowledgement](#acknowledgement)

### Installation

1. Environment requirement
    - anaconda3
    - python 3.6
    - pytorch, torchvision, numpy, pillow etc, refer to [requirements.txt](requirements.txt) for more details.
    - vai_q_pytorch(Optional, required for quantization)
    - XIR Python frontend (Optional, required for dumping xmodel)

2. Installation with Docker

   First refer to [vitis-ai](https://github.com/Xilinx/Vitis-AI/tree/master/) to obtain the docker image.
   ```bash
   conda activate vitis-ai-pytorch
   pip install --user -r requirements.txt 
   cd code
   pip install --user -v -e .
   pip install --user pycocotools
   cd ..
   ```

### Preparation

1. Dataset description

Based on the KITTI dataset, 3 classes are used: Car, Pedestrian and Cyclist. The original KITTI training set includes 7481 samples, which is split into two parts: training set = 3712, validation set = 3769.

2. Download KITTI dataset and create some directories first:
  ```plain
  └── data
       └── KITTI
             ├── ImageSets    <-- is contained in this code repo
             |   ├── test.txt
             |   ├── train.txt
             |   ├── trainval.txt
             |   └── val.txt
             ├── training    <-- 7481 train data
             |   ├── image_2
             |   ├── calib
             |   ├── label_2
             |   ├── ...
             └── testing     <-- 7580 test data
                 ├── image_2
                 ├── calib
                 └── ...
  ```

### Train/Eval/QAT

1. Evaluation
  - Execute run_eval.sh.
  ```shell
  bash code/run_eval.sh
  ```

2. Training
  ```shell
  bash code/run_train.sh
  ```

3. Model quantization and xmodel dumping
  ```shell
  bash code/run_quant.sh
  ```

4. QAT(Quantization-Aware-Training), model converting and xmodel dumping
  - Configure the variables and in `code/run_qat.sh`, read the steps(including QAT, model testing, model converting and xmodel dumping) in the script and run the step you want.
  ```shell
  bash code/run_qat.sh
  ```

### Prepare 2d detection prediction for CLOCs training
1. Configure path in `code/predict_kitti.sh` and run the script
  ```shell
  bash code/predict_kitti.sh
  ```

### Performance
|Metric | Float | Quantized | QAT |
| -     | -    | - | - |
|Car 2d bbox AP@0.70 (easy,moderate,hard)|91.08,89.40,85.04|48.63,47.32,46.19|93.78,89.50,85.04|


### Model_info

1. Data preprocess
  ```
  data channels order: BGR
  keeping the aspect ratio of H/W, resize image with bilinear interpolation to shape of (H,W)=(H*1248/W,1248), pad the image with (114,114,114) along the height side to get image with shape of (H,W)=(384,1248)
  ``` 


### Acknowledgement

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)
