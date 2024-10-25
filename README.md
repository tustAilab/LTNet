<p align="center">
    <h1 align="center">A Lightweight Target-Driven Network of Stereo Matching for Inland Waterways</h1>
    <p align="center">
        Jing Su, Yiqing Zhou, Yu Zhang, Chao Wang, Yi Wei
    </p>
    <h3 align="center"><a href="https://arxiv.org/pdf/2410.07915.pdf">Paper</a>
    <div align="center"></div>
</p>

<p align="center">
    <a href="">
    <!-- <img src="https://github.com/Open-YiQingZhou/LTNet/blob/main/images/LT-network.jpg" alt="Logo" width="90%"> -->
    <img src="./images/LT-network.jpg" alt="Logo" width="90%">
    </a>
</p>

# How to use

## Environment
* NVIDIA RTX 4090
* Python 3.9
* Pytorch 2.0.1

## Install

### Create a virtual environment and activate it.

```
conda create -n LTNet python=3.9
conda activate LTNet
```
### Dependencies

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib
pip install tqdm
pip install timm==0.5.4
```

# Dataset

<details>
<summary>SceneFlow</summary>

```
./data/SceneFlow/
└───Driving/
│   └───disparity/
│   └───frames_finalpass/
└───FlyingThings/
│   └───disparity/
│   └───frames_finalpass/
└───Monkaa/
    └───disparity/
    └───frames_finalpass/
```
</details>

<details>
<summary>USVInland</summary>

```
./data/USVInland/Stereo Matching/Low_Res_640_320/
└───Disp_Map/
└───Disp_Map_Distil/
└───Left_Img_Rectified/
└───Right_Img_Rectified/
```
</details>

<!-- <details>
<summary>KITTI</summary>

```
./data/KITTI/
└───KITTI2012/data_stereo_flow/
│   └───testing/
│   │   └───colored_0/
│   │   └───colored_1/
│   └───training/
│       └───disp_occ/
│       └───colored_0/
│       └───colored_1/
└───KITTI2015/data_scene_flow/
    └───testing/
    │   └───image_2/
    │   └───image_3/
    └───training/
        └───disp_occ_0/
        └───image_2/
        └───image_3/
```
</details> -->

<details>
<summary>Spring</summary>

```
./data/Spring/
└───test/
│   └───0003/
│   │   └───frame_left/
│   │   └───frame_right/
│   └───xxxx/
│       └───frame_left/
│       └───frame_right/
└───train/
    └───0001/
    │   └───disp1_left/
    │   └───frame_left/
    │   └───frame_right/
    └───xxxx/
        └───disp1_left/
        └───frame_left/
        └───frame_right/
```
</details>

<!-- <details>
<summary>Middlebury</summary>

```
./data/Middlebury/
└───MiddEval3-GT0-H/MiddEval3/trainingH/
│   └───Adirondack/
│   └───ArtL/
│   └───......
└───MiddEval3-data-H/MiddEval3/
    └───testH/
    │   └───Australia/
    │   └───AustraliaP/
    │   └───......
    └───trainingH/
        └───Adirondack/
        └───ArtL/
        └───......
```
</details>

<details>
<summary>ETH3D</summary>

```
./data/ETH3D/
└───two_view_test/
│   └───lakeside_1l/
│   └───lakeside_1s/
│   └───......
└───two_view_training_gt/
│   └───delivery_area_1l/
│   └───delivery_area_1s/
│   └───......
└───two_view_training/
    └───delivery_area_1l/
    └───delivery_area_1s/
    └───......
```
</details> -->

#  Training
Train LTNet on Scene Flow. First training,
```
nohup python -u train_sceneflow.py --saveckpt ./checkpoints/sceneflow/first/ --batch_size 12 --test_batch_size 12 --num_workers 12 > ./logs/nohup.log 2>&1 &
```
Train LTNet on Scene Flow. Second training,
```
nohup python -u train_sceneflow.py --saveckpt ./checkpoints/sceneflow/second/ --loadckpt ./checkpoints/sceneflow/first/checkpoint_000031.ckpt --batch_size 12 --test_batch_size 12 --num_workers 12 > ./logs/nohup.log 2>&1 &
```

#  Finetune
Finetune LTNet on USVInland (using pretrained model on Scene Flow),
```
nohup python -u train_LTNet.py --dataset usvinland --maxdisp 64 --saveckpt ./checkpoints/usvinland/ --trainlist ./filenames/usvinland_train.txt --testlist ./filenames/usvinland_val.txt --batch_size 8 --test_batch_size 8 --num_workers 8 --kfold 1 > ./logs/nohup.log 2>&1 &
```
<!-- Finetune LTNet on KITTI (using pretrained model on Scene Flow, mix 12 and 15),
```
nohup python -u train_LTNet.py --dataset kitti --maxdisp 192 --saveckpt ./checkpoints/kitti/ --trainlist ./filenames/kitti12_15_all.txt --testlist ./filenames/kitti15_val.txt --batch_size 4 --test_batch_size 4 --num_workers 8 --save_freq 10 > ./logs/nohup.log 2>&1 &
```
Finetune LTNet on Middlebury (using pretrained model on Scene Flow),
```
nohup python -u train_LTNet.py --dataset middlebury --maxdisp 320 --saveckpt ./checkpoints/middlebury/ --trainlist ./filenames/middlebury_train.txt --testlist ./filenames/middlebury_val.txt --batch_size 1 --test_batch_size 1 --num_workers 8 > ./logs/nohup.log 2>&1 &
```
Finetune LTNet on ETH3D (using pretrained model on Scene Flow),
```
nohup python -u train_LTNet.py --dataset eth3d --maxdisp 64 --saveckpt ./checkpoints/eth3d/ --trainlist ./filenames/eth3d_train.txt --testlist ./filenames/eth3d_val.txt --batch_size 1 --test_batch_size 1 --num_workers 8 > ./logs/nohup.log 2>&1 &
``` -->
Finetune LTNet on Spring (using pretrained model on Scene Flow). First training,
```
nohup python -u train_LTNet.py --dataset spring --maxdisp 512 --saveckpt ./checkpoints/spring/first/ --trainlist ./filenames/spring_train.txt --testlist ./filenames/spring_val.txt --epochs 32 --lrepochs 10,14,18,22,26:2 --batch_size 8 --test_batch_size 1 --num_workers 8 > ./logs/nohup.log 2>&1 &
```
Finetune LTNet on Spring. Second training,
```
nohup python -u train_LTNet.py --dataset spring --maxdisp 512 --saveckpt ./checkpoints/spring/second/ --loadckpt ./checkpoints/spring/first/best.ckpt --trainlist ./filenames/spring_all.txt --testlist ./filenames/spring_val.txt --epochs 64 --lrepochs 10,14,18,22,26:2 --batch_size 8 --test_batch_size 1 --num_workers 8 > ./logs/nohup.log 2>&1 &
```

#  Save Disparity
Generate LTNet disparity images of Scene Flow test set,
```
python -u save_disp.py --dataset sceneflow --maxdisp 192 --testlist ./filenames/sceneflow_test.txt --loadckpt ./checkpoints/sceneflow/second/best.ckpt
```
Generate LTNet disparity images of USVInland test set,
```
python -u save_disp.py --dataset usvinland --maxdisp 64 --testlist ./filenames/usvinland_val.txt --loadckpt ./checkpoints/usvinland/kfold-distill/kfold_1/best.ckpt --kfold 1
```
<!-- Generate LTNet disparity images of KITTI test set,
```
python -u save_disp.py --dataset kitti --maxdisp 192 --testlist ./filenames/kitti15_test.txt --loadckpt ./checkpoints/kitti/best.ckpt
``` -->
Generate LTNet disparity images of Spring test set,
```
python -u save_disp.py --dataset spring --maxdisp 512 --testlist ./filenames/spring_test.txt --loadckpt ./checkpoints/spring/second/best.ckpt --submit
```
<!-- Generate LTNet disparity images of Middlebury test set (for generalization experiment),
```
python -u save_disp.py --dataset middlebury --maxdisp 320 --testlist ./filenames/middlebury_all.txt --loadckpt ./checkpoints/sceneflow/second/best.ckpt
```
Generate LTNet disparity images of ETH3D test set (for generalization experiment),
```
python -u save_disp.py --dataset eth3d --maxdisp 64 --testlist ./filenames/eth3d_all.txt --loadckpt ./checkpoints/sceneflow/second/best.ckpt
``` -->

# Distillation for USVInland

## Instructions

1. CroCo-Stereo is used as the teacher network and LTNet is used as the student network
2. Fine-tune CroCo on the USVInland dataset with 5-fold cross validation to obtain 5 dense test results, with a total of 180 disparity maps
3. Mask the 180 disparity maps with water surface to obtain the teacher label (saved as Disp_Map_Distil)
4. Pre-train LTNet using the Scene Flow dataset to obtain the pre-training weights
4. Use the pre-training weights and teacher labels as training labels to fine-tune LTNet with 5-fold cross validation to obtain the distillation weights
5. Use the distillation weights and radar gt as training and validation labels, and fine-tune LTNet again with 5-fold cross validation to obtain the final weights

## Example kfold=1
```
nohup python -u train_LTNet.py --dataset usvinland --maxdisp 64 --trainlist ./filenames/usvinland_train.txt --testlist ./filenames/usvinland_val.txt --loadckpt ./checkpoints/sceneflow/second/best.ckpt --epoch 300 --batch_size 8 --test_batch_size 8 --num_workers 8 --logdir ./checkpoints/usvinland/kfold-distill/kfold_1/logs-first --saveckpt ./checkpoints/usvinland/kfold-distill/kfold_1 --kfold 1 --distill > ./checkpoints/usvinland/kfold-distill/kfold_1/logs-first/nohup.log 2>&1 &
# Modify the best.ckpt file name to best-first.ckpt
nohup python -u train_LTNet.py --dataset usvinland --maxdisp 64 --trainlist ./filenames/usvinland_train.txt --testlist ./filenames/usvinland_val.txt --loadckpt ./checkpoints/usvinland/kfold-distill/kfold_1/best-first.ckpt --epoch 50 --lrepochs 0:10 --batch_size 8 --test_batch_size 8 --num_workers 8 --logdir ./checkpoints/usvinland/kfold-distill/kfold_1/logs --saveckpt ./checkpoints/usvinland/kfold-distill/kfold_1 --kfold 1 > ./checkpoints/usvinland/kfold-distill/kfold_1/logs/nohup.log 2>&1 &
python -u save_disp.py --dataset usvinland --maxdisp 64 --testlist ./filenames/usvinland_val.txt --loadckpt ./checkpoints/usvinland/kfold-distill/kfold_1/best.ckpt --kfold 1
```

# Acknowledgements

The project structure and portions of the code originate from [CGI-Stereo](https://github.com/gangweiX/CGI-Stereo) and [AANet](https://github.com/haofeixu/aanet). We extend our appreciation to the original authors for their remarkable work.
