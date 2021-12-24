# Training and Testing the Non-Deep Network (ParNet) on ImageNet  

## Installation

### Download ParNet and install dependecies:

```
git clone https://github.com/imankgoyal/NonDeepNetworks
cd NonDeepNetworks/imagenet
pip install -r requirements.txt
```

### Download ImageNet

For downloading the imagenet dataset, use the instructions provided here: https://github.com/pytorch/examples/tree/master/imagenet. Let `<path-to-imagenet>` be the folder where the dataset is stored. The training images are stored at `<path-to-imagenet>/train` and the validation images are stored at `<path-to-imagenet>/val`.

### Install Pytorch 1.8.2 + CUDA 11.1

To reproduce the results in the paper, please use this Pytorch version:

```
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

## Testing

### Validate on ImageNet (ILSVRC2012-val)

Download pre-trained weights-files to the `/models/` directory from: https://github.com/imankgoyal/NonDeepNetworks/releases/tag/v.0.1.0

Take a look at examples of validation commands: `validate_models.sh`. Make sure to replace `<path-to-imagenet>` with the correct path.

ParNet-XL model 224x224:
```
validate.sh --amp -b 8 --model simplenet_repvgg --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.pth.tar --num-classes 1000 <path-to-imagenet>/val 
```

ParNet-XL model with high inference resolution 352x352:
```
validate.sh --amp --model simplenet_repvgg --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/ft2_init_lr_0.001_cosine_epoch_16_is_320_we_0.0_zero_init_head_2_scale_0.5_1.0_mixup_0.1_reprob_0.6.pth.tar --input-size 3 352 352 --batch-size 128  <path-to-imagenet>/val 
```

ParNet-XL model with high inference resolution 352x352 and 10 crops:
```
validate.sh --amp --model simplenet_repvgg --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/ft2_init_lr_0.001_cosine_epoch_16_is_320_we_0.0_zero_init_head_2_scale_0.5_1.0_mixup_0.1_reprob_0.6.pth.tar --input-size 3 352 352 --ten-crop --batch-size 128  <path-to-imagenet>/val
```

### Validate on ImageNet (ILSVRC2012-val) to compare 1xGPU vs 3xGPUs

Take a look at examples of validation commands: `validate_parnet_mgpu.sh`. Make sure to replace `<path-to-imagenet>` with the correct path.

Preferably, use Pytorch 1.9.1+cu111: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```


## Training

### Train on ImageNet (ILSVRC2012-train)

Take a look at examples of training commands: `train_models.sh`. Make sure to replace `<path-to-imagenet>` with the correct dataset path. Note that we assume a node with 8 GPUS. Depending on the hardware configuration, the training script might have to be modified. 

Train ParNet-XL model for 200 epochs:

```
# ParNet-XL (200 epochs)
./distributed_train_port_auto.sh 1 0 8 localhost 27304 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin --model-cfg ./models/reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --lr 0.001 --sched cosine --opt sgd --input-size 3 224 224 --remode pixel --aa rand-m9-mstd0.5 --mixup 0.0 --epochs 200 --reprob 0.4
```


Fine-tune pre-trained ParNet-XL model for 16 epochs with high resolution 320x320:

```
# ParNet-XL (200 epochs), high resolution 320x320
./distributed_train_port_auto.sh 1 0 4 localhost 27305 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name ft2_init_lr_0.001_cosine_epoch_16_is_320_we_0.0_zero_init_head_2_scale_0.5_1.0_mixup_0.1_reprob_0.6 --model-cfg ./models/reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --finetune ./output/train/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin_0.1/last.pth.tar --epochs 16 --lr 0.001 --sched cosine --opt sgd --input-size 3 320 320 --finetune_fc --scale 0.5 1.0 --mixup 0.1 --remode pixel --reprob 0.6 --aa rand-m9-mstd0.5
```

## Licenses

### Code
The code here is licensed BSD 3-Clause License. 

## Acknowledgements
Our work builds on and uses code from [timm](https://github.com/rwightman/pytorch-image-models). We'd like to thank the authors for making these libraries available.


## Citing

### BibTeX

```
@article{goyal2021nondeep,
  title={Non-deep Networks},
  author={Goyal, Ankit and Bochkovskiy, Alexey and Deng, Jia and Koltun, Vladlen},
  journal={arXiv:2110.07641},
  year={2021}
}
```
