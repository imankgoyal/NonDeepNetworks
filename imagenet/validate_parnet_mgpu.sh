#!/bin/bash
# Validate Top1/Top5 accuracy on ILSVRC2012 dataset
# Requires 3 GPUs for multi-gpu testing

# Change <path-to-imagenet> to the correct path to ImageNet (ILSVRC2012)
# Uncomment line for required model and run this bash-file to train

chmod +x validate.sh

#### ParNet-L ####

# ParNet-L 1 x GPU (200 epochs) - (EMA) - planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin
# ./validate.sh --model simplenet_repvgg --model-cfg ./models/planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin.pth.tar --num-classes 1000 <path-to-imagenet>/val --use-ema -b 1 --fuse

# ParNet-L 3 x GPU (200 epochs) - (EMA) - planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin_mgpu
# ./validate.sh --model simplenet_repvgg_mgpu --model-cfg ./models/planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin_mgpu.yaml --num-classes 1000 <path-to-imagenet>/val -b 1 --fuse


#### ParNet-XL ####

# ParNet-XL 1 x GPU (200 epochs) - (EMA) - reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin
# ./validate.sh --model simplenet_repvgg --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.pth.tar --num-classes 1000 <path-to-imagenet>/val --use-ema -b 1 --fuse

# ParNet-XL 3 x GPU (200 epochs) - (EMA) - reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin_mgpu
# ./validate.sh --model simplenet_repvgg_mgpu --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin_mgpu.yaml --num-classes 1000 <path-to-imagenet>/val -b 1 --fuse


#### ResNet ####

# ResNet34
# validate.sh --model resnet34 --num-classes 1000 <path-to-imagenet>/val --checkpoint ./models/resnet34.pth.tar -b 1

# ResNet50
# validate.sh --model resnet50 --num-classes 1000 <path-to-imagenet>/val --checkpoint ./models/resnet50.pth.tar -b 1

# ResNet101
# validate.sh --model resnet101 --num-classes 1000 <path-to-imagenet>/val --checkpoint ./models/resnet101.pth.tar -b 1

