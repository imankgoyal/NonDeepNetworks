#!/bin/bash

# Change <path-to-imagenet> to the correct path to ImageNet (ILSVRC2012)
# By default it uses 8 x GPUs with batch size -b 128
# Uncomment line for required model and run this bash-file to train

chmod +x validate.sh

# ParNet-S - planes_92_192_384_1280_num_blocks_5_6_6_1_sebv_13
# ./validate.sh --amp -b 8 --model simplenet_repvgg --model-cfg ./models/planes_92_192_384_1280_num_blocks_5_6_6_1_sebv_13.yaml --checkpoint ./models/planes_92_192_384_1280_num_blocks_5_6_6_1_sebv_13.pth.tar --num-classes 1000 <path-to-imagenet>/val

# ParNet-M - planes_128_256_512_2048_num_blocks_5_6_6_1_sebv_13
# ./validate.sh --amp -b 8 --model simplenet_repvgg --model-cfg ./models/planes_128_256_512_2048_num_blocks_5_6_6_1_sebv_13.yaml --checkpoint ./models/planes_128_256_512_2048_num_blocks_5_6_6_1_sebv_13.pth.tar --num-classes 1000 <path-to-imagenet>/val

# ParNet-L - planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin
# ./validate.sh --amp -b 8 --model simplenet_repvgg --model-cfg ./models/planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin.pth.tar --num-classes 1000 <path-to-imagenet>/val

# ParNet-XL - reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin
# ./validate.sh --amp -b 8 --model simplenet_repvgg --model-cfg ./models/reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.pth.tar --num-classes 1000 <path-to-imagenet>/val

# ParNet-XL(200 epochs) - reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin
# ./validate.sh --amp -b 8 --model simplenet_repvgg --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.pth.tar --num-classes 1000 <path-to-imagenet>/val

# ParNet-XL(200 epochs, 352x352) - reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin
# ./validate.sh --amp --model simplenet_repvgg --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/ft2_init_lr_0.001_cosine_epoch_16_is_320_we_0.0_zero_init_head_2_scale_0.5_1.0_mixup_0.1_reprob_0.6.pth.tar --input-size 3 352 352 --batch-size 128  <path-to-imagenet>/val

# ParNet-XL(200 epochs, 10 crops, 352x352) - reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin
# ./validate.sh --amp --model simplenet_repvgg --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --checkpoint ./models/ft2_init_lr_0.001_cosine_epoch_16_is_320_we_0.0_zero_init_head_2_scale_0.5_1.0_mixup_0.1_reprob_0.6.pth.tar --input-size 3 352 352 --ten-crop --batch-size 128  <path-to-imagenet>/val
