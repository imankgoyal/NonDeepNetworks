#!/bin/bash

# Change <path-to-imagenet> to the correct path to ImageNet (ILSVRC2012)
# By default it uses 8 x GPUs with batch size -b 128
# Uncomment line for required model and run this bash-file to train

chmod +x distributed_train_port_auto.sh

# ParNet-S
#./distributed_train_port_auto.sh 1 0 8 localhost 27300 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name planes_92_192_384_1280_num_blocks_5_6_6_1_sebv_13 --model-cfg ./models/planes_92_192_384_1280_num_blocks_5_6_6_1_sebv_13.yaml --lr 0.001 --sched cosine --opt sgd --input-size 3 224 224 --remode pixel --aa rand-m9-mstd0.5 --mixup 0.0 --epochs 120 --reprob 0.2

# ParNet-M
#./distributed_train_port_auto.sh 1 0 8 localhost 27301 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name planes_128_256_512_2048_num_blocks_5_6_6_1_sebv_13 --model-cfg ./models/planes_128_256_512_2048_num_blocks_5_6_6_1_sebv_13.yaml --lr 0.001 --sched cosine --opt sgd --input-size 3 224 224 --remode pixel --aa rand-m9-mstd0.5 --mixup 0.0 --epochs 120 --reprob 0.2

# ParNet-L
#./distributed_train_port_auto.sh 1 0 8 localhost 27302 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin --model-cfg ./models/planes_160_320_640_2560_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --lr 0.001 --sched cosine --opt sgd --input-size 3 224 224 --remode pixel --aa rand-m9-mstd0.5 --mixup 0.0 --epochs 120 --reprob 0.4

# ParNet-XL
#./distributed_train_port_auto.sh 1 0 8 localhost 27303 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin --model-cfg ./models/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --lr 0.001 --sched cosine --opt sgd --input-size 3 224 224 --remode pixel --aa rand-m9-mstd0.5 --mixup 0.0 --epochs 120 --reprob 0.4

# ParNet-XL (200 epochs)
#./distributed_train_port_auto.sh 1 0 8 localhost 27304 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin --model-cfg ./models/reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --lr 0.001 --sched cosine --opt sgd --input-size 3 224 224 --remode pixel --aa rand-m9-mstd0.5 --mixup 0.0 --epochs 200 --reprob 0.4

# ParNet-XL (200 epochs), fine-tuning the previous trained model, with high resolution 320x320
#./distributed_train_port_auto.sh 1 0 4 localhost 27305 <path-to-imagenet> --model simplenet_repvgg -b 128 --warmup-epochs 0 --amp -j 16 --model-ema --model-ema-decay 0.9999 --smoothing 0.1 --exp-name ft2_init_lr_0.001_cosine_epoch_16_is_320_we_0.0_zero_init_head_2_scale_0.5_1.0_mixup_0.1_reprob_0.6 --model-cfg ./models/reg_se13_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin.yaml --finetune ./output/train/reg_se13_cosine_planes_200_400_800_3200_num_blocks_5_6_6_1_sebv_13_dropout_lin_0.1/last.pth.tar --epochs 16 --lr 0.001 --sched cosine --opt sgd --input-size 3 320 320 --finetune_fc --scale 0.5 1.0 --mixup 0.1 --remode pixel --reprob 0.6 --aa rand-m9-mstd0.5

