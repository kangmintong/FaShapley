

pretrain_prune_finetune() {
python train.py --exp-name $1 --arch $2 --exp-mode prune --configs configs/configs_crown-ibp.yml \
--trainer $3 --val_method $4 --gpu $5 --save-dense \
--source-net ./trained_models/wide_resnet_imagenet64_200 --epochs $6 \
--schedule_length 1 --lr 1e-6 --dataset tinyimagenet --batch-size 128 --eps 0.003921568627451 ;
}

arch="wide_resnet_imagenet64"
(
    pretrain_prune_finetune  "auto_lirpa"  $arch  "lirpa"   "lirpa"  "6"  0 ;
);

