mdt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

pretrain_prune_finetune() {

    # pre-training
    python train.py --exp-name $1 --arch $2 --exp-mode pretrain --configs configs/configs_mixtrain.yml \
    --trainer $3 --val_method $4 --gpu $5 --k 1.0 --save-dense --dataset SVHN --schedule_length 15 \
    --mixtraink $8 --batch-size 50;

    # pruning
    python -W ignore train.py --exp-name $1 --arch $2 --exp-mode prune --configs configs/configs_mixtrain.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense --shapley_init \
    --source-net ./trained_models/$1/pretrain/latest_exp/checkpoint/model_best.pth.tar --epochs 30 \
    --schedule_length 0 --lr 0.00001 --dataset SVHN  --mixtraink $8 --batch-size 50;
}

arch="cifar_model_large"

(
    pretrain_prune_finetune  "svhn_model_large-trainer_mixtraink1-k_0.01-prunepochs_20"  $arch  "mixtrain"   "mixtrain"  "7"   0.01  20 1 ;
);


arch="cifar_model"

(
    pretrain_prune_finetune  "svhn_model-trainer_mixtraink5-k_0.01-prunepochs_20"  $arch "mixtrain"   "mixtrain"  "1"   0.01  100 5;
);