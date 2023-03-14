dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

pretrain_prune_finetune() {

    # pre-training
#    python train.py --exp-name $1 --arch $2 --exp-mode pretrain --configs configs/configs_crown-ibp.yml \
#    --trainer $3 --val_method $4 --gpu $5 --k 1.0 --save-dense --dataset CIFAR10 --batch-size 128 --epochs 200 --schedule_length 120;


    # pruning with FaShapley
    python train.py --is-semisup --exp-name $1 --arch $2 --exp-mode prune --configs configs/configs_crown-ibp.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense  --shapley_init \
    --source-net /home/zongwei/mintong/tmp/cifar_crown_ibp_CNN_7.pth.tar  --epochs $7 \
    --schedule_length 1 --lr 1e-6 --dataset CIFAR10 --batch-size 128;
}


arch="cifar_model_large"

(
    pretrain_prune_finetune  "cifar_large_model-trainer_crown-ibp-k_0.01-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "6"   0.05  30 ;
);


#arch="cifar_model"
#
#(
#    pretrain_prune_finetune  "cifar_model-trainer_crown-ibp_new-k_0.01-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "0"   0.05  30 ;
#);