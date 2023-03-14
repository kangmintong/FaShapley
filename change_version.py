import torch

model = torch.load('./trained_models/svhn_model-trainer_mixtraink5-k_0.01-prunepochs_2'
                   '0/pretrain/latest_exp/checkpoint/checkpoint.pth.tar')


torch.save(model, './trained_models/svhn_model-trainer_mixtraink5-k_0.01-prunepochs_2'
                   '0/pretrain/latest_exp/checkpoint/checkpoint_low_version.pth.tar', _use_new_zipfile_serialization=False)