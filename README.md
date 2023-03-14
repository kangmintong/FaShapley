Pre-train, prune, and fine-tune on CIFAR-10 based on CROWN-IBP training:
```bash
sh cifar_crown-ibp.sh
```
Pre-train, prune, and fine-tune on CIFAR-10 based on MixTrain training:
```bash
sh cifar_mixtrain.sh
```

Pre-train, prune, and fine-tune on SVHN based on CROWN-IBP training:
```bash
sh svhn_crown-ibp.sh
```

Pre-train, prune, and fine-tune on SVHN based on MixTrain training:
```bash
sh svhn_mixtrain.sh
```

Prune on Tiny-ImageNet based on Auto-LiRPA training:
```bash
sh tinyimagenet_auto_lirpa.sh
```

For structured pruning, use ``` shapley_init_structured.py``` instead of ```shapley_init.py```.