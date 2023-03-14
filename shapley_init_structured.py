from utils.eval import accuracy
from utils.adv import trades_loss
import numpy as np
from symbolic_interval.symbolic_network import sym_interval_analyze, naive_interval_analyze
from crown.eps_scheduler import EpsilonScheduler
from crown.bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel, Flatten
from models.layers import SubnetConv, SubnetLinear
from tqdm import tqdm
from torch.nn.parameter import Parameter
from utils.model import set_prune_rate_model
import numpy as np
import torch
import random
import copy
import torch.nn.functional as F



class protocol3():

    def __init__(self, model, criterion, loader, device, args, optimizer=None):
        self.loader = loader
        self.device = device
        self.criterion = criterion
        model.train()
        self.model = model
        self.ckp_name = f"net_params_{device}.pkl"
        torch.save(model, self.ckp_name)
        self.args = args
        self.optimizer = optimizer

        self.idx_to_i2 = {}

        self.signal = []


        _, _, self.num_weights = self.get_layers(self.model)
        self.shapley = np.zeros((self.num_weights,))
        self.space = list(range(self.num_weights))
        self.mask = np.ones((self.num_weights), dtype=int)

        self.method = 'ours'
        self.calculate = True
        self.signed = False
        self.random_prune = False

        if self.method == 'ours':
            self.sv_samples = 30
            self.sampling_ratio = 0.1
        else:
            self.sv_samples = 1
            self.sampling_ratio = 0.0
        # self.sv_samples = 1000   # 1000
        # self.sampling_ratio = 0.9  # 0.9

        self.prune_sequence = []



    def get_layers(self, model):
        layers = []
        num_weights = 0
        start = 0
        idx_to_layer = {}
        for layer in model.modules():
            if isinstance(layer, SubnetLinear):
                if layer.weight.shape[1]==4096:  # 2048 # 4096
                    continue

                layers.append(layer)
                num_weights += layer.weight.shape[1]

                for i2 in range(layer.weight.shape[1]):
                    idx_to_layer[start] = layer
                    self.idx_to_i2[start] = i2
                    start += 1
                # print(layer.weight.shape)


            elif isinstance(layer, SubnetConv):
                layers.append(layer)
                num_weights += layer.weight.shape[0]
                # print(layer.weight.shape)

                for i2 in range(layer.weight.shape[0]):
                    idx_to_layer[start] = layer
                    self.idx_to_i2[start] = i2
                    start += 1

        return layers, idx_to_layer, num_weights


    def run_taylor(self, ratio, step=0, total_step=0):
        if self.calculate:
            shapley = self.update_shapley()
            torch.save(shapley, f'shapley/structured_{self.args.dataset}_{self.args.trainer}_{self.args.arch}_{self.sampling_ratio}_{self.sv_samples}_{step}_{total_step}')
        else:
            shapley = torch.load(f'shapley/structured_{self.args.dataset}_{self.args.trainer}_{self.args.arch}_{self.sampling_ratio}_{self.sv_samples}_{step}_{total_step}')
        pruned_index = np.argsort(shapley)

        self.model = torch.load(self.ckp_name)
        self.prune_final(pruned_index, self.model, int((len(shapley))*ratio))

        # self.prune_final(pruned_index, self.model, int((len(shapley)) * ratio))
        return self.model

    def update_shapley(self):

        method = self.method
        for i in tqdm(range(self.sv_samples)):
            model_tmp = torch.load(self.ckp_name)

            S = random.sample(self.space, int(len(self.space) * self.sampling_ratio))
            self.prune(S, model_tmp)
            self.run_all_forward_and_backward(model_tmp)

            start = 0
            for layer in model_tmp.modules():
                n_weights = 0
                if isinstance(layer, SubnetLinear):
                    if layer.weight.shape[1] == 4096: # 2048 4096
                        continue

                    n_weights = layer.weight.shape[1]
                    if method=='weight':
                        sh = torch.abs((layer.weight.data)).mean(
                            dim=0).clone().cpu().numpy().flatten()
                    elif method=='gradient':
                        sh = torch.abs((layer.weight.grad.data)).mean(
                            dim=0).clone().cpu().numpy().flatten()
                    elif method=='ours':
                        sh = torch.abs((layer.weight.grad.data * layer.weight.data)).mean(
                            dim=0).clone().cpu().numpy().flatten()
                    elif method=='L2':
                        sh = torch.abs((layer.weight.data)).square().mean(
                            dim=0).clone().cpu().numpy().flatten()


                    self.shapley[start: start+n_weights] += sh
                elif isinstance(layer, SubnetConv):

                    n_weights = layer.weight.shape[0]

                    if method=='weight':
                        sh = torch.abs((layer.weight.data)).mean(
                            dim=3).mean(dim=2).mean(dim=1).clone().cpu().numpy().flatten()
                    elif method=='gradient':
                        sh = torch.abs((layer.weight.grad.data)).mean(
                            dim=3).mean(dim=2).mean(dim=1).clone().cpu().numpy().flatten()
                    elif method=='ours':
                        sh = torch.abs((layer.weight.grad.data * layer.weight.data)).mean(
                            dim=3).mean(dim=2).mean(dim=1).clone().cpu().numpy().flatten()
                    elif method == 'L2':
                        sh = torch.abs((layer.weight.data)).square().mean(
                            dim=3).mean(dim=2).mean(dim=1).clone().cpu().numpy().flatten()

                    self.shapley[start: start + n_weights] += sh
                start += n_weights


        return self.shapley


    def prune(self, index, model):
        layers, idx_to_layer, num_weights = self.get_layers(model)

        for ind in index:
            layer = idx_to_layer[ind]
            i2 = self.idx_to_i2[ind]
            if isinstance(layer, SubnetLinear):
                layer.weight.data[:,i2:i2+1].fill_(0.0)
            elif isinstance(layer, SubnetConv):
                layer.weight.data[i2:i2 + 1].fill_(0.0)


    def prune_final(self, index, model, cnt):
        layers, idx_to_layer, num_weights = self.get_layers(model)

        dict_p = {}
        dict_n_weights = {}
        for layer in layers:
            if isinstance(layer, SubnetLinear):
                n_weights = layer.weight.shape[1]
                dict_n_weights[layer] = n_weights
            elif isinstance(layer, SubnetConv):
                n_weights = layer.weight.shape[0]
                dict_n_weights[layer] = n_weights

        for layer in layers:
            # layer.popup_scores.fill_(1)
            dict_p[layer] = 0

        print(f'using shapley to prune the model')
        for ind in tqdm(index):
            if cnt==0:
                break
            layer = idx_to_layer[ind]
            if 1.0*dict_p[layer]/dict_n_weights[layer] < 1.0:
                i2 = self.idx_to_i2[ind]

                dict_p[layer] += 1
                cnt -= 1
                if isinstance(layer, SubnetLinear):
                    layer.popup_scores[:,i2:i2+1].fill_(0)
                elif isinstance(layer, SubnetConv):
                    layer.popup_scores[i2:i2 + 1].fill_(0)

        torch.save(model, self.ckp_name)

        for layer in layers:
            if isinstance(layer, SubnetLinear) or isinstance(layer, SubnetConv):
                print(f'{layer}: prune {dict_p[layer]} units out of {dict_n_weights[layer]}: ratio: {1.0*dict_p[layer]/dict_n_weights[layer]}')


    def run_all_forward_and_backward(self, model):
        """
        Run forward and backward passes on all data in `data_gen`
        :return: None
        """
        self.set_deterministic()
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)

            if self.args.trainer == 'crown-ibp':
                # crown-ibp
                loss, rerr = naive_interval_analyze(
                    model,
                    self.args.epsilon,
                    x,
                    y,
                    use_cuda=torch.cuda.is_available(),
                    parallel=False,# reduction='none'
                )
            elif self.args.trainer == 'smooth':
                # random smoothing
                output = model(x)
                loss_natural = nn.CrossEntropyLoss()(output, y)

                perturbation = torch.randn_like(x).to(self.device)
                loss_robust = (1.0 / len(x)) * nn.KLDivLoss(size_average=False)(
                    F.log_softmax(
                        model(
                            x + perturbation * self.args.noise_std
                        ),
                        dim=1,
                    ),
                    F.softmax(output, dim=1),
                )
                loss = loss_natural + self.args.beta * loss_robust

                # loss = loss_natural
            elif self.args.trainer == 'mixtrain':
            # mixtrain
                output = model(x)
                # y = torch.tensor(y, dtype=torch.bool)
                # y = y.bool()
                ce = nn.CrossEntropyLoss()(output, y)

                r = np.random.randint(low=0, high=x.shape[0], size=5)

                rce, rerr = sym_interval_analyze(model, self.args.epsilon,
                                                 x[r], y[r],
                                                 use_cuda=torch.cuda.is_available(),
                                                 parallel=False)
                # loss = 50 * rce + ce
                loss = 50 * rce + ce
            elif self.args.trainer == 'CE':
                # CE loss
                loss = self.criterion(model(x), y)
            elif self.args.trainer == 'adv':
                # calculate robust loss
                loss = trades_loss(
                    model=model,
                    x_natural=x,
                    y=y,
                    device=self.device,
                    optimizer=self.optimizer,
                    step_size=self.args.step_size,
                    epsilon=self.args.epsilon,
                    perturb_steps=self.args.num_steps,
                    beta=self.args.beta,
                    clip_min=self.args.clip_min,
                    clip_max=self.args.clip_max,
                    distance=self.args.distance,
                )

            loss.backward(retain_graph=True)

        self.restore_deterministic()

    def set_deterministic(self):
        self.deterministic = torch.backends.cudnn.deterministic
        self.benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def restore_deterministic(self):
        torch.backends.cudnn.deterministic = self.deterministic
        torch.backends.cudnn.benchmark = self.benchmark

    def get_weights_and_gradient_information(self):
        model_tmp = torch.load(self.ckp_name)
        self.run_all_forward_and_backward(model_tmp)
        for layer in model_tmp.modules():
            if isinstance(layer, SubnetLinear) or isinstance(layer, SubnetConv):
                weight = torch.abs(layer.weight.data).clone().cpu().numpy()
                gradient = torch.abs(layer.weight.grad.data).clone().cpu().numpy()
                print(f'{layer}: weight: {np.mean(weight)}/{np.std(weight)}, gradient: {np.mean(gradient)}/{np.std(gradient)}')