import random
import time
import argparse
import multiprocessing
import logging
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundedModule, BoundedTensor, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.bound_ops import BoundExp
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter, logger, get_spec_matrix
import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from auto_LiRPA.eps_scheduler import *
import torch
import numpy as np

def get_exp_module(bounded_module):
    for _, node in bounded_module.named_modules():
        # Find the Exp neuron in computational graph
        if isinstance(node, BoundExp):
            return node
    return None


def Train(model, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust', loss_fusion=True,
          final_node_name=None):
    num_class = 200
    meter = MultiAverageMeter()
    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
    else:
        model.eval()
        eps_scheduler.eval()

    exp_module = get_exp_module(model)

    def get_bound_loss(x=None, c=None):
        if loss_fusion:
            bound_lower, bound_upper = False, True
        else:
            bound_lower, bound_upper = True, False

        if bound_type == 'IBP':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                           final_node_name=final_node_name, no_replicas=True)
        elif bound_type == 'CROWN':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=False, C=c, method='backward',
                           bound_lower=bound_lower, bound_upper=bound_upper)
        elif bound_type == 'CROWN-IBP':
            # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method='backward')  # pure IBP bound
            # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
            factor = (eps_scheduler.get_max_eps() - eps_scheduler.get_eps()) / eps_scheduler.get_max_eps()
            ilb, iub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                             final_node_name=final_node_name, no_replicas=True)
            if factor < 1e-50:
                lb, ub = ilb, iub
            else:
                clb, cub = model(method_opt="compute_bounds", IBP=False, C=c, method='backward',
                                 bound_lower=bound_lower, bound_upper=bound_upper, final_node_name=final_node_name,
                                 no_replicas=True)
                if loss_fusion:
                    ub = cub * factor + iub * (1 - factor)
                else:
                    lb = clb * factor + ilb * (1 - factor)

        if loss_fusion:
            if isinstance(model, BoundDataParallel):
                max_input = model(get_property=True, node_class=BoundExp, att_name='max_input')
            else:
                max_input = exp_module.max_input
            return None, torch.mean(torch.log(ub) + max_input)
        else:
            # Pad zero at the beginning for each example, and use fake label '0' for all examples
            lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
            return lb, robust_ce

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()
        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = method
        if eps < 1e-50:
            batch_method = "natural"
        if train:
            opt.zero_grad()
        # bound input for Linf norm used only
        if norm == np.inf:
            data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / loader.std).view(1, -1, 1, 1), data_max)
            data_lb = torch.max(data - (eps / loader.std).view(1, -1, 1, 1), data_min)
        else:
            data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data, labels = data.cuda(), labels.cuda()
            data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

        ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb)
        if loss_fusion:
            if batch_method == 'natural' or not train:
                output = model(x, labels)
                regular_ce = torch.mean(torch.log(output))
            else:
                model(x, labels)
                regular_ce = torch.tensor(0., device=data.device)
            meter.update('CE', regular_ce.item(), x.size(0))
            x = (x, labels)
            c = None
        else:
            c = get_spec_matrix(data, labels, num_class)
            x = (x, labels)
            output = model(x, final_node_name=final_node_name)
            regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
            meter.update('CE', regular_ce.item(), x[0].size(0))
            meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).item() / x[0].size(0), x[0].size(0))

        if batch_method == 'robust':
            # print(data.sum())
            lb, robust_ce = get_bound_loss(x=x, c=c)
            loss = robust_ce
        elif batch_method == 'natural':
            loss = regular_ce

        if train:
            loss.backward()

            if args.clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
                meter.update('grad_norm', grad_norm)

            if isinstance(eps_scheduler, AdaptiveScheduler):
                eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
        meter.update('Loss', loss.item(), data.size(0))

        if batch_method != 'natural':
            meter.update('Robust_CE', robust_ce.item(), data.size(0))
            if not loss_fusion:
                # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
                # If any margin is < 0 this example is counted as an error
                meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
        meter.update('Time', time.time() - start)

        if (i + 1) % 250 == 0 and train:
            logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))

    logger.info('[{:2d}:{:4d}]: eps={:.12f} {}'.format(t, i + 1, eps, meter))
    return meter

def train(
    model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer
):
    dummy_input = torch.randn(2, 3, 56, 56)
    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    model = BoundedModule(model, dummy_input, bound_opts={'relu': args.bound_opts}, device=args.device)
    model_loss = BoundedModule(CrossEntropyWrapper(model), (dummy_input, torch.zeros(1, dtype=torch.long)),
                               bound_opts={'relu': args.bound_opts, 'loss_fusion': True}, device=args.device)
    model_loss = BoundDataParallel(model_loss)

    ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
    opt = optim.Adam(model_loss.parameters(), lr=args.lr)
    norm = float(args.norm)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_decay_milestones, gamma=0.1)
    eps_scheduler = eval(args.scheduler_name)(args.eps, args.scheduler_opts)

    ## Step 5: start training
    if args.verify:
        eps_scheduler = FixedScheduler(args.eps)
        with torch.no_grad():
            Train(model, 1, test_data, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False, final_node_name=None)
    else:
        timer = 0.0
        best_err = 1e10
        # with torch.autograd.detect_anomaly():
        for t in range(epoch + 1, args.num_epochs + 1):
            logger.info("Epoch {}, learning rate {}".format(t, lr_scheduler.get_last_lr()))
            start_time = time.time()
            Train(model_loss, t, train_data, eps_scheduler, norm, True, opt, args.bound_type, loss_fusion=True)
            lr_scheduler.step()
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.info('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))

            logger.info("Evaluating...")
            torch.cuda.empty_cache()

            # remove 'model.' in state_dict
            state_dict_loss = model_loss.state_dict()
            state_dict = {}
            for name in state_dict_loss:
                assert (name.startswith('model.'))
                state_dict[name[6:]] = state_dict_loss[name]

            with torch.no_grad():
                if int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']) > t >= int(
                        eps_scheduler.params['start']):
                    m = Train(model_loss, t, test_data, eps_scheduler, norm, False, None, args.bound_type,
                              loss_fusion=True)
                else:
                    model_ori.load_state_dict(state_dict)
                    model = BoundedModule(model_ori, dummy_input, bound_opts={'relu': args.bound_opts},
                                          device=args.device)
                    model = BoundDataParallel(model)
                    m = Train(model, t, test_data, eps_scheduler, norm, False, None, 'IBP', loss_fusion=False)
                    del model

            save_dict = {'state_dict': state_dict, 'epoch': t, 'optimizer': opt.state_dict()}
            if t < int(eps_scheduler.params['start']):
                torch.save(save_dict, 'saved_models/natural_' + exp_name)
            elif t > int(eps_scheduler.params['start']) + int(eps_scheduler.params['length']):
                current_err = m.avg('Verified_Err')
                if current_err < best_err:
                    best_err = current_err
                    torch.save(save_dict, 'saved_models/' + exp_name + '_best_' + str(best_err)[:6])
            else:
                torch.save(save_dict, 'saved_models/' + exp_name)
            torch.cuda.empty_cache()
