"""
CNN model for 32x32x3 image classification
"""
from typing import Optional

from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torch.autograd as autograd
from torch.nn.parameter import Parameter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        return scores

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = torch.ones((self.weight.shape[0],self.weight.shape[1],self.weight.shape[2],self.weight.shape[3]),requires_grad=False).to(device)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0
        self.k = -1

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        #self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        # nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        #self.popup_scores[:] = 1.0

        self.popup_scores = torch.ones(
            (self.weight.shape[0], self.weight.shape[1]),
            requires_grad=False).to(device)

        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0
        self.k = -1
        # self.register_buffer('w', None)

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):
    """
    This is a simple CNN for CIFAR-10 and does not achieve SotA performance
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = conv_layer(3, 16, 4, stride=2, padding=1)
        self.relu1 = nn.relu()
        self.conv2 = conv_layer(16, 32, 4, stride=2, padding=1)
        self.relu2 = nn.relu()
        self.flat = Flatten()
        self.fc1 = linear_layer(32 * 8 * 8, 100),
        self.relu3 = nn.ReLU()
        self.fc2 = linear_layer(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def make_cifar_model(**kwargs) -> Net:
    return Net()


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = make_cifar_model(**model_kwargs)
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(32, 32, 3),
        channels_first=False,
        nb_classes=10,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
