import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

num_class = 40
smooth = 0.05


class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."

    def __init__(self, class_num, smoothing):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.class_num = class_num

    def forward(self, x, target):
        assert x.size(1) == self.class_num
        if self.smoothing is None:
            return nn.CrossEntropyLoss()(x, target)

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.class_num - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        logprobs = F.log_softmax(x, dim=-1)
        mean_loss = -torch.sum(true_dist * logprobs) / x.size(-2)
        return mean_loss


def initialize_weight(model, method):
    method = dict(normal=['normal_', dict(mean=0, std=0.01)],
                  xavier_uni=['xavier_uniform_', dict()],
                  xavier_normal=['xavier_normal_', dict()],
                  he_uni=['kaiming_uniform_', dict()],
                  he_normal=['kaiming_normal_', dict()]).get(method)
    if method is None:
        return None

    for module in model.modules():
        # LSTM
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                # Not BN
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                # BN
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)