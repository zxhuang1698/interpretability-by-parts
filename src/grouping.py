# import libs
import math
import numpy as np
import torch
import torch.nn as nn

class GroupingUnit(nn.Module):

    def __init__(self, in_channels, num_parts):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels

        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1)) 
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering 
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)


    def forward(self, inputs):
        assert inputs.dim()==4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels==self.in_channels

        # 1. generate the grouping centers
        grouping_centers = self.weight.contiguous().view(1, self.num_parts, self.in_channels).expand(batch_size, self.num_parts, self.in_channels)

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h*input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1) # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)
        
        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)
        
        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C 
        outputs = nn.functional.normalize(out, dim=2)
        outputs_t = outputs.permute(0, 2, 1)
        
        # generate assignment map for basis for visualization
        return outputs_t, assign
        
    # name
    def __repr__(self):                                                         
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'

