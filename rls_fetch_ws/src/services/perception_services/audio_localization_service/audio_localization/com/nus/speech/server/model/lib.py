import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroExpandInput(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_image, T, device=torch.device('cpu')):
        """
        Args:
            input_image: normalized within (0,1)
        """
        N, dim = input_image.shape
        input_image_sc = input_image
        zero_inputs = torch.zeros(N, T - 1, dim).to(device)
        input_image = input_image.unsqueeze(dim=1)
        input_image_spike = torch.cat((input_image, zero_inputs), dim=1)

        return input_image_spike, input_image_sc

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""

        return None, None, None


class LinearIF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, spike_in, ann_output, weight, device=torch.device('cpu'), bias=None):
        """
        args:
            spike_in: (N, T, in_features)
            weight: (out_features, in_features)
            bias: (out_features)
        """
        N, T, _ = spike_in.shape
        out_features = bias.shape[0]
        pot_in = spike_in.matmul(weight.t())
        spike_out = torch.zeros_like(pot_in, device=device)
        bias_distribute = bias.repeat(N, 1) / T  # distribute bias through simulation time steps
        pot_aggregate = torch.zeros(N, out_features, device=device)

        # Iterate over simulation time steps to determine output spike trains
        for t in range(T):
            pot_aggregate = pot_aggregate + pot_in[:, t, :].squeeze() + bias_distribute
            bool_spike = torch.ge(pot_aggregate, 1.0).float()
            spike_out[:, t, :] = bool_spike
            pot_aggregate -= bool_spike

        spike_count_out = torch.sum(spike_out, dim=1).squeeze()

        return spike_out, spike_count_out

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""
        grad_ann_out = grad_spike_count_out.clone()

        return None, grad_ann_out, None, None, None, None


class LinearBN1d(nn.Module):

    def __init__(self, D_in, D_out, device=torch.device('cpu'), bias=True):
        super(LinearBN1d, self).__init__()
        self.linearif = LinearIF.apply
        self.linear = torch.nn.Linear(D_in, D_out, bias=bias)
        self.device = device
        self.bn1d = torch.nn.BatchNorm1d(D_out, eps=1e-4, momentum=0.9)
        nn.init.normal_(self.bn1d.weight, 0, 2.0)

    def forward(self, input_feature_st, input_features_sc):
        # weight update based on the surrogate linear layer
        T = input_feature_st.shape[1]
        output_bn = self.bn1d(self.linear(input_features_sc))
        output = F.relu(output_bn)

        # extract the weight and bias from the surrogate linear layer
        linearif_weight = self.linear.weight.detach().to(self.device)
        linearif_bias = self.linear.bias.detach().to(self.device)

        bnGamma = self.bn1d.weight
        bnBeta = self.bn1d.bias
        bnMean = self.bn1d.running_mean
        bnVar = self.bn1d.running_var

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Linear' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0)
        biasNorm = torch.mul(linearif_bias - bnMean, ratio) + bnBeta

        # propagate the input spike train through the linearIF layer to get actual output
        # spike train
        output_st, output_sc = self.linearif(input_feature_st, output, weightNorm, \
                                             self.device, biasNorm)

        return output_st, output_sc


def angular_distance_compute(label, pred):
    mae = []
    for i in range(len(label)):
        result = 180 - abs(abs(label[i] - pred[i]) - 180)
        mae.append(result)

    return sum(mae) / len(mae)


def testing(model, testSample, device):
    model.eval()  # Put the model in test mode

    inputs = testSample.type(torch.FloatTensor).to(device)

    # Model computation
    y_pred = model.forward(inputs)

    return y_pred

# def testing(model, testSample, device):
#     print testSample.shape
#     model.eval()  # Put the model in test mode
#
#     y_pred = []
#
#     for one_frame in testSample:
#         inputs = one_frame.type(torch.FloatTensor).to(device)
#
#         # Model computation
#         x_spike1, x_spike2, x_spike3, out, y_pred = model.forward(inputs)
#
#     return y_pred
