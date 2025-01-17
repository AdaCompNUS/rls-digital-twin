import torch
import torch.nn as nn
import scipy
import scipy.io.wavfile as wav
import numpy as np
import os

from audio_localization.com.nus.speech.server.model.lib import LinearBN1d, ZeroExpandInput, angular_distance_compute, testing


class sMLP(nn.Module):
    ''' Build the SNN network'''

    def __init__(self, Tsim):
        super(sMLP, self).__init__()
        self.T = Tsim
        self.fc1 = LinearBN1d(306, 1000)
        self.fc2 = LinearBN1d(1000, 1000)
        self.fc3 = LinearBN1d(1000, 1000)
        self.fc4 = nn.Linear(1000, 360)

    def forward(self, x):
        x = x.view(-1, 306)
        x_spike, x = ZeroExpandInput.apply(x, self.T)

        x_spike1, x1 = self.fc1(x_spike, x)
        x_spike2, x2 = self.fc2(x_spike1, x1)
        x_spike3, x3 = self.fc3(x_spike2, x2)
        out = self.fc4(x3)

        return x_spike1, x_spike2, x_spike3, out, torch.sigmoid(out)


if __name__ == '__main__':
    # Perform inference on the cpu
    device = 'cpu'

    # Load Data and Label
    data = scipy.io.loadmat("Record_Spike_180.mat")
    class_label = 180
    Xte = data['Xte2']
    num_sample = Xte.shape[0]
    Xmean = data['Xte2'].mean(axis=1)
    Xte = torch.tensor((Xte.T - Xmean.reshape(1, -1)).T)
    Yte = torch.ones((num_sample)) * class_label  # Binary target

    # Models and training configuration
    Tsim = 10
    model = sMLP(Tsim)
    model = model.to(device)


    # Load Pre-trainde ANN model
    checkpoint = torch.load("{0}.pt.tar".format("best"))
    model.load_state_dict(checkpoint['model_state_dict'])  # restore the model with best_loss

    x_spike1, x_spike2, x_spike3, out, y_pred = testing(model, Xte, device)

    # Result Analysis
    angle_pred = torch.argmax(y_pred, 1)

    mae = angular_distance_compute(Yte, angle_pred)

    step = 30
    bin_range = np.arange((0 - step / 2), 360 + (step / 2) + 1, step)

    hist, bin = np.histogram(angle_pred, bin_range)
    hist_copy = hist[0:12]
    hist_copy[0] = hist_copy[0] + hist[12]

    decision = bin_range[np.argmax(hist_copy)] + step / 2

    print(decision)
    print(hist_copy)
    print("Test MAE: {}".format(mae))
