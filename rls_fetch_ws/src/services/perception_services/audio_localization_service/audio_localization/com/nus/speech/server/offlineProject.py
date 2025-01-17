import scipy
import scipy.io.wavfile as wav
import torch
import os
import numpy as np

from com.nus.speech.server.model.sl_inference_small_batch import sMLP
from com.nus.speech.server.model.lib import angular_distance_compute, testing

import com.nus.speech.server.config as cfg


class Project(object):

    def __init__(self):
        print("init project")
        self.step = 30
        self.Tsim = 10
        self.model = sMLP(self.Tsim)
        self.device = 'cpu'
        self.class_label = 180
        print("load model")
        # Models and training configuration
        model = sMLP(self.Tsim)
        self.model = model.to(self.device)

        # Load Pre-trainde ANN model
        abs_path = os.path.abspath(__file__).split('/')
        self.path = ""
        for i in range(len(abs_path) - 1):
            self.path = self.path + abs_path[i]
            self.path = self.path + "/"
        print(self.path)
        print("finish load model -1 ")
        self.checkpoint = torch.load(self.path + "{0}.pt.tar".format("best"), map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # self.checkpoint = torch.load("{0}.pt.tar".format("best"))
        print(self.checkpoint)
        print("finish load model00 ")
        self.model.load_state_dict(self.checkpoint['model_state_dict'])  # restore the model with best_loss
        print("finish load model")
        for i in self.model.parameters():
            print i

    def get_calculate_results(self, path0):
        print("get calculate result")
        change_flag = True
        # Load Data and Label
        data = scipy.io.loadmat(self.path + path0)
        Xte = data['Xte2']
        num_sample = Xte.shape[0]
        Xmean = data['Xte2'].mean(axis=1)
        print(type(Xte))
        print(Xte.shape)
        Xte = torch.tensor((Xte.T - Xmean.reshape(1, -1)).T)
        Yte = torch.ones((num_sample)) * self.class_label  # Binary target

        x_spike1, x_spike2, x_spike3, out, y_pred = testing(self.model, Xte, self.device)
        print("finish testing and the decision is ", y_pred)
        # Result Analysis
        angle_pred = torch.argmax(y_pred, 1)
        print("finish testing and the decision is ", angle_pred)

        mae = angular_distance_compute(Yte, angle_pred)

        bin_range = np.arange((0 - self.step / 2), 360 + (self.step / 2) + 1, self.step)

        hist, bin = np.histogram(angle_pred, bin_range)
        hist_copy = hist[0:12]
        hist_copy[0] = hist_copy[0] + hist[12]

        decision = bin_range[np.argmax(hist_copy)] + self.step / 2
        print("finish testing and the decision is ", decision)
        return {cfg.grid_conv_1: x_spike1[-1].tolist(),
                cfg.grid_conv_2: x_spike2[-1].tolist(),
                cfg.grid_conv_3: x_spike3[-1].tolist(),
                cfg.grid_conv_4: y_pred[-1].tolist(),
                cfg.location: decision,
                cfg.locationBins: hist_copy.tolist(),
                cfg.change_flag: change_flag}

if __name__ == '__main__':
    system = Project()
    system.get_calculate_results("model/Record_Spike_150.mat")
    system.get_calculate_results("model/Record_Spike_90.mat")