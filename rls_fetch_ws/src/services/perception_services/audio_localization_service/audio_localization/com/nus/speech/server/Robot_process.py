# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:43:16 2019

@author: Pan Zihan  input Data_matrix
"""

# !/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import torch
import torch.nn.functional as F


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        #        self.input_bn = torch.nn.BatchNorm1d(306)
        self.linear1 = torch.nn.Linear(306, 1000)
        self.dp1 = torch.nn.Dropout(p=0.2)
        self.dense1_bn = torch.nn.BatchNorm1d(1000)
        self.linear2 = torch.nn.Linear(1000, 1000)
        self.dp2 = torch.nn.Dropout(p=0.2)
        self.dense2_bn = torch.nn.BatchNorm1d(1000)
        self.linear3 = torch.nn.Linear(1000, 1000)
        self.dp3 = torch.nn.Dropout(p=0.2)
        self.dense3_bn = torch.nn.BatchNorm1d(1000)
        self.linear4 = torch.nn.Linear(1000, 360)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        #        x = self.input_bn(x)
        x1 = F.relu(self.dense1_bn(self.dp1(self.linear1(x))))
        x2 = F.relu(self.dense2_bn(self.dp2(self.linear2(x1))))
        x3 = F.relu(self.dense3_bn(self.dp3(self.linear3(x2))))
        y_pred = F.sigmoid(self.linear4(x3))
        return y_pred


class SnnInf:
    ## Dummy values for testing
    def __init__(self):

        torch.manual_seed(7)
        # torch.cuda.manual_seed_all(7)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        use_gpu = False

        self.WIN_SIZE = 8192  # signal window and step
        self.HOP_SIZE = 4096

        self.fft_win = 8192  # fft window and step
        self.fft_hop = 8192
        self.nfft = 1024
        self.fs = 16000.0
        # our model
        self.model = Model()
        self.model.load_state_dict(torch.load('here16.model'))
        # In[49]:
        self.model.eval()
        # In[50]:
        self.model = self.model.double()

    def stft(self, xsig, fs, wlen, hop, nfft):  # input 170ms
        """
        Compute STFT  % several frames per segament: we sum together
        """
        xlen = len(xsig)

        coln = 1 + np.abs(xlen - wlen) // hop
        rown = 1 + nfft // 2
        stft_data = np.zeros((coln, rown), dtype=np.complex)
        win = np.hamming(wlen)

        for v in range(coln):
            x = xsig[v * hop:v * hop + wlen] * win
            x1 = np.fft.fft(x, nfft)
            #x1 = scipy.fftpack.fft(x, nfft)
            stft_data[v, :] = x1[:rown]

        f = np.arange(rown) * fs / nfft;

        return stft_data, f

    def Spike_encoder(self, x, fs, wlen, hop, nfft):
        """
        encode FFT results into spikes for 4 channel microphone array
        """
        nc = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        spike_train = np.zeros([6, 51])
        for n in range(nc.shape[0]):
            X1, f = self.stft(x[:, nc[n][0]], fs, wlen, hop, nfft)
            X2, f = self.stft(x[:, nc[n][1]], fs, wlen, hop, nfft)

            t1 = (np.pi + np.angle(X1[:, 1:])) / np.array([2 * np.pi * f[1:]])
            t2 = (np.pi + np.angle(X2[:, 1:])) / np.array([2 * np.pi * f[1:]])

            delta_t = t1 - t2

            t_resolution = 1 / fs

            tau_index = delta_t // t_resolution + 27

            # idx1=np.where(tau_index>51)
            # idx2=np.where(tau_index<1)

            # np.delete(tau_index,idx1)
            # np.delete(tau_index,idx2)

            binaural_spike = np.zeros([1, 51])

            for tau in range(51):
                spike_idx = np.where(tau_index == (tau + 1))
                binaural_spike[0, tau] = len(spike_idx[1])

            spike_train[n, :] = binaural_spike

        output = np.reshape(spike_train, (1, 306), order='F')

        return output

    def Encode_segament(self, segament_matrix ,fs): # 2730*4

        x_frame = segament_matrix
        output_spike = self.Spike_encoder(x_frame, fs, self.fft_win // 3, self.fft_hop // 3, self.nfft)

        return output_spike

    def Encode_robot_read(self, Data_matrix):

        testData=Data_matrix

        Xtr = {}
        Xtr['Xte2'] = np.zeros([1, 306])

        num_files = testData.shape[0]

        for t in range(num_files):

            temp_matrix = np.squeeze(testData[t, :, :])

            Xtr['Xte2'] = np.append(Xtr['Xte2'], self.Encode_segament(temp_matrix, self.fs), axis=0)

        temp = Xtr['Xte2']
        temp = temp[1:, :]
        Xtr['Xte2'] = temp

        return Xtr

    def inference_direction(self, Data_matrix):
        Xtr = self.Encode_robot_read(Data_matrix)

        Xte2 = Xtr['Xte2']
        mn = Xte2.mean(axis=1)
        Xte2 = (Xte2.T - mn.reshape(1, -1)).T
        # In[48]:

        Y = self.model(torch.from_numpy(Xte2))

        # In[60]:
        estimates = torch.argmax(Y, 1).numpy()

        step = 30
        bin_range = np.arange(step/2, 360-step/2, step)
        hist, bin = np.histogram(estimates, bin_range)

        decision = bin_range[np.argmax(hist)]+step/2
        list1 = decision.tolist()
        print(list1)
        return list1
