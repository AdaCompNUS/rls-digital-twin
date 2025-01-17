import numpy as np
import torch
import torch.nn as nn

from .lib import LinearBN1d, ZeroExpandInput


class EncodeSpiking:

    def __init__(self):
        self.WIN_SIZE = 8192  # signal window and step
        self.HOP_SIZE = 4096

        self.fft_win = 8192  # fft window and step
        self.fft_hop = 8192
        self.nfft = 1024
        self.fs = 16000.0

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
            # x1 = scipy.fftpack.fft(x, nfft)
            stft_data[v, :] = x1[:rown]

        f = np.arange(rown) * fs / nfft;

        return stft_data, f

    def spike_encoder(self, x, fs, wlen, hop, nfft):
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


            binaural_spike = np.zeros([1, 51])

            for tau in range(51):
                spike_idx = np.where(tau_index == (tau + 1))
                binaural_spike[0, tau] = len(spike_idx[1])
                spike_train[n, :] = binaural_spike

        output = np.reshape(spike_train, (1, 306), order='F')


        return output

    def encode_segament(self, segament_matrix):  # 2730*4

        x_frame = segament_matrix
        output_spike = self.spike_encoder(x_frame, self.fs, self.fft_win // 3, self.fft_hop // 3, self.nfft)

        return output_spike

    def encode_robot_read(self, data_matrix):
        testData = data_matrix

        Xtr = {}
        Xtr['Xte2'] = np.zeros([1, 306])

        num_files = testData.shape[0]

        for t in range(num_files):
            temp_matrix = np.squeeze(testData[t, :, :])


            Xtr['Xte2'] = np.append(Xtr['Xte2'], self.encode_segament(temp_matrix), axis=0)

        temp = Xtr['Xte2']
        temp = temp[1:, :]
        Xtr['Xte2'] = temp


        return Xtr


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

# if __name__ == '__main__':
#     # Perform inference on the cpu
#     device = 'cpu'
#
#     # Load Data and Label
#     # data = hdf5storage.loadmat("Record_Spike_180.mat") # load data
#     # Xte2 = data['Xte2']
#     # Xmean = Xte2.mean(axis=1)
#     # Xte2 = torch.tensor((Xte2.T - Xmean.reshape(1, -1)).T)
#     Xte2 = Encode_robot_read(Data_matrix)
#
#     # Models and training configuration
#     Tsim = 10
#     model = sMLP(Tsim)
#     model = model.to(device)
#
#     # Load Pre-trainde ANN model
#     checkpoint = torch.load("{0}.pt.tar".format("best"))
#     model.load_state_dict(checkpoint['model_state_dict'])  # restore the model with best_loss
#
#     x_spike1, x_spike2, x_spike3, out, y_pred = testing(model, Xte2, device)
#
#     # Result Analysis
#     angle_pred = torch.argmax(y_pred, 1)
#
#     step = 30
#     bin_range = np.arange(step / 2, 360 - step / 2, step)
#     hist, bin = np.histogram(angle_pred, bin_range)
#
#     decision = bin_range[np.argmax(hist)] + step / 2
#
#     print(decision)
