#!/usr/bin/env python

from model_tc import *
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import os
import config as args

# Constants
MODE = 'wxy'
METHOD = 'simcom'
W = 128.0  # !!!! Important to make it float to prevent integer division becomes zeros
H = 106.0


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


class PushNetModel:
    def __init__(self):
        print("PushNet model init")
        self.bs = args.batch_size
        model_path = os.path.dirname(__file__) + '/model'
        best_model_name = 'model_best_' + args.arch[METHOD] + '.pth.tar'
        self.model_path = os.path.join(model_path, best_model_name)
        self.model = self.build_model()
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def build_model(self):
        if METHOD == 'simcom':
            return COM_net_sim(self.bs)
        elif METHOD == 'sim':
            return COM_net_sim_only(self.bs)
        elif METHOD == 'nomem':
            return COM_net_nomem(self.bs)

    def reset_model(self):
        self.model.hidden = self.model.init_hidden()

    def update(self, start, end, img_curr, img_goal):
        bs = self.bs
        A1 = []
        I1 = []
        Ig = []
        for i in range(bs):
            a1 = [[start[0]/W, start[1]/H, end[0]/W, end[1]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        if METHOD == 'simcom':
            sim_out, com_out = self.model(
                A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)

    def select_action(self, img_curr, img_goal, actions):
        bs = self.bs
        A1 = []
        I1 = []
        Ig = []
        print("Predictor/select_action")
        #print bs, H, W

        for i in range(bs):
            a1 = [[actions[4*i]/W, actions[4*i+1]/H,
                   actions[4*i+2]/W, actions[4*i+3]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        #com_np = None
        sim_out = None
        com_out = None
        #st_time = time.time()
        if METHOD == 'simcom':
            sim_out, com_out = self.model(
                A1, I1, A1, Ig, [1 for j in range(bs)], bs)
            #com_np = com_out.data.cpu().data.numpy()
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        #print time.time() - st_time

        sim_np = sim_out.data.cpu().data.numpy()

        if MODE == 'wxy':
            sim_sum = np.sum(sim_np, 1)  # measure (w ,x, y)
        elif MODE == 'xy':
            sim_sum = np.sum(sim_np[:, 1:], 1)  # measure (x, y)
        else:
            sim_sum = np.sum(sim_np, 1)  # measure (w ,x, y)
            # sim_sum = sim_np[:, 0] # measure (w)
        action_value = []
        for ii in range(len(sim_sum)):
            s = [actions[4 * ii], actions[4 * ii + 1]]
            e = [actions[4 * ii + 2], actions[4 * ii + 3]]
            action_value.append([[s, e], sim_sum[ii].item()])

        return action_value
        #min_idx = np.argmin(sim_sum)
        #best_sim_score = np.min(sim_sum)

        #start = [actions[4*min_idx], actions[4*min_idx+1]]
        #end = [actions[4*min_idx+2], actions[4*min_idx+3]]

        #sim_measure = list(sim_np[min_idx,:])
        #com_pred = [com_np[min_idx, 0] * W, com_np[min_idx, 1] * H]
        # return start, end, sim_measure, com_pred, best_sim_score
