import imp
import os
import math
import numpy as np
import random
import multiprocessing as mp
import torch
import torchvision
from torch.utils.data import Dataset
from torch import nn
from torch import optim
from functools import partial
from torch.optim import lr_scheduler
import torch.nn.functional as F
from NN import *
# from make_MPS_data_regression_1 import dynamical_phy_intuition, dynamical_gauge
from make_MPS_data_regression_1 import dynamical_phy_intuition, dynamical_gauge
from correctionVMC import VMC
import warnings
warnings.filterwarnings("ignore")

DefaultPath = '/home/sijing/ML/MPS/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(100)
# np.random.seed(100)
# random.seed(100)

# def Network(Net, input_size, output_size=1, embedding_size=500, hidden_size=100, hidden_size_1=100, hidden_numbers=[1, 1, 1, 1],
#             block_numbers=[1, 1, 1, 1], drop=0.5):
#     return {
#         'Net': Model(input_size, output_size=output_size, embedding_size=embedding_size, hidden_size=hidden_size,
#                      hidden_numbers=hidden_numbers, block_numbers=block_numbers, drop=drop),
#         'Net_sig': Model_sig(input_size, output_size=output_size, embedding_size=embedding_size,
#                              hidden_size=hidden_size, hidden_numbers=hidden_numbers, block_numbers=block_numbers,
#                              drop=drop),
#         'SimpleNet': SimpleModel(input_size, embedding_size=embedding_size, hidden_size=hidden_size,
#                                  output_size=output_size),
#         'SimpleNet_sig': SimpleModel_sig(input_size, embedding_size=embedding_size, hidden_size=hidden_size,
#                                          output_size=output_size),
#         'NaiveNet': NaiveModel(input_size, embedding_size=embedding_size, hidden_size=hidden_size,
#                                output_size=output_size),
#         'NaiveNet_sig': NaiveModel_sig(input_size, embedding_size=embedding_size, hidden_size=hidden_size,
#                                        output_size=output_size),
#         'OneLayerNet': one_hidden_layer_net(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
#     }.get(Net)


def Optimizer(opt, net, lr=1e-3, weight_decay=0, momentum=0, betas=(0.9, 0.999), max_iter=20):
    return {
        'Adadelta': optim.Adadelta(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                   rho=0.9, eps=1e-06, weight_decay=weight_decay),
        #'Adagrad': optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
        #                         lr_decay=0, weight_decay=weight_decay, initial_accumulator_value=0, eps=1e-10),
        'Adagrad': optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                 lr_decay=0, weight_decay=weight_decay, initial_accumulator_value=0), # in the original code there is an extra parameter 'eps'
        'Adam': optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                               betas=betas, eps=1e-08, weight_decay=weight_decay, amsgrad=False),
        'AdamW': optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                             betas=betas, eps=1e-08, weight_decay=weight_decay, amsgrad=False),
        'Adamax': optim.Adamax(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                               betas=betas, eps=1e-08, weight_decay=weight_decay),
        'SGD': optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                              momentum=momentum, dampening=0, weight_decay=weight_decay, nesterov=False),
        'ASGD': optim.ASGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                           lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=weight_decay),
        'RMSprop': optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                  alpha=0., eps=1e-08, weight_decay=weight_decay, momentum=momentum, centered=False),
        'Rprop': optim.Rprop(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50)),
        'LBFGS': optim.LBFGS(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, max_iter=max_iter,
                             max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100,
                             line_search_fn=None)
    }.get(opt)


def Scheduler(sch, optimizer, lr_lambda=lambda epoch:1, step_size=1, gamma=1, milestones=[0], T_max=1, T_0=1, T_mult=1):
    return {
        'LambdaLR': lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1),
        'MultiplicativeLR': lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1),
        'StepLR': lr_scheduler.StepLR(optimizer, step_size, gamma=gamma, last_epoch=-1),
        'MultiStepLR': lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1),
        'ExponentialLR': lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1),
        'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1),
        'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=gamma, patience=10, verbose=False, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08),
        'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult=T_mult, eta_min=0, last_epoch=-1)
    }.get(sch)


class LoadMPSData_Reg(Dataset): #data-loading function
    def __init__(self, dataload, scale=1e3):
        self.dataload = dataload
        self.scale = scale

    def __getitem__(self, index):
        s = self.dataload[index]
        # label = torch.tensor([np.tanh(self.scale * s[-1])], dtype=torch.float32)
        label = torch.tensor([np.tanh(self.scale*s[-1])], dtype=torch.float32)
        s = torch.from_numpy(s[:-1]).float()
        return s, label

    def __len__(self):
        return len(self.dataload)


# class LoadMPSData_Reg(Dataset):
#     def __init__(self, dataload, scale = 1e3):
#         self.data = np.load(dataload, allow_pickle=True)['arr_0']
#         self.scale = scale

#     def __getitem__(self, k):
#         s = self.data[k]
#         label = torch.tensor([np.tanh(self.scale * s[-1])], dtype=torch.float32)
#         s = torch.from_numpy(s[:-1]).float()
#         return s, label

#     def __len__(self):
#         return len(self.data)

# class LoadDynamicalData(Dataset):
#     def __init__(self, scale, func, n = 8, amount = 20, process_number_time = 4, filename=None):
#         self.scale = scale
#         self.n = n
#         self.amount = amount
#         self.func = func
#         self.process_number_time = process_number_time
#         self.proportion = 2/(n-1)
#         self.data = self.genBlock()
#         if filename is None:
#             self.data = self.genBlock()
#         else:
#             self.data = self.loadBlock(filename)

#     def genBlock(self):
#         # sim = Sim(self.agent, ed)
#         # tolData = sim(0)
#         # return tolData
#         pnt = self.process_number_time
#         pool = mp.Pool(processes=32)
#         pool_list = pnt * list((i, self.amount) for i in range(self.n))
#         if self.func == 0:
#             tolData = pool.map(dynamical, pool_list) #每次dynamical生成amount个物理数据，根据其参数生成x个非物理数据，n对应一个物理数据最多生成的非物理数据量
#         elif self.func == 1:
#             tolData = pool.map(dynamical_phy_intuition, pool_list)  # 每次dynamical生成amount个物理数据，根据其参数生成x个非物理数据
#         #共有 (amount * pn * n) + (n*(n-1)/2 * amount * pn) 个数据，即 [n(n+1)/2 * pn * amount] 条数据; 数据比例: 物理/非物理 = 2/(n-1)
#         pool.close()
#         pool.join()
#         return [x for y in tolData for x in y]

#     def saveBlock(self, filename):
#         np.savez(filename, self.data)

#     def loadBlock(self, filename):
#         return np.load(filename, allow_pickle=True)['arr_0']

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, k):
#         s = self.data[k]
#         label = torch.tensor([np.tanh(self.scale * s[-1])], dtype=torch.float32)
#         s = torch.from_numpy(s[:-1]).float()
#         return s, label

class LoadDynamicalData_gauge(Dataset):
    def __init__(self, func, chi, L, decay, amount_dv, scale = 200, rate = 100, amount = 100, process_number_time = 1, filename=None):
        self.scale = scale
        self.rate = rate
        self.amount = amount
        self.func = func
        self.chi = chi
        self.L = L
        self.decay = decay
        self.amount_dv = amount_dv
        self.process_number_time = process_number_time
        self.proportion = 1/rate
        # self.data = self.genBlock()
        if filename is None:
            self.data = self.genBlock()
        else:
            self.data = self.loadBlock(filename)

    def genBlock(self):
        pnt = self.process_number_time
        pool = mp.Pool(processes=32)
        pool_list = pnt * list((self.rate, self.amount, self.chi, self.L, self.amount_dv, self.decay) for i in range(1))

        if self.func == 0:
            tolData = pool.map(dynamical_phy_intuition, pool_list) #每次dynamical生成amount个物理数据，每个物理数据对应生成rate个非物理数据
        elif self.func == 1:
            tolData = pool.map(dynamical_gauge, pool_list)
        
        pool.close()
        pool.join()
        return [x for y in tolData for x in y]

    def saveBlock(self, filename):
        np.savez(filename, self.data)

    def loadBlock(self, filename):
        return np.load(filename, allow_pickle=True)['arr_0']

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, k):
    #     s = self.data[k]
    #     label = torch.tensor([np.tanh(self.scale * s[-1])], dtype=torch.float32)
    #     s = torch.from_numpy(s[:-1]).float()
    #     return s, label


class LoadVMCData(Dataset):
    def __init__(self, L, amount_dv, decay, func=0, scale = 200, rate = 100, amount = 100, process_number_time = 1, filename=None):
        self.scale = scale
        self.L = L
        self.rate = rate
        self.amount = amount
        self.func = func
        self.amount_dv = amount_dv
        self.decay = decay
        self.process_number_time = process_number_time
        self.proportion = 1/rate
        # self.data = self.genBlock()
        if filename is None:
            self.data = self.genBlock()
        else:
            self.data = self.loadBlock(filename)

    def genBlock(self):
        pnt = self.process_number_time
        pool = mp.Pool(processes=32)
        pool_list = pnt * list((self.rate, self.amount, self.L, self.amount_dv, self.decay) for i in range(1))

        tolData = pool.map(VMC, pool_list)
        
        pool.close()
        pool.join()
        return [x for y in tolData for x in y]

    def saveBlock(self, filename):
        np.savez(filename, self.data)

    def loadBlock(self, filename):
        return np.load(filename, allow_pickle=True)['arr_0']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, k):
        s = self.data[k]
        label = torch.tensor([np.tanh(self.scale * s[-1])], dtype=torch.float32)
        s = torch.from_numpy(s[:-1]).float()
        return s, label
# class LoadDynamicalData_long(Dataset):
#     def __init__(self, scale, func, n = 8, amount = 20, process_number_time = 4, filename=None):
#         self.scale = scale
#         self.n = n
#         self.amount = amount
#         self.func = func
#         self.process_number_time = process_number_time
#         self.proportion = 2/(n-1)
#         self.data = self.genBlock()
#         if filename is None:
#             self.data = self.genBlock()
#         else:
#             self.data = self.loadBlock(filename)

#     def genBlock(self):
#         # sim = Sim(self.agent, ed)
#         # tolData = sim(0)
#         # return tolData
#         pnt = self.process_number_time
#         pool = mp.Pool(processes=32)
#         pool_list = pnt * list((i, self.amount) for i in range(self.n))
#         if self.func == 0:
#             tolData = pool.map(dynamical, pool_list) #每次dynamical生成amount个物理数据，根据其参数生成x个非物理数据
#         elif self.func == 1:
#             tolData = pool.map(dynamical_phy_intuition, pool_list)  # 每次dynamical生成amount个物理数据，根据其参数生成x个非物理数据
#         #共有 (amount * pn * n) + (n*(n-1)/2 * amount * pn) 个数据，即 [n(n+1)/2 * pn * amount] 条数据; 数据比例: 物理/非物理 = 2/(n-1)
#         pool.close()
#         pool.join()
#         return [x for y in tolData for x in y]

#     def saveBlock(self, filename):
#         np.savez(filename, self.data)

#     def loadBlock(self, filename):
#         return np.load(filename, allow_pickle=True)['arr_0']

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, k):
#         s = self.data[k]
#         label = torch.tensor([np.tanh(self.scale * s[-1])], dtype=torch.long)
#         s = torch.from_numpy(s[:-1]).float()
#         return s, label

# def DynamicalMPSx():
#     # sim = Sim(self.agent, ed)
#     # tolData = sim(0)
#     # return tolData
#     pool = mp.Pool(processes=32)
#     tolData = pool.map(dynamical, 4 * list(np.arange(8))) #4*8=32次调用dynamical，每次dynamical生成amount个真实数据，根据其参数生成x个非物理数据
#     #此时我们设amount=20，则共有2*320个真实数据，有2*280*4=1120个非物理数据,共2880个数据
#     pool.close()
#     pool.join()
#     return [x for y in tolData for x in y]

# def DynamicalMPS(range1, range2):
#     tolData = []
#     for _ in range(range1):#~4
#         for i in range(range2):#~8
#             s = dynamical(i, amount=20)#amount~20
#             for x in s:
#                 tolData.append(x)
#     return tolData

class LagInsulator(nn.Module):
    def __init__(self, model, init, N, penalty):
        super(LagInsulator, self).__init__()
        self.m = model
        for p in self.parameters():
            p.requires_grad = False
        self.m.eval()
        self.N, self.H, self.P, self.penalty = N, None, None, penalty
        self.g = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))

    def Clip_Norm(self):
        norm = torch.norm(self.g)
        if norm < 0.7:
            self.g = nn.Parameter(self.g * 0.7 / norm)
        elif norm > 1.1:
            self.g = nn.Parameter(self.g * 1.1 / norm)

    def forward(self, t): # t = [V, Re(t), Im(t)]
        self.Clip_Norm()
        self.P = F.relu(self.m(self.g)).squeeze()
        self.H = -(t[1] * (self.g[0, self.N * 3 + 1] + self.g[0, self.N * 3]) - t[2] * (
                    self.g[0, self.N * 5 + 1] - self.g[0, self.N * 5])) + t[0] * (
                        0.5 - 0.5 * self.g[0, 0].pow(2) - 0.25 * self.g[0, self.N * 3 + 1].pow(2) - 0.25 * self.g[
                    0, self.N * 5 + 1].pow(2) - 0.25 * self.g[0, self.N * 3].pow(2) - 0.25 * self.g[0, self.N * 5].pow(
                    2))
        return self.H + self.penalty[0] * self.P.pow(self.penalty[1])

class LagMPS(nn.Module):
    def __init__(self, model, init, N, penalty): #init is the initial data we feed to the ANN
        super(LagMPS, self).__init__()
        self.m = model
        for p in self.parameters():
            p.requires_grad = False
        self.m.eval()
        self.N, self.H, self.P, self.penalty = N, None, None, penalty #penalty here is just a pair of coefficients
        self.s = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))# self.s is the initial correlations we feed to the ANN

    # def Clip_Norm(self):
    #     norm = torch.norm(self.s)
    #     if norm < 0.7:
    #         self.s = nn.Parameter(self.s * 0.7 / norm)
    #     elif norm > 1.1:
    #         self.s = nn.Parameter(self.s * 1.1 / norm)

    def forward(self, C): # C = [J, g] for simplicity we can choose J=1 and only change g
        # self.Clip_Norm()
        self.P = self.m(self.s).squeeze() # why relu?
        self.H = -C[0] * self.N * (self.s[0, 5]) - self.N * C[1] * (self.s[0, 0] )
        return self.H + self.penalty[0] * self.P.pow(self.penalty[1]) #real peanlty is the output of the ANN


class LagMPS_multi(nn.Module):
    def __init__(self, model, init, N, penalty, bc): #init is the initial data we feed to the ANN
        super(LagMPS_multi, self).__init__()
        self.m = model
        for p in self.parameters():
            p.requires_grad = False
        for mi in self.m:
            mi.eval()
        self.N, self.H, self.P, self.penalty, self.bc = N, None, None, penalty, bc #penalty here is just a pair of coefficients
        self.Eps = None
        self.value = float(-0.)
        self.s = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))# self.s is the initial correlations we feed to the ANN

    # def Clip_Norm(self):
    #     norm = torch.norm(self.s)
    #     if norm < 0.7:
    #         self.s = nn.Parameter(self.s * 0.7 / norm)
    #     elif norm > 1.1:
    #         self.s = nn.Parameter(self.s * 1.1 / norm)

    def forward(self, C): # C = [J, g] for simplicity we can choose J=1 and only change g
        # self.Clip_Norm()
        # self.P = self.m(self.s).squeeze() # why relu?
        self.P = 0
        for mi in self.m:
            self.P += mi(self.s).squeeze()
        self.P /= len(self.m)
        if self.bc == 'open':
            self.H = -C[0] * (self.N - 1) * (self.s[0, 5]) - self.N * C[1] * (self.s[0, 0] ) #open bc
        elif self.bc == 'periodic':
            self.H = -C[0] * self.N * (self.s[0, 5]) - self.N * C[1] * (self.s[0, 0]) #periodic bc

        self.Eps = self.H/self.N #energy per site
        self.value = float(self.H + self.penalty[0] * self.P.pow(self.penalty[1]))
        return self.H + self.penalty[0] * self.P.pow(self.penalty[1]) #real peanlty is the output of the ANN


class LagMPS_multi_antif(nn.Module):
    def __init__(self, model, init, N, penalty, bc, C): #init is the initial data we feed to the ANN
        super(LagMPS_multi_antif, self).__init__()
        self.m = model
        for p in self.parameters():
            p.requires_grad = False
        for mi in self.m:
            mi.eval()
        self.N, self.H, self.P, self.penalty, self.bc = N, None, None, penalty, bc #penalty here is just a pair of coefficients
        self.Eps = None
        self.C=C
        self.s = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))# self.s is the initial correlations we feed to the ANN, only change parameter when optimization

    # def Clip_Norm(self):
    #     norm = torch.norm(self.s)
    #     if norm < 0.7:
    #         self.s = nn.Parameter(self.s * 0.7 / norm)
    #     elif norm > 1.1:
    #         self.s = nn.Parameter(self.s * 1.1 / norm)

    def forward(self): # C = [J, g] for simplicity we can choose J=1 and only change g
        # self.Clip_Norm()
        # self.P = self.m(self.s).squeeze() # why relu?
        self.P = 0
        for mi in self.m:
            self.P += mi(self.s).squeeze()
        self.P /= len(self.m)
        if self.bc == 'open':
            self.H = -self.C[0] * (self.N - 1) * (self.s[0, 5]) + self.N * self.C[1] * (self.s[0, 8] ) #open bc
        elif self.bc == 'periodic':
            self.H = -self.C[0] * self.N * (self.s[0, 5]) + self.N * self.C[1] * (self.s[0, 8]) #periodic bc

        self.Eps = self.H/self.N #energy per site

        return self.H + self.penalty[0] * self.P.pow(self.penalty[1]) #real peanlty is the output of the ANN

class LagMPS_multi_lt(nn.Module):
    def __init__(self, model, init, N, penalty, bc, C): #init is the initial data we feed to the ANN
        super(LagMPS_multi_lt, self).__init__()
        self.m = model
        for p in self.parameters():
            p.requires_grad = False
        for mi in self.m:
            mi.eval()
        self.N, self.H, self.P, self.penalty, self.bc = N, None, None, penalty, bc #penalty here is just a pair of coefficients
        self.Eps = None
        self.s = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))# self.s is the initial correlations we feed to the ANN
        self.C = C

    # def Clip_Norm(self):
    #     norm = torch.norm(self.s)
    #     if norm < 0.7:
    #         self.s = nn.Parameter(self.s * 0.7 / norm)
    #     elif norm > 1.1:
    #         self.s = nn.Parameter(self.s * 1.1 / norm)

    def forward(self): # C = [J, g, a] for simplicity we can choose J=1 and only change g
        # self.Clip_Norm()
        # self.P = self.m(self.s).squeeze() # why relu?
        self.P = 0
        for mi in self.m:
            self.P += mi(self.s).squeeze()
        self.P /= len(self.m)
        if self.bc == 'open':
            self.H = -self.C[0] * (self.N - 1) * (self.s[0, 5]) - self.N * (self.C[1] * self.s[0, 0] + self.C[2] * self.s[0, 2])    #open bc
        elif self.bc == 'periodic':
            self.H = -self.C[0] * self.N * (self.s[0, 5]) - self.N * self.C[1] * (self.s[0, 0]) - self.N * self.C[2] * self.s[0, 2] #periodic bc

        self.Eps = self.H/self.N #energy per site

        return self.H + self.penalty[0] * self.P.pow(self.penalty[1]) #real peanlty is the output of the ANN



class LagMPS_multi_XXZ(nn.Module):
    def __init__(self, model, init, N, penalty, bc): #init is the initial data we feed to the ANN
        super(LagMPS_multi_XXZ, self).__init__()
        self.m = model
        for p in self.parameters():
            p.requires_grad = False
        for mi in self.m:
            mi.eval()
        self.N, self.H, self.P, self.penalty, self.bc = N, None, None, penalty, bc #penalty here is just a pair of coefficients
        self.Eps = None
        self.s = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))# self.s is the initial correlations we feed to the ANN

    # def Clip_Norm(self):
    #     norm = torch.norm(self.s)
    #     if norm < 0.7:
    #         self.s = nn.Parameter(self.s * 0.7 / norm)
    #     elif norm > 1.1:
    #         self.s = nn.Parameter(self.s * 1.1 / norm)

    def forward(self, C): # C = [J, g] for simplicity we can choose J=1 and only change g
        # self.Clip_Norm()
        # self.P = self.m(self.s).squeeze() # why relu?
        self.P = 0
        for mi in self.m:
            self.P += mi(self.s).squeeze()
        self.P /= len(self.m)
        # 2021.8.29 need changing to XXZ model
        if self.bc == 'open':
            self.H = -C[0] * (self.N - 1) * (self.s[0, 5]) - self.N * C[1] * (self.s[0, 3] + self.s[0, 4] ) #open bc
        elif self.bc == 'periodic':
            self.H = -C[0] * self.N * (self.s[0, 5]) - self.N * C[1] * (self.s[0, 3] + self.s[0, 4]) #periodic bc

        self.Eps = self.H/self.N #energy per site

        return self.H + self.penalty[0] * self.P.pow(self.penalty[1]) #real peanlty is the output of the ANN
