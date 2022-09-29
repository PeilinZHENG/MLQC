import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import optim
from torch.optim import lr_scheduler
import warnings, mkl
warnings.filterwarnings('ignore')
mkl.set_num_threads(1)

DefaultPath = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Optimizer(opt, net, lr=1e-3, weight_decay=0., momentum=0., betas=(0.9, 0.999), max_iter=20):
    params = filter(lambda p: p.requires_grad, net.parameters())
    if opt == 'Adadelta':
        return optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'Adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'Adam':
        return optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'AdamW':
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'SparseAdam':
        return optim.SparseAdam(params, lr=lr, betas=betas)
    elif opt == 'Adamax':
        return optim.Adamax(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'ASGD':
        return optim.ASGD(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'LBFGS':
        return optim.LBFGS(params, lr=lr, max_iter=max_iter)
    elif opt == 'NAdam':
        return optim.NAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'RAdam':
        return optim.RAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif opt == 'RMSprop':
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif opt == 'Rprop':
        return optim.Rprop(params, lr=lr)
    elif opt == 'SGD':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NameError('Wrong Optimizer Type')


def Scheduler(sch, optimizer, base_lr=1e-3, max_lr=1., gamma=1., factor=0.333, lr_lambda=lambda epoch: 1, step_size=100,
              milestones=(10,), epochs=10, steps_per_epoch=100, T_max=1, T_0=1, T_mult=1):
    if type(sch) == str:
        return singleScheduler(sch, optimizer, base_lr, max_lr, gamma, factor, lr_lambda, step_size, milestones, epochs,
                               steps_per_epoch, T_max, T_0, T_mult)
    elif type(sch) == tuple or type(sch) == list:
        schedulers = []
        for s in sch[-1]:
            schedulers.append(
                singleScheduler(s, optimizer, base_lr, max_lr, gamma, factor, lr_lambda, step_size, milestones,
                                epochs, steps_per_epoch, T_max, T_0, T_mult))
        if sch[0] == 'ChainedScheduler':
            return lr_scheduler.ChainedScheduler(schedulers)
        elif sch[0] == 'SequentialLR':
            return lr_scheduler.SequentialLR(optimizer, schedulers, milestones)
        else:
            raise NameError('Wrong lr_Scheduler Type')
    else:
        raise NameError('Wrong lr_Scheduler Type')


def singleScheduler(sch, optimizer, base_lr=1e-3, max_lr=1., gamma=1., factor=0.333, lr_lambda=lambda epoch: 1,
                    step_size=100, milestones=(10,), epochs=10, steps_per_epoch=100, T_max=1, T_0=1, T_mult=1):
    if sch == 'LambdaLR':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sch == 'MultiplicativeLR':
        return lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
    elif sch == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
    elif sch == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    elif sch == 'ConstantLR':
        return lr_scheduler.ConstantLR(optimizer, factor=factor, total_iters=milestones[0])
    elif sch == 'LinearLR':
        return lr_scheduler.LinearLR(optimizer, start_factor=gamma, end_factor=factor, total_iters=milestones[0])
    elif sch == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma)
    elif sch == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    elif sch == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor)
    elif sch == 'CyclicLR':
        return lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, gamma=gamma)
    elif sch == 'OneCycleLR':
        return lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    elif sch == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult)
    else:
        raise NameError('Wrong lr_Scheduler Type')


class LagInsulator(nn.Module):
    def __init__(self, N, V, U, penalty, init, Nets, device):
        super(LagInsulator, self).__init__()
        self.device = device
        self.model = nn.ModuleList()
        for Net in Nets:
            m = torch.load('models/OneDInsulator/N={}/{}.pkl'.format(N, Net), map_location=self.device)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.model.append(m)
        # print('after:{}'.format(self.model[0].simplemodel.fc1.weight.requires_grad))
        self.N, self.V, self.U, self.penalty = N, V, U, penalty
        self.H, self.P, self.Pmax = torch.tensor(0.), torch.tensor(-1.), torch.tensor(-1.)
        self.g = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))
        if self.g.shape[1] == 6 * self.N + 1:
            self.gauge, self.Pg  = False, torch.tensor(-1.)
        else:
            self.gauge, self.Pg = True, None

    def Clip_Norm(self):
        norm = torch.norm(self.g)
        if norm < 0.75:
            self.g = nn.Parameter(self.g * 0.7 / norm)
        elif norm > 1.:
            self.g = nn.Parameter(self.g / norm)

    def gauge_trans(self):
        # gp[-N / 2], ..., gp[-1 / 2], gp[1 / 2], ..., gp[N / 2]
        gp = self.g[0, self.N * 2 + 1:self.N * 4 + 1].detach() + 1j * self.g[0, self.N * 4 + 1:].detach()
        theta = torch.conj(gp[self.N]) / torch.abs(gp[self.N])
        phi = torch.conj(gp[self.N - 1]) / torch.abs(gp[self.N - 1])
        # gauge: [phi ** N * theta ** (-N + 1), ..., phi, theta, ..., phi ** (-N + 1) * theta ** N]
        gp *= phi ** self.N / theta ** (self.N - 1) * (theta / phi) ** torch.arange(self.N * 2).to(self.device)
        # g[0], g[1], ..., g[N]
        g_ = self.g[0, :self.N + 1].detach() + \
             1j * torch.cat((torch.zeros(1, device=self.device), self.g[0, self.N + 1:self.N * 2 + 1].detach()))
        # gauge: [1, phi ** (-1) * theta, ..., phi ** (-N) * theta ** N]
        g_ *= (theta / phi) ** torch.arange(self.N + 1).to(self.device)
        return torch.cat((torch.real(g_), torch.imag(g_)[1:], torch.real(gp), torch.imag(gp))).unsqueeze(0)

    def forward(self):
        self.Clip_Norm()
        P_ = torch.cat([m(self.g) for m in self.model], dim=1).squeeze()
        self.Pmax, self.P = torch.max(P_), torch.mean(P_)
        if self.gauge:
            self.H = -(self.g[0, self.N * 3 + 1] + self.g[0, self.N * 3]) + self.V * (
                        0.5 - 0.5 * self.g[0, 0] ** 2 - 0.25 * self.g[0, self.N * 3 + 1] ** 2 - 0.25 * self.g[
                    0, self.N * 3] ** 2) + 0.5 * self.U * (
                                 self.g[0, 0] ** 2 - self.g[0, 1] ** 2 - self.g[0, self.N + 1] ** 2)
        else:
            self.H = -(self.g[0, self.N * 3 + 1] + self.g[0, self.N * 3]) + self.V * (
                        0.5 - 0.5 * self.g[0, 0] ** 2 - 0.25 * self.g[0, self.N * 3 + 1] ** 2 - 0.25 * self.g[
                    0, self.N * 5 + 1] ** 2 - 0.25 * self.g[0, self.N * 3] ** 2 - 0.25 * self.g[
                            0, self.N * 5] ** 2) + 0.5 * self.U * (
                                 self.g[0, 0] ** 2 - self.g[0, 1] ** 2 - self.g[0, self.N + 1] ** 2)
            with torch.no_grad():
                temp = self.gauge_trans()
                self.Pg = torch.mean(torch.cat([m(temp) for m in self.model], dim=1))

        return self.H + self.penalty * self.P


class LagInsOperator(nn.Module):
    def __init__(self, N, V, U, penalty, init, Nets, device):
        super(LagInsOperator, self).__init__()
        self.device = device
        self.model = nn.ModuleList()
        for Net in Nets:
            m = torch.load('models/OneDInsulator/N={}/{}.pkl'.format(N, Net), map_location=self.device)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            self.model.append(m)
        # print('after:{}'.format(self.model[0].simplemodel.fc1.weight.requires_grad))
        self.N, self.V, self.U, self.penalty = N, V, U, penalty
        self.H, self.P, self.Pmax = torch.tensor(0.), torch.tensor(-1.), torch.tensor(-1.)
        self.g = nn.Parameter(torch.from_numpy(init).float().unsqueeze(0))
        if self.g.shape[1] == 6 * self.N + 1:
            self.gauge, self.Pg  = False, torch.tensor(-1.)
        else:
            self.gauge, self.Pg = True, None

    def Clip_Norm(self):
        norm = torch.norm(self.g)
        if norm < 0.75:
            self.g = nn.Parameter(self.g * 0.7 / norm)
        elif norm > 1.:
            self.g = nn.Parameter(self.g / norm)

    def gauge_trans(self):
        # gp[-N / 2], ..., gp[-1 / 2], gp[1 / 2], ..., gp[N / 2]
        gp = self.g[0, self.N * 2 + 1:self.N * 4 + 1].detach() + 1j * self.g[0, self.N * 4 + 1:].detach()
        theta = torch.conj(gp[self.N]) / torch.abs(gp[self.N])
        phi = torch.conj(gp[self.N - 1]) / torch.abs(gp[self.N - 1])
        # gauge: [phi ** N * theta ** (-N + 1), ..., phi, theta, ..., phi ** (-N + 1) * theta ** N]
        gp *= phi ** self.N / theta ** (self.N - 1) * (theta / phi) ** torch.arange(self.N * 2).to(self.device)
        # g[0], g[1], ..., g[N]
        g_ = self.g[0, :self.N + 1].detach() + \
             1j * torch.cat((torch.zeros(1, device=self.device), self.g[0, self.N + 1:self.N * 2 + 1].detach()))
        # gauge: [1, phi ** (-1) * theta, ..., phi ** (-N) * theta ** N]
        g_ *= (theta / phi) ** torch.arange(self.N + 1).to(self.device)
        return torch.cat((torch.real(g_), torch.imag(g_)[1:], torch.real(gp), torch.imag(gp))).unsqueeze(0)

    def forward(self):
        self.Clip_Norm()
        P_ = torch.cat([m(self.g) for m in self.model], dim=1).squeeze()
        self.Pmax, self.P = torch.max(P_), torch.mean(P_)
        self.H = -(self.g[0, self.N * 3 + 1] + self.g[0, self.N * 3]) * self.g[0, 1].pow(self.V) # g0 or g1
        if not self.gauge:
            with torch.no_grad():
                temp = self.gauge_trans()
                self.Pg = torch.mean(torch.cat([m(temp) for m in self.model], dim=1))

        return self.H + self.penalty * self.P


class LoadInsulatorData_Reg(Dataset):
    def __init__(self, dataload, scale):
        self.dataload = dataload
        self.scale = scale

    def __getitem__(self, index):
        g = self.dataload[index]
        label = torch.tensor([g[-1]], dtype=torch.float32)
        # label = torch.tensor([np.tanh(self.scale * g[-1])], dtype=torch.float32)
        g = torch.from_numpy(g[:-1]).float()
        return g, label

    def __len__(self):
        return len(self.dataload)


class LoadInsulatorData_Cla(Dataset):
    def __init__(self, dataload):
        self.dataload = dataload

    def __getitem__(self, index):
        g = self.dataload[index]
        label = torch.tensor(g[-1], dtype=torch.long)
        g = torch.from_numpy(g[:-1]).float()
        return g, label

    def __len__(self):
        return len(self.dataload)

