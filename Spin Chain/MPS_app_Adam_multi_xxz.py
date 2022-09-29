import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.nn.utils import clip_grad_norm_
from utils import *
import copy
import random
import numpy as np
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import time
import warnings
plt.switch_backend('agg')
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DefaultPath = '/home/sijing/ML/MPS/'

# hyper-parameter of Lagrange function
L, TYPE, N = 6, 'regression', 1
Net, no = 'Net_sig', ['04','05','06','07','02']
# Net, no = 'Net_sig', ['04','05','02','06','01']
id = ['02']
amount, epochs, iter_times, processors = 50, 1, 200, 15
init_no, init_amount, keep_amount = 0, 100, 10
tol, delta = 1e-3, 5e-2
opt, sch, opt_default = 'Adam', 'StepLR', 'Adam'               # optimizer & lr_scheduler
lr, wd, gn = 1e-3, 0., 1.                 # learning rate & weight decay & grad norm
momentum, betas = 0.99, (0.9, 0.999)      # momentum
gamma, ss = 0.5, 50                       # gamma & step size
eta = 1000/3

class optimization():
    def __init__(self, model, J, g, tol):
        self.model = copy.deepcopy(model)
        self.J = J
        self.tol = tol
        self.g = g

    def __call__(self, init):
        # Lagrange functions
        rate = torch.tensor([2, 1], dtype=torch.float32).to(device)
        init_penalty = torch.tensor([eta, 1], dtype=torch.float32).to(device)
        LH = LagMPS_multi_XXZ(self.model, init, N, init_penalty, 'periodic').to(device)

        # optimization
        for epoch in range(epochs):
            optimizer = Optimizer(opt, LH, lr=lr, weight_decay=wd, momentum=momentum, betas=betas)
            scheduler = Scheduler(sch, optimizer, step_size=ss, gamma=gamma, T_max=ss)
            for j in range(iter_times):
                lag = LH([self.J, self.g])
                optimizer.zero_grad()
                lag.backward()
                clip_grad_norm_(filter(lambda p: p.requires_grad, LH.parameters()), gn)
                optimizer.step()
                scheduler.step()
                # if j != iter_times - 1:
                #     LH.g[0].data[random.randint(0, 2)] += (random.random() * 2 - 1.0) * 0.001
                #     LH.g[0].data[random.randint(N + 1, N + 2)] += (random.random() * 2 - 1.0) * 0.001
                #     LH.g[0].data[random.randint(N * 3 - 1, N * 3 + 2)] += (random.random() * 2 - 1.0) * 0.001
                #     LH.g[0].data[random.randint(N * 5 - 1, N * 5 + 2)] += (random.random() * 2 - 1.0) * 0.001
            if LH.P.item() > self.tol * 10 or LH.P.item() < self.tol:
                break
            else:
                LH.penalty *= rate
        if LH.P.item() < self.tol:
            return (LH.H.item(), LH.s[0, 17].item(), LH.P.item(), LH.Eps.item(), LH.s.data.cpu().numpy())
        else:
            return (LH.H.item(),)


def perturbation(r, init): #to create our initial data, for MPS this should be different
    # random.seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    data = init.copy()
    for i in range(len(init)):
        if r < 0.25:
            data[i][random.randint(0, 2)] += (random.random() * 2 - 1.0) * delta
        elif r < 0.5:
            data[i][random.randint(3, 3 * L)] += (random.random() * 2 - 1.0) * delta
        elif r < 0.75:
            data[i][random.randint(3 * L, 9 * L - 7)] += (random.random() * 2 - 1.0) * delta
        # else:
        #     data[i][random.randint(N * 3 - 1, N * 3 + 2)] += (random.random() * 2 - 1.0) * delta
    # c = data - init
    # print(c.ravel()[np.flatnonzero(c)])
    return data


def checknonexist(s, sini, tolerance=1e-6):
    dif = 1.0
    for sp in sini: #find smallest sin(/theta)
        dif = min(dif, 1 - abs(np.dot(sp, s)) ** 2 / (np.dot(sp, sp) * np.dot(s, s)))
    return dif > tolerance

def plot_g(g, y, y_label):
    print(y_label + '=', y)
    print(y_label + '=', y, file=f)
    plt.figure()
    plt.plot(g, y, c='k', linestyle='-')
    plt.xlabel('g')
    plt.ylabel(y_label)
    plt.savefig(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz/{}.png'.format(L, opt, Net, id[-1], y_label))
    plt.close()


def plot_Hs(g, Hs):
    plt.figure()
    for i in range(amount):
        plt.scatter(len(Hs[i]) * [g[i]], Hs[i], s=1, c='b')
    plt.xlabel('g')
    plt.ylabel('Hs')
    plt.savefig(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz/Hs.png'.format(L, opt, Net, id[-1]))
    plt.close()



def main():
    Hmin, OP, P, Hs, Eps = np.ones(amount), np.zeros(amount), np.ones(amount), [[] for _ in range(amount)], np.ones(amount)

    # slist = [[] for _ in range(amount)]
    g = 1. + 0.02 * np.arange(1, amount + 1)

    print('Eta:\t', eta, file=f)

    init_s, init_s_next = s_0[np.newaxis, :], None #need to define s_0
    s = [{str(0): init_s}] #what is our initial data?How to determine? we need init_s
    for i in range(amount):
        coefficients = torch.tensor([1., g[i]]).to(device) # [J, g]
        while True:
            perSeeds = int(init_amount / len(init_s))
            mp.set_start_method('fork', force=True)
            pool = mp.Pool(processes=perSeeds)
            totData = pool.imap(partial(perturbation, init=init_s), np.random.rand(perSeeds))
            pool.close()
            pool.join()
            totInits = np.vstack((x for x in totData))

            # Parallel
            mp.set_start_method('spawn', force=True)
            pool = mp.Pool(processes=processors)
            totData = pool.map(optimization(model=model, J=coefficients[0], g=coefficients[1], tol=tol), totInits)
            pool.close()
            pool.join()
            for data in totData:  # data:[LH.H.item(), LH.s[0, 17].item(), LH.P.item(), LH.Eps.item(), LH.s.data.cpu().numpy()]
                Hs[i] += [data[0]]
                if len(data) > 1:
                    if data[0] < Hmin[i] and checknonexist(data[4][0], init_s):
                        if init_s_next is None:
                            init_s_next = data[4]
                        elif len(init_s_next) == keep_amount:
                            init_s_next = np.vstack((data[4], init_s_next[:-1]))
                        else:
                            init_s_next = np.vstack((data[4], init_s_next))
                        Hmin[i], OP[i], P[i], Eps[i] = data[0], data[1], data[2], data[3]

            # # No parallel
            # for init in totInits: #totInits is a list of our initial points in the manifold
            #     # Lagrange functions
            #     rate = torch.tensor([2, 1], dtype=torch.float32).to(device)
            #     init_penalty = torch.tensor([1e3, 1], dtype=torch.float32).to(device) #the coefficient to the penalty
            #     LH = LagMPS_multi(model, init, N, init_penalty, 'periodic').to(device)#H+L the model will be loaded later
            #
            #     # optimization
            #     for epoch in range(epochs):
            #         optimizer = Optimizer(opt, LH, lr=lr, weight_decay=wd, momentum=momentum, betas=betas) #这里不应该有wd
            #         scheduler = Scheduler(sch, optimizer, step_size=ss, gamma=gamma, T_max=ss)
            #         for j in range(iter_times):
            #             lag = LH(coefficients)
            #             optimizer.zero_grad()
            #             lag.backward()
            #             clip_grad_norm_(filter(lambda p: p.requires_grad, LH.parameters()), gn)
            #             optimizer.step()
            #             scheduler.step()
            #             # if j != iter_times - 1:
            #             #     LH.g[0].data[random.randint(0, 2)] += (random.random() * 2 - 1.0) * 0.001
            #             #     LH.g[0].data[random.randint(N + 1, N + 2)] += (random.random() * 2 - 1.0) * 0.001
            #             #     LH.g[0].data[random.randint(N * 3 - 1, N * 3 + 2)] += (random.random() * 2 - 1.0) * 0.001
            #             #     LH.g[0].data[random.randint(N * 5 - 1, N * 5 + 2)] += (random.random() * 2 - 1.0) * 0.001
            #         if LH.P > tol * 10 or LH.P < tol:
            #             break
            #         else:
            #             LH.penalty *= rate
            #     Hs[i] += [LH.H.item()]
            #
            #     if LH.P < tol:
            #         temp = LH.s.data.cpu().numpy()
            #         if LH.H < Hmin[i] and checknonexist(temp[0], init_s):
            #             if init_s_next is None:
            #                 init_s_next = temp
            #             elif len(init_s_next) == keep_amount: #仅保留最近的10个s
            #                 init_s_next = np.vstack((temp, init_s_next[:-1]))
            #             else:
            #                 init_s_next = np.vstack((temp, init_s_next))
            #             Hmin[i], OP[i], P[i], Eps[i] = LH.H.item(), LH.s[0, 17].item(), LH.P.item(), LH.Eps.item()

            if init_s_next is not None:
                init_s = init_s_next
                init_s_next = None
                break #break while

        s += [{'%.3f' % (g[i]): init_s}]
        print(
            'g=%.3f\tH=%.8f\tP=%.8f\tOP=%.8f\tEps=%.8f\tNext_init_amount=%d' % (g[i], Hmin[i], P[i], OP[i], Eps[i], len(init_s)))
        print(
            'g=%.3f\tH=%.8f\tP=%.8f\tOP=%.8f\tEps=%.8f\tNext_init_amount=%d' % (g[i], Hmin[i], P[i], OP[i], Eps[i], len(init_s)),
            file=f) #f will be defined later

    plot_g(g, Hmin, 'Ham')
    plot_g(g, P, 'Penalty')
    plot_g(g, OP, 'OP')
    plot_Hs(g, Hs)
    np.save(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz/s.npy'.format(L, opt, Net, id[-1]), s)
    np.save(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz/E.npy'.format(L, opt, Net, id[-1]), Hmin)
    np.save(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz/P.npy'.format(L, opt, Net, id[-1]), P)
    np.save(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz/Es.npy'.format(L, opt, Net, id[-1]), Hs)


if __name__ == "__main__":
    if not os.path.exists(DefaultPath + 'OUT/mps_app/L={}/{}'.format(L, opt)):
        os.mkdir(DefaultPath + 'OUT/mps_app/L={}/{}'.format(L, opt))
    if not os.path.exists(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz'.format(L, opt, Net, id[-1])):
        os.mkdir(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz'.format(L, opt, Net, id[-1]))

    init_set = \
        np.load(DefaultPath + 'datasets/StandardForm/init_mps_exact_L={}.npz'.format(L), allow_pickle=True)[
            'arr_0']

    s_0 = init_set
    # modelfile = DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}/{}_{}_dynamical.pkl'.format(TYPE, L, opt, Net, no[-1], Net, no[-1])
    # model = torch.load(modelfile).to(device)
    model = []
    for i in range(len(no)):
        modelfile = torch.load(DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}/{}_{}_dynamical.pkl'.format(TYPE, L, opt_default, Net, no[i], Net, no[i])).to(device)
        model.append(modelfile)

    f = open(DefaultPath + 'OUT/mps_app/L={}/{}/{}_{}_multi_xxz/{}_{}_multi_xxz.txt'.format(L, opt, Net, id[-1], Net, id[-1]), 'w+')
    print(Net, id[-1], no, file=f)
    t = time.time()
    main()
    print(time.time() - t)
    f.close()
