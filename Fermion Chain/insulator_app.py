import os
import argparse
parser = argparse.ArgumentParser(description='Trainning ...')
parser.add_argument('--gpuid', dest='gpuid',
                    help="assign the GPU to use")
parser.add_argument('--increment', dest='increment', type=float,
                    help="increment of U(V)")
parser.add_argument('--appAmount', dest='appAmount', type=int,
                    help="amount of U(V)")
parser.add_argument('--appN', dest='appN', type=int,
                    help="number of truncation")
parser.add_argument('--appNets', dest='appNets',
                    help="name of the models")
parser.add_argument('--Ytuple', dest='Ytuple',
                    help="tuple of Vs")
parser.add_argument('--epochs', dest='epochs', type=int,
                    help="epochs of SUMT")
parser.add_argument('--iter_times', dest='iter_times', type=int,
                    help="iter_times for Adam or max_iter for LBFGS")
parser.add_argument('--tol', dest='tol', type=float,
                    help="tolerance")
parser.add_argument('--pert', dest='pert',
                    help="perturbation type")
parser.add_argument('--delta', dest='delta', type=float,
                    help="perturbation value")
parser.add_argument('--init_p', dest='init_p', type=float,
                    help="initial penalty coefficient")
parser.add_argument('--rate', dest='rate', type=float,
                    help="multiplier of SUMT")
parser.add_argument('--opt', dest='opt',
                    help="optimizer")
parser.add_argument('--sch', dest='sch',
                    help="learning rate scheduler")
parser.add_argument('--lr', dest='lr', type=float,
                    help="learning rate")
parser.add_argument('--wd', dest='wd', type=float,
                    help="weight decay")
parser.add_argument('--gamma', dest='gamma', type=float,
                    help="gamma")
parser.add_argument('--ss', dest='ss', type=int,
                    help="step size")
parser.add_argument('--app_seed', dest='app_seed', type=int, default=0,
                    help="parameter of torch.manual_seed()")
parser.add_argument('--increase', dest='increase', type=int,
                    help="U(V) increase or decrease")
parser.add_argument('--appgauge', dest='appgauge', type=int,
                        help="gauge transformation or not")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from torch.nn.utils import clip_grad_norm_
from functools import partial
from utils import Optimizer, Scheduler, LagInsulator, LagInsOperator
from testins import penaltydis, gauge_trans
import random
import numpy as np
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import time
import warnings, mkl
warnings.filterwarnings('ignore')
mkl.set_num_threads(1)
plt.switch_backend('agg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameter of Lagrange function
N, increase = args.appN, '+' if bool(args.increase) else '-'
Nets, gauge = list(args.appNets.split(',')), bool(args.appgauge)
init_p, rate, increment, amount, keep_amount = args.init_p, args.rate, args.increment, args.appAmount, 10
epochs, iter_times, num_seed, processors = args.epochs, args.iter_times, args.app_seed, keep_amount
tol, pert, delta, init_amount = args.tol, args.pert, args.delta, processors * 2
opt, sch = args.opt, args.sch  # optimizer & lr_scheduler
lr, wd, gn = args.lr, args.wd, 1.  # learning rate & weight decay & grad norm
momentum, betas = 0.99, (0.9, 0.999)  # momentum
gamma, ss = args.gamma, args.ss  # gamma & step size
length = 6 * N - 1 if gauge else 6 * N + 1
if delta < 1e-10: keep_amount, processors, init_amount = 1, 1, 1

random.seed(num_seed + 1000)
np.random.seed(num_seed + 100)
torch.manual_seed(num_seed)


class optimization():
    def __init__(self, V, U):
        self.V, self.U = V, U

    def __call__(self, init):
        # Lagrange functions
        if abs(self.U) < 50.:
            LH = LagInsulator(N, self.V, self.U, init_p, init, Nets, device).to(device)
        else:
            LH = LagInsOperator(N, self.V, self.U, init_p, init, Nets, device).to(device)

        # optimization
        for epoch in range(epochs):
            if opt == 'LBFGS':
                optimizer = Optimizer('LBFGS', LH, lr=lr, weight_decay=wd, momentum=momentum, betas=betas, max_iter=iter_times)
                scheduler = Scheduler(sch, optimizer, step_size=ss, gamma=gamma, T_max=ss)
                def closure():
                    optimizer.zero_grad()
                    L = LH()
                    L.backward()
                    clip_grad_norm_(filter(lambda p: p.requires_grad, LH.parameters()), gn)
                    scheduler.step()
                    # print(optimizer.state_dict()['param_groups'][0]['lr'], scheduler.get_lr())
                    return L
                optimizer.step(closure)
            else:
                optimizer = Optimizer(opt, LH, lr=lr, weight_decay=wd, momentum=momentum, betas=betas)
                scheduler = Scheduler(sch, optimizer, step_size=ss, gamma=gamma, T_max=ss)
                for j in range(iter_times):
                    L = LH()
                    optimizer.zero_grad()
                    L.backward()
                    clip_grad_norm_(filter(lambda p: p.requires_grad, LH.parameters()), gn)
                    optimizer.step()
                    scheduler.step()
            LH()
            if LH.P.item() > tol * 100 or LH.P.item() < 2e-5:
                break
            else:
                LH.penalty *= rate
        if LH.Pmax.item() < tol:
            if gauge:
                return (LH.H.item(), LH.P.item(), LH.g.data.cpu().numpy())
            else:
                return (LH.H.item(), LH.P.item(), LH.Pg.item(), LH.g.data.cpu().numpy())
        else:
            return (LH.H.item(),)


def plotfigure(x, y, xlabel, ylabel):
    plt.figure()
    if ylabel[:2] == 'Hs':
        for i in range(amount):
            plt.scatter(len(y[i]) * [x[i]], y[i], s=1, c='b')
    else:
        print(ylabel + '=', y)
        f.write('{}={}\n'.format(ylabel, list(y)))
        f.flush()
        plt.plot(x, y, c='k', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(DefaultPath + '/{}{}.png'.format(ylabel, increase))
    plt.close()


def perturbation(r, init):
    random.seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    np.random.seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    data = init.copy()
    if pert == 'normal':
        for i in range(len(r)):
            eta = random.gauss(0, delta)
            if r[i] < 0.5:
                d = np.random.randn(length)
                data[i] += d * eta / np.linalg.norm(d)
            elif r[i] < 0.625:
                data[i][random.randint(0, 2)] += eta
            elif r[i] < 0.75:
                data[i][random.randint(N + 1, N + 2)] += eta
            elif r[i] < 0.875:
                data[i][random.randint(N * 3 - 1, N * 3 + 2)] += eta
            else:
                if gauge:
                    data[i][random.randint(N * 5 - 1, N * 5)] += eta
                else:
                    data[i][random.randint(N * 5 - 1, N * 5 + 2)] += eta
    else:
        for i in range(len(r)):
            eta = random.uniform(-delta, delta)
            if r[i] < 0.5:
                d = np.random.rand(length)
                data[i] += d * eta / np.linalg.norm(d)
            elif r[i] < 0.625:
                data[i][random.randint(0, 2)] += eta
            elif r[i] < 0.75:
                data[i][random.randint(N + 1, N + 2)] += eta
            elif r[i] < 0.875:
                data[i][random.randint(N * 3 - 1, N * 3 + 2)] += eta
            else:
                if gauge:
                    data[i][random.randint(N * 5 - 1, N * 5)] += eta
                else:
                    data[i][random.randint(N * 5 - 1, N * 5 + 2)] += eta
    return data


def checknonexist(g, gini, tolerance=1e-6):
    dif = np.min(1 - (np.abs(np.matmul(gini, g.T).squeeze()) / np.linalg.norm(gini, axis=1)) ** 2 / np.dot(g[0], g[0]))
    return dif > tolerance


def main(x, y):
    Hmin, cdw, bdw = np.ones(amount), np.zeros(amount), np.zeros(amount)
    P, D, Hs = -np.ones(amount), -np.ones(amount), [[] for _ in range(amount)]
    if not gauge: Pg, Dg = -np.ones(amount), -np.ones(amount)
    g, init_g_next = {}, None
    for j in range(len(y)):
        if x_label == 'V':
            if gauge:
                init_g = np.delete(gauge_trans(N, init_set[round(V[0] / 0.05)]), [N * 5, N * 5 + 1], axis=0)[np.newaxis, :]
            else:
                init_g = init_set[round(V[0] / 0.05)][np.newaxis, :]
        else:
            if gauge:
                init_g = np.delete(gauge_trans(N, init_set[round(U[0] / 0.01)]), [N * 5, N * 5 + 1], axis=0)[np.newaxis, :]
            else:
                init_g = init_set[round(U[0] / 0.01)][np.newaxis, :]
        for i in range(amount):
            if x_label == 'V':
                v, u = V[i], U[j]
            else:
                v, u = V[j], U[i]
            H_rec = [Hmin[i]]
            while True:
                if delta < 1e-10:
                    totInits = init_g
                else:
                    perSeeds = int(init_amount / len(init_g))
                    mp.set_start_method('fork', force=True)
                    pool = mp.Pool(processes=perSeeds)
                    totData = pool.imap(partial(perturbation, init=init_g), np.random.rand(perSeeds, len(init_g)))
                    pool.close()
                    pool.join()
                    temp = random.sample(list(totData), init_amount - len(init_g))
                    totInits = np.vstack([init_g] + temp)

                # Parallel
                mp.set_start_method('spawn', force=True)
                pool = mp.Pool(processes=processors)
                totData = pool.map(optimization(v, u), totInits)
                pool.close()
                pool.join()
                for data in totData:
                    Hs[i] += [data[0]]
                    if len(data) > 1:
                        if data[0] < max(H_rec) and checknonexist(data[-1], init_g):
                            if init_g_next is None:
                                H_rec, init_g_next = np.array([data[0]]), data[-1]
                            elif len(init_g_next) == keep_amount:
                                index = np.nanargmax(H_rec)
                                H_rec[index] = data[0]
                                init_g_next[index] = data[-1][0]
                            else:
                                H_rec = np.append(H_rec, data[0])
                                init_g_next = np.vstack((init_g_next, data[-1]))
                            if data[0] < Hmin[i]:
                                Hmin[i], P[i], D[i], cdw[i] = data[0], data[1], penaltydis(N, data[-1][0]), data[-1][0][0]
                                if gauge:
                                    bdw[i] = np.sqrt((data[-1][0][3 * N] - data[-1][0][3 * N + 1]) ** 2) / 2
                                else:
                                    bdw[i] = np.sqrt((data[-1][0][3 * N] - data[-1][0][3 * N + 1]) ** 2 + (
                                                data[-1][0][N * 5] - data[-1][0][N * 5 + 1]) ** 2) / 2
                                    Pg[i], Dg[i] = data[2], penaltydis(N, gauge_trans(N, data[-1][0]))

                if init_g_next is not None:
                        print('V={}\tU={}\tcur_length={}'.format(v, u, len(init_g_next)))
                    # if len(init_g_next) >= int(keep_amount / 2):
                        init_g = init_g_next
                        init_g_next = None
                        break
                else:
                    print('V={}\tU={}\tcur_length={}'.format(v, u, 0))
            g['(%.3f, %.3f)' % (v, u)] = init_g
            np.save(DefaultPath + '/g_{}={}{}.npy'.format(y_label, y, increase), [g])
            if gauge:
                print('V=%.3f\tU=%.3f\tH=%.8f\tP=%.8f\tD=%.8f\tCDW=%.8f\tBDW=%.8f\tNext_init_amount=%d' % (
                v, u, Hmin[i], P[i], D[i], cdw[i], bdw[i], len(init_g)))
                f.write('V=%.3f\tU=%.3f\tH=%.8f\tP=%.8f\tD=%.8f\tCDW=%.8f\tBDW=%.8f\tNext_init_amount=%d\n' % (
                v, u, Hmin[i], P[i], D[i], cdw[i], bdw[i], len(init_g)))
            else:
                print(
                    'V=%.3f\tU=%.3f\tH=%.8f\tP=%.8f\tPg=%.8f\tD=%.8f\tDg=%.8f\tCDW=%.8f\tBDW=%.8f\tNext_init_amount=%d' % (
                    v, u, Hmin[i], P[i], Pg[i], D[i], Dg[i], cdw[i], bdw[i], len(init_g)))
                f.write(
                    'V=%.3f\tU=%.3f\tH=%.8f\tP=%.8f\tPg=%.8f\tD=%.8f\tDg=%.8f\tCDW=%.8f\tBDW=%.8f\tNext_init_amount=%d\n' % (
                    v, u, Hmin[i], P[i], Pg[i], D[i], Dg[i], cdw[i], bdw[i], len(init_g)))
            f.flush()

        plotfigure(x, Hmin, x_label, 'Ham_{}={}'.format(y_label, y[j]))
        plotfigure(x, P, x_label, 'Penalty_{}={}'.format(y_label, y[j]))
        plotfigure(x, D, x_label, 'DisToPhy_{}={}'.format(y_label, y[j]))
        plotfigure(x, cdw, x_label, 'CDW_{}={}'.format(y_label, y[j]))
        plotfigure(x, bdw, x_label, 'BDW_{}={}'.format(y_label, y[j]))
        plotfigure(x, Hs, x_label, 'Hs_{}={}'.format(y_label, y[j]))
        if not gauge:
            plotfigure(x, P, x_label, 'Penalty_gauge_{}={}'.format(y_label, y[j]))
            plotfigure(x, Dg, x_label, 'DisToPhy_gauge_{}={}'.format(y_label, y[j]))


if __name__ == "__main__":
    DefaultPath = 'InsResults/N={}/{}/{}/{}'.format(N, opt, Nets, time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    if amount <= 5: DefaultPath = DefaultPath + '-'
    if not os.path.exists('InsResults/N={}/{}'.format(N, opt)):
        os.mkdir('InsResults/N={}/{}'.format(N, opt))
    if not os.path.exists('InsResults/N={}/{}/{}'.format(N, opt, Nets)):
        os.mkdir('InsResults/N={}/{}/{}'.format(N, opt, Nets))
    if not os.path.exists(DefaultPath):
        os.mkdir(DefaultPath)

    # Ytuple == U
    U = tuple(map(float, args.Ytuple.split(',')))
    V = np.arange(2. - increment * (amount - 1), 2. + increment, increment)
    if not bool(args.increase): V = V[::-1]
    if abs(U[0]) < 50.:
        # InitFile = 'init_ins_ex_N={}_H.npz'.format(N)
        InitFile = 'init_ins_ex_N={}_H_.npz'.format(N)
    elif U[0] < 0.:
        InitFile = 'init_ins_se_N={}_g1.npz'.format(N)
    else:
        InitFile = 'init_ins_de_N={}_g1.npz'.format(N)
    if InitFile == 'init_ins_ex_N={}_H_.npz'.format(N):
        U = list(U)
    init_set = np.load('datasets/StandardForm/{}'.format(InitFile), allow_pickle=True)['arr_0']

    if type(V) == np.ndarray:
        x, y, x_label, y_label = V, U, 'V', 'U'
    else:
        x, y, x_label, y_label = U, V, 'U', 'V'

    f = open(DefaultPath + '/{}={}{}.txt'.format(y_label, y, increase), 'w+')
    f.write('Nets={}\tGPUid={}\nN={}\tgauge={}\tincrement={}\tamount={}\tepochs={}\tinit_p={}\trate={}\n'.format(Nets, args.gpuid, N, gauge, increment, amount, epochs, init_p, rate))
    f.write('InitFile={}\tinit_amount={}\tkeep_amount={}\tprocessors={}\ttol={}\tpert={}\tdelta={}\tnum_seed={}\n'.format(InitFile, init_amount, keep_amount, processors, tol, pert, delta, num_seed))
    f.write('opt={}\tsch={}\tlr={}\twd={}\tgn={}\tmomentum={}\tbetas={}\tgamma={}\tss={}\t'.format(opt, sch, lr, wd, gn, momentum, betas, gamma, ss))
    if opt == 'LBFGS':
        f.write('max_iter={}\n'.format(iter_times))
    else:
        f.write('iter_times={}\n'.format(iter_times))
    f.flush()

    t = time.time()
    main(x, y)
    t = time.time() - t
    print(t)
    f.write('{}\n'.format(t))
    f.close()
