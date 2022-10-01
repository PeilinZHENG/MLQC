import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy import optimize
import warnings, mkl
warnings.filterwarnings('ignore')
mkl.set_num_threads(1)


# The correlations include:
# g_0: real part only.
# g_1 to g_N: real and imaginary parts.
# g_1/2-N to g_N-1/2: real and imaginary parts.
# 6N+1 real numbers in total.

# Gauge transformation of gp+/-1/2
def gauge_trans(N, g_):
    assert len(g_) == 6 * N + 1
    # gp[-N / 2], ..., gp[-1 / 2], gp[1 / 2], ..., gp[N / 2]
    gp = g_[N * 2 + 1:N * 4 + 1] + 1j * g_[N * 4 + 1:]
    if np.absolute(gp[N]) < 1e-8 or np.absolute(gp[N - 1]) < 1e-8:
        return g_
    else:
        theta = np.conj(gp[N]) / np.absolute(gp[N])
        phi = np.conj(gp[N - 1]) / np.absolute(gp[N - 1])
    # gauge: [phi ** N * theta ** (-N + 1), ..., phi, theta, ..., phi ** (-N + 1) * theta ** N]
    gp *= phi ** N / theta ** (N - 1) * (theta / phi) ** np.arange(N * 2)

    # g[0], g[1], ..., g[N]
    g = g_[:N + 1] + 1j * np.concatenate((np.zeros(1), g_[N + 1:N * 2 + 1]))
    # gauge: [1, phi ** (-1) * theta, ..., phi ** (-N) * theta ** N]
    g *= (theta / phi) ** np.arange(N + 1)
    return np.concatenate((np.real(g), np.imag(g)[1:], np.real(gp), np.imag(gp)))


# The cost of violation of physical constraints.
def penaltytoE(disqr):
    return disqr


# Given the correlation parameters g, evaluate whether the physical constraints are satisfied.
def penaltydis(N, g_):
    smax = int(N / 2)
    g = np.concatenate((np.flip(g_[1:N + 1]), g_[:N + 1])) + 1j * np.concatenate(
        (-np.flip(g_[N + 1:N * 2 + 1]), np.zeros(1), g_[N + 1:N * 2 + 1]))
    if len(g_) == 6 * N + 1:
        gp = g_[N * 2 + 1:N * 4 + 1] + 1j * g_[N * 4 + 1:]
    else:
        gp = g_[N * 2 + 1:N * 4 + 1] + 1j * np.insert(g_[N * 4 + 1:], N - 1, [0., 0.])
    residue = np.absolute(np.dot(g, g.conj()) + np.dot(gp, gp.conj()) - 1) ** 2
    for s in np.arange(1, smax + 1):
        residue += np.absolute(np.dot(g[s:], g[:N * 2 + 1 - s].conj()) + np.dot(gp[s:], gp[:N * 2 - s].conj())) ** 2
    return residue


def g2p(N, g):
    if g.ndim == 1:
        C0 = (1 + g[0]) / 2  # C^AA_0
        reC, imC, reCp = g[1:N + 1] / 2, g[N + 1:2 * N + 1] / 2, g[N * 2 + 1:N * 4 + 1] / 2
        if g.shape[0] == 6 * N + 1:
            imCp = g[N * 4 + 1:] / 2
        else:
            imCp = np.insert(g[N * 4 + 1:], N - 1, [0., 0.]) / 2
    else:
        C0 = (1 + g[0, :]) / 2  # C^AA_0
        reC, imC, reCp = g[1:N + 1, :] / 2, g[N + 1:2 * N + 1, :] / 2, g[N * 2 + 1:N * 4 + 1, :] / 2
        if g.shape[0] == 6 * N + 1:
            imCp = g[N * 4 + 1:, :] / 2
        else:
            imCp = np.insert(g[N * 4 + 1:, :], N - 1, np.zeros((2, 1)), axis=0) / 2
    return np.concatenate(([C0], reC + C0, imC + C0, reCp + 0.5, imCp + 0.5), axis=0)  # shape = (6 * N + 1, amount)


def p2g(N, p):
    if p.ndim == 1:
        C0 = p[0]  # C^AA_0
        reC, imC = p[1:N + 1] - C0, p[N + 1:2 * N + 1] - C0
        reCp, imCp = p[N * 2 + 1:N * 4 + 1] - 0.5, p[N * 4 + 1:] - 0.5
        if abs(imCp[N - 1]) < 1e-8 and abs(imCp[N]) < 1e-8:
            imCp = np.delete(imCp, [N - 1, N], axis=0)
    else:
        C0 = p[0, :]  # C^AA_0
        reC, imC = p[1:N + 1, :] - C0, p[N + 1:2 * N + 1, :] - C0
        reCp, imCp = p[N * 2 + 1:N * 4 + 1, :] - 0.5, p[N * 4 + 1:, :] - 0.5
        if abs(max(imCp[N - 1])) < 1e-8 and abs(max(imCp[N])) < 1e-8:
            imCp = np.delete(imCp, [N - 1, N], axis=0)
    return np.concatenate(([C0 * 2 - 1], reC * 2, imC * 2, reCp * 2, imCp * 2),
                          axis=0)  # shape = (6 * N + 1, amount) or (6 * N - 1, amount)


def dg2dp(N, dg):
    if dg.ndim == 1:
        if dg.shape[0] != 6 * N + 1:
            dg = np.insert(dg, 5 * N, [0., 0.])
        dp = np.zeros(dg.shape)
        dp[0] = dg[0] / 2
        dp[1:2 * N + 1] = (dg[1:2 * N + 1] + dg[0]) / 2
        dp[2 * N + 1:] = dg[2 * N + 1:] / 2
    else:
        if dg.shape[0] != 6 * N + 1:
            dg = np.insert(dg, 5 * N, np.zeros((2, 1)), axis=0)
        dp = np.zeros(dg.shape)
        dp[0, :] = dg[0, :] / 2
        dp[1:2 * N + 1, :] = (dg[1:2 * N + 1, :] + dg[0, :]) / 2
        dp[2 * N + 1:, :] = dg[2 * N + 1:, :] / 2
    return dp  # shape = (6 * N + 1, amount)


def dp2dg(N, dp):
    dg = np.zeros(dp.shape)
    if dp.ndim == 1:
        dg[0] = 2 * dp[0]
        dg[1:2 * N + 1] = 2 * (dp[1:2 * N + 1] - dp[0])
        dg[2 * N + 1:] = 2 * dp[2 * N + 1:]
    else:
        dg[0, :] = 2 * dp[0, :]
        dg[1:2 * N + 1, :] = 2 * (dp[1:2 * N + 1, :] - dp[0, :])
        dg[2 * N + 1:, :] = 2 * dp[2 * N + 1:, :]
    return dg


def genweight(N, alpha=0.3, gauge=False):
    scale = np.exp(-alpha * np.arange(N))
    if gauge:
        weight = np.concatenate(([1.], scale, scale, scale[::-1], scale, scale[:0:-1], scale[1:]), axis=0)
    else:
        weight = np.concatenate(([1.], scale, scale, scale[::-1], scale, scale[::-1], scale), axis=0)
    return weight


def penaltyMSE(dg, projector):  # shape = (6 * N + 1, )
    if dg.shape[0] != 6 * N + 1:
        dg_ = np.insert(dg, 5 * N, [0., 0.])
    else:
        dg_ = dg.copy()
    dg_ -= np.dot(projector, dg_)
    penalty = dg_ ** 2
    return np.sum(penalty)


def penaltyME(p0, dp, projector):  # shape = (6 * N + 1, )
    dp_ = dp.copy()
    dp_ -= np.dot(projector, dp_)
    penalty = dp_ ** 2 / (2 * p0 * (1 - p0))
    return np.sum(penalty)


def penaltyME0(p0, dp):  # shape = (6 * N + 1, )
    penalty0 = (1 / 2 / p0 + 1 / 2 / (1 - p0)) * dp ** 2
    return np.sum(penalty0)


# Input a set of correlation parameters g, return a set of physical correlation parameters g via Wannier-wave-function construction method.
def maptophysical(N, g, kN=1000, sample=False, gauge=False):
    reg = np.expand_dims(np.concatenate((np.flip(g[1:N + 1]), g[:N + 1])), axis=0)
    img = np.expand_dims(np.concatenate((-np.flip(g[N + 1:N * 2 + 1]), np.zeros(1), g[N + 1:N * 2 + 1])), axis=0)
    regp = np.expand_dims(g[N * 2 + 1:N * 4 + 1], axis=0)
    imgp = np.expand_dims(g[N * 4 + 1:], axis=0)
    k = np.expand_dims(np.arange(kN) * 2 * np.pi / kN, axis=0)
    if sample:  # 如果sample为true，不就会得到恒定的fk和fkt吗
        fk = 0.6 * np.ones((1, kN))  # 能隙？(是的)
        fkt = np.cos(k / 2)  # fkt是与k有关的
    else:
        r = np.expand_dims(np.arange(N * 2 + 1), axis=1) - N
        fk = np.matmul(reg + 1j * img, np.cos(np.matmul(r, k)) - 1j * np.sin(np.matmul(r, k)))
        r = np.expand_dims(np.arange(N * 2), axis=1) - N + 0.5
        fkt = np.matmul(regp + 1j * imgp, np.cos(np.matmul(r, k)) - 1j * np.sin(np.matmul(r, k)))
    amp = np.absolute(fk) ** 2 + np.absolute(fkt) ** 2
    fk = fk / np.sqrt(amp)
    fkt = fkt / np.sqrt(amp)
    # gp[-N / 2], ..., gp[-1 / 2], gp[1 / 2], ..., gp[N / 2]
    rp = np.expand_dims(np.arange(N * 2), axis=0) - N + 0.5
    temp = np.squeeze(np.matmul(fkt, np.cos(np.matmul(k.T, rp)) + 1j * np.sin(np.matmul(k.T, rp)))) / kN
    if gauge:
        theta = np.conj(temp[N]) / np.absolute(temp[N])
        phi = np.conj(temp[N - 1]) / np.absolute(temp[N - 1])
        # gauge: [phi ** N * theta ** (-N + 1), ..., phi, theta, ..., phi ** (-N + 1) * theta ** N]
        temp *= phi ** N / theta ** (N - 1) * (theta / phi) ** np.arange(N * 2)
        fkt = np.matmul(np.expand_dims(temp, axis=0), np.cos(np.matmul(rp.T, k)) - 1j * np.sin(np.matmul(rp.T, k)))
        newimgp = np.delete(np.imag(temp), [N - 1, N], axis=0)
    else:
        newimgp = np.imag(temp)
    newregp = np.real(temp)
    # g[0], g[1], ..., g[N]
    r = np.expand_dims(np.arange(N + 1), axis=0)
    temp = np.squeeze(np.matmul(fk, np.cos(np.matmul(k.T, r)) + 1j * np.sin(np.matmul(k.T, r)))) / kN
    if gauge:
        # gauge: [1, phi ** (-1) * theta, ..., phi ** (-N) * theta ** N]
        temp *= (theta / phi) ** np.arange(N + 1)
        r_ = np.expand_dims(np.arange(N * 2 + 1), axis=1) - N
        fk = np.matmul(np.expand_dims(np.concatenate((temp[:0:-1], temp)), axis=0),
                       np.cos(np.matmul(r_, k)) - 1j * np.sin(np.matmul(r_, k)))
    g0 = np.concatenate((np.real(temp), np.imag(temp)[1:], newregp, newimgp))

    vp2 = np.tile(fkt, (2 * N, 1)) * (np.cos(np.matmul(rp.T, k)) + 1j * np.sin(np.matmul(rp.T, k)))
    dv2 = np.zeros((6 * N + 1, kN))
    dv2[2 * N + 1: 4 * N + 1] = -np.imag(vp2)
    dv2[4 * N + 1: 6 * N + 1] = np.real(vp2)
    dv2 /= np.linalg.norm(dv2, axis=0)

    fk *= (fkt / np.absolute(fkt))
    fkt = np.absolute(fkt)
    v = -np.tile(fkt, (N + 1, 1)) * (np.cos(np.matmul(r.T, k)) + 1j * np.sin(np.matmul(r.T, k)))
    vp = np.tile(fk, (2 * N, 1)) * (np.cos(np.matmul(rp.T, k)) + 1j * np.sin(np.matmul(rp.T, k)))
    dv = np.zeros((6 * N + 1, kN))
    dv[0] = np.real(v[0])
    dv[1:N + 1] = np.real(v[1:N + 1]) + np.real(v[0])
    dv[N + 1:2 * N + 1] = np.imag(v[1:N + 1]) + np.real(v[0])
    dv[2 * N + 1: 4 * N + 1] = np.real(vp)
    dv[4 * N + 1: 6 * N + 1] = np.imag(vp)
    dv /= np.linalg.norm(dv, axis=0)
    return g0, np.concatenate((dv, dv2), axis=1)  # shape = (6 * N + 1, 2 * kN)


def create_physical_g(N, alpha=0.3, gauge=False):  # 可以用penalty函数来检测是否物理
    g_0 = np.random.rand(1)
    g_1 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_2 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_3 = np.flip(np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N)))
    g_4 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_5 = np.flip(np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N)))
    g_6 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    gx = np.concatenate((g_0, g_1, g_2, g_3, g_4, g_5, g_6), axis=0)
    g, dv = maptophysical(N, gx, kN=1000, sample=False, gauge=gauge)
    return g, dv


def create_CDW_g(N, alpha=0.3, gauge=False):  # 可以用penalty函数来检测是否物理
    g_0 = np.random.rand(1)
    g_1 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_2 = np.zeros(N)
    g_4 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_3 = np.flip(g_4)
    g_6 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_5 = np.flip(g_6)
    gx = np.concatenate((g_0, g_1, g_2, g_3, g_4, g_5, g_6), axis=0)
    g, dv = maptophysical(N, gx, kN=1000, sample=False, gauge=gauge)
    return g, dv


def create_BDW_g(N, alpha=0.3, gauge=False):  # 可以用penalty函数来检测是否物理
    g_0 = np.zeros(1)
    g_1 = np.zeros(N)
    g_2 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_3 = np.flip(np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N)))
    g_4 = np.exp(-alpha * np.arange(int(N))) * (-1 + 2 * np.random.rand(N))
    g_5 = np.zeros(N)
    g_6 = np.zeros(N)
    gx = np.concatenate((g_0, g_1, g_2, g_3, g_4, g_5, g_6), axis=0)
    g, dv = maptophysical(N, gx, kN=1000, sample=False, gauge=gauge)
    return g, dv


def Hamsolve(N, Delta, kN=1000):
    newreg = np.zeros(N + 1)
    newimg = np.zeros(N + 1)
    newregp = np.zeros(N * 2)
    newimgp = np.zeros(N * 2)
    for i in range(kN):
        k = 2 * np.pi * i / kN
        # fk = Delta # g0
        fk = 2 * Delta * np.cos(k) # g1
        fkt = 2 * np.cos(k / 2)
        amp = abs(fk) ** 2 + abs(fkt) ** 2
        fk = fk / np.sqrt(amp)
        fkt = fkt / np.sqrt(amp)
        for r in range(N + 1):
            temp = fk * (np.cos(k * r) + 1j * np.sin(k * r))
            newreg[r] += temp.real / kN
            newimg[r] += temp.imag / kN
        for rp in range(N * 2):
            temp = fkt * (np.cos(k * (rp - N + 0.5)) + 1j * np.sin(k * (rp - N + 0.5)))
            newregp[rp] += temp.real / kN
            newimgp[rp] += temp.imag / kN
    return np.concatenate((newreg, newimg[1:], newregp, newimgp))


def Ham(g, V, U=0.):
    if abs(U) < 50.:
        return -(g[N * 3 + 1] + g[N * 3]) + V * (
                    0.5 - 0.5 * g[0] ** 2 - 0.25 * g[N * 3 + 1] ** 2 - 0.25 * g[N * 5 + 1] ** 2 - 0.25 * g[
                N * 3] ** 2 - 0.25 * g[N * 5] ** 2) + 0.5 * U * (g[0] ** 2 - g[1] ** 2 - g[N + 1] ** 2)
    else:
        return -(g[N * 3] + g[N * 3 + 1]) * g[1] ** V # g0 or g1


def E(g, V, U=0., eta=1000.):
    return Ham(g, V, U) + penaltytoE(penaltydis(N, g)) * eta


def plotfig(x, y, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y, c='k', linestyle='-', marker='+')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()


def EtaExtrap():
    # An example of g.
    g1 = maptophysical(N, np.zeros(N * 6 + 1), sample=True)
    temp = np.array([2 * (-1) ** r / np.pi / (2 * r + 1) for r in range(N)])
    g2 = np.concatenate((np.zeros(N * 2 + 1), np.flip(temp), temp, np.zeros(N * 2)))
    g3 = np.ones(N * 6 + 1)

    f = open('OUT/insulator_app/testins.txt', 'w+')
    V = 0.5 + 0.05 * np.arange(31)
    Eta = [1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 500, 1000]
    f.write('Eta={}\tV={}\n'.format(Eta, list(V)))
    g = {}
    for (index, eta) in enumerate(Eta):
        init = g2  # create_physical_g(N, gauge=False)
        PD, H, L = [], [], []
        for i in V:
            result = optimize.minimize(partial(E, V=i, eta=eta), init, method="BFGS", tol=1e-10)  # 取g3会发生对称性破缺
            init = result.x
            g['(%.2f, %.2f)' % (eta, i)] = result.x
            PD.append(penaltydis(N, result.x))
            H.append(-Ham(result.x, i))
            L.append(E(result.x, i))
        f.write('eta={}\nL={}\nH={}\nDisToPhy={}\n\n'.format(eta, L, H, PD))
        print('eta=', eta)
        print('L=', L)
        print('H=', H)
        print('DisToPhy=', PD)
    f.close()
    np.savez('OUT/insulator_app/testins_g.npz', [g])


def main(x, y, y_label, model, gauge, device, eta=1000.):
    amount = len(x)
    # An example of g.
    g1 = maptophysical(N, np.zeros(N * 6 + 1), sample=True)[0]
    temp = np.array([2 * (-1) ** r / np.pi / (2 * r + 1) for r in range(N)])
    g2 = np.concatenate((np.zeros(N * 2 + 1), np.flip(temp), temp, np.zeros(N * 2)))

    if y_label == 'V':
        x_label = 'U'
        v, u = [y] * amount, x
    else:
        x_label = 'V'
        v, u = x, [y] * amount

    g = []
    PD, cdw, bdw, predicted, H, L = [], [], [], [], [], []
    init = g2
    for i in range(amount):
        # init = g2
        result = optimize.minimize(partial(E, V=v[i], U=u[i], eta=eta), init, method="BFGS", tol=1e-8)
        init = result.x
        g.append(init)
        PD.append(penaltydis(N, init))
        cdw.append(init[0])
        bdw.append(np.sqrt((init[3 * N] - init[3 * N + 1]) ** 2 + (init[N * 5] - init[N * 5 + 1]) ** 2) / 2)
        H.append(Ham(init, v[i], u[i]))
        L.append(E(init, v[i], u[i]))
        if gauge:
            input = np.delete(gauge_trans(N, init), [5 * N, 5 * N + 1], axis=0)
        else:
            input = init
        output = model(torch.from_numpy(input).float().unsqueeze(0).to(device))
        if output.shape[1] == 1:
            predicted.append(output.item())
        else:
            predicted.append((F.softmax(output, dim=1))[0, 1].item())
    g = np.array(g)
    if abs(y) < 50.:
        np.savez('datasets/StandardForm/init_ins_ex_N=' + str(N) + '_H_.npz', g)
    else:
        np.savez('datasets/StandardForm/init_ins_de_N=' + str(N) + '_g1.npz', g)
    print('L=', L)
    print('H=', H)
    print('cdw=', cdw)
    print('bdw=', bdw)
    print('penaltydis=', PD)
    print('penaltyANN=', predicted)
    plotfig(x, H, x_label, 'H')
    plotfig(x, cdw, x_label, 'cdw')
    plotfig(x, bdw, x_label, 'bdw')
    plotfig(x, PD, x_label, 'DisToPhy')
    plotfig(x, predicted, x_label, 'predicted')

    if abs(y) > 50.:
        targetmax, g_best = np.zeros(amount), np.zeros((amount, 6 * N + 1))
        Dsteps = np.arange(1, step=0.01)
        for i in Dsteps:
            g_ = Hamsolve(N, i)
            for estep in range(amount):
                target = -Ham(g_, v[estep], u[estep])
                if target > targetmax[estep]:
                    targetmax[estep] = target
                    g_best[estep] = g_
        np.savez('datasets/StandardForm/init_ins_se_N=' + str(N) + '_g1.npz', g_best)
        print('Target=', list(targetmax))
        plotfig(x, targetmax, x_label, 'Target')


def MEtest():
    gauge = True
    amount, rate = 10, 100
    weight = genweight(N, 0.3, gauge=gauge)
    penDr = []
    peng0r, penp0r, penMSEr, penMEr, penME0r = [0.] * amount, [0.] * amount, [0.] * amount, [0.] * amount, [0.] * amount
    penD, penMSE, penME, penME0 = [], [], [], []
    peng0, penp0 = [], []
    for i in range(amount):
        g_real, dv = create_physical_g(N, alpha=0.3, gauge=gauge)
        p_real = g2p(N, g_real)
        dv = dg2dp(N, dv)

        w, v = np.linalg.eigh(np.dot(dv, dv.T))
        projector = np.zeros((6 * N + 1, 6 * N + 1))
        for j in range(6 * N + 1):
            if w[j] > 1e-1:
                v = v[:, j:]
                projector = np.dot(v, v.T)
                break

        penDr.append(penaltydis(N, g_real))

        count = 0
        while count < rate:
            dg = np.random.normal(size=len(g_real), loc=0, scale=0.01) * weight
            # print(np.linalg.norm(dg))
            dp = dg2dp(N, dg)
            p_fake = p_real + dp
            if (np.max(p_fake) > 1.) or (np.min(p_fake) < 0.):
                continue
            g_fake = g_real + dg
            pdis = penaltydis(N, g_fake)
            penD.append(pdis)
            penMSE.append(penaltyMSE(dg, projector))
            penME.append(penaltyME(p_real, dp, projector))
            penME0.append(penaltyME0(p_real, dp))
            penp0.append(np.linalg.norm(dp))
            peng0.append(np.linalg.norm(dg))
            count += 1

    print(penDr)
    print(penD)
    print(penMSE)
    print(penME)
    print(penME0)
    print(peng0)
    print(penp0)

    plt.figure()
    plt.axis([-1e-3, max(penD), -1e-4, max(penME)])
    plt.scatter(penDr, penMSEr, s=50, c='k', marker='x')
    plt.scatter(penD, penMSE, s=5, c='r')
    plt.title('Penalty Comparison, gauge={}'.format(gauge))
    plt.xlabel('Penalty Diatance')
    plt.ylabel('Penalty MSE')
    plt.show()
    plt.close()

    plt.figure()
    plt.axis([-1e-3, max(penD), -1e-4, max(penME)])
    plt.scatter(penDr, penMEr, s=50, c='k', marker='x')
    plt.scatter(penD, penME, s=5, c='r')
    plt.title('Penalty Comparison, gauge={}'.format(gauge))
    plt.xlabel('Penalty Diatance')
    plt.ylabel('Penalty KLD')
    plt.show()
    plt.close()

    plt.figure()
    plt.axis([-1e-3, max(penD), -1e-3, max(penME0)])
    plt.scatter(penDr, penME0r, s=50, c='k', marker='x')
    plt.scatter(penD, penME0, s=5, c='y')
    plt.title('Penalty Comparison, gauge={}'.format(gauge))
    plt.xlabel('Penalty Diatance')
    plt.ylabel('Penalty KLD Without Projection')
    plt.show()
    plt.close()

    plt.figure()
    plt.axis([-1e-3, max(penD), -5e-3, max(peng0)])
    plt.scatter(penDr, peng0r, s=50, c='k', marker='x')
    plt.scatter(penD, peng0, s=5, c='g')
    plt.title('Penalty Comparison, gauge={}'.format(gauge))
    plt.xlabel('Penalty Diatance')
    plt.ylabel('Norm(dg)')
    plt.show()
    plt.close()

    plt.figure()
    plt.axis([-1e-3, max(penD), -5e-3, max(penp0)])
    plt.scatter(penDr, penp0r, s=50, c='k', marker='x')
    plt.scatter(penD, penp0, s=5, c='b')
    plt.title('Penalty Comparison, gauge={}'.format(gauge))
    plt.xlabel('Penalty Diatance')
    plt.ylabel('Norm(dp)')
    plt.show()
    plt.close()

    plt.figure()
    plt.axis([-1e-3, max(penME0), -1e-3, max(penME0)])
    plt.plot([-1e-3, np.max(penME0)], [-1e-3, np.max(penME0)], c='k')
    plt.scatter(penMEr, penME0r, s=50, c='b', marker='x')
    plt.scatter(penME, penME0, s=5, c='b')
    plt.title('Penalty KLD, gauge={}'.format(gauge))
    plt.xlabel('Projection')
    plt.ylabel('Without Projection')
    plt.show()
    plt.close()


if __name__ == "__main__":
    np.random.seed(6)  # random seed of numpy.random

    N = 20  # Cut off distance of incorporated correlations.

    MEtest()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Net = 'Deep_sig.21'
    model = torch.load('models/OneDInsulator/N={}/{}.pkl'.format(N, Net), map_location=device)
    model.eval()
    main(0.05 * np.arange(41), 0., 'U', model=model, gauge=True, device=device, eta=1000.)
    main(0.05 * np.arange(41), -100., 'U', model=model, gauge=True, device=device, eta=1000.)
