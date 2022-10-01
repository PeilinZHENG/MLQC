import multiprocessing as mp
from testins import *
import time
import warnings, mkl
import argparse
warnings.filterwarnings('ignore')
mkl.set_num_threads(1)


parser = argparse.ArgumentParser(description='Trainning ...')
parser.add_argument('--pattern', dest='pattern',
                    help="train or test")
parser.add_argument('--dataset_id', dest='dataset_id',
                    help="id of datasets")
parser.add_argument('--dataN', dest='dataN', type=int,
                    help="number of truncation")
parser.add_argument('--amount', dest='amount', type=int,
                    help="amount of data")
parser.add_argument('--processors', dest='processors', type=int,
                    help="number of processors")
parser.add_argument('--decay', dest='decay', type=float,
                    help="decay in label")
parser.add_argument('--rate', dest='rate', type=int,
                    help="ratio of real and fake")
parser.add_argument('--gauge', dest='gauge', type=int,
                    help="gauge transformation or not")

args = parser.parse_args()

DefaultPath = ''
pattern, dataset_id = args.pattern, args.dataset_id
amount, N, processors, rate = args.amount, args.dataN, args.processors, args.rate
decay, gauge = args.decay, bool(args.gauge)
weight = genweight(N, decay, gauge=gauge)
# random.seed(100)               # random seed of random
# np.random.seed(10)            # random seed of numpy.random


def genData(rate, id, N, b, weight, gauge=False):
    np.random.seed(int(time.time() * 1e16) % (2 ** 31 - 1))
    g = []
    for i in range(amount):
        if i == int(amount/2):
            weight = np.ones(len(weight))
        if id[0] == 'B':
            g_real, dv = create_BDW_g(N, alpha=b, gauge=gauge)
        elif id[0] == 'C':
            g_real, dv = create_CDW_g(N, alpha=b, gauge=gauge)
        else:
            g_real, dv = create_physical_g(N, alpha=b, gauge=gauge)
        w, v = np.linalg.eigh(np.dot(dv, dv.T))
        for j in range(6 * N + 1):
            if w[j] > 1e-3:
                v = v[:, j:]
                break
        projector = np.dot(v, v.T)
        p_real = g2p(N, g_real)
        # print('%d, real, %.8f, %.8f' % (rate, g_real[0], penaltydis(N, g_real)))
        g.append(np.concatenate((g_real, [0.]), axis=0))

        count = 0
        while count < rate:
            dg = np.random.normal(size=len(g_real), loc=0, scale=0.01) * weight
            dp = dg2dp(N, dg)
            p_fake = p_real + dp
            if (np.max(p_fake) > 1.) or (np.min(p_fake) < 0.):
                continue
            g_fake = g_real + dg
            # print('%d, fake, %.8f, %.8f' % (rate, g_fake[0], penaltydis(N, g_fake)))
            g.append(np.concatenate((g_fake, [penaltyME(p_real, dp, projector)]), axis=0))
            count += 1
    return g


if __name__ == "__main__":
    f = open(DefaultPath + 'datasets/{}sets/OneDInsulator/N={}/ME_{}.txt'.format(pattern, N, dataset_id), 'w')
    f.write('pattern={}\tdataset_id={}\tamount={}\tN={}\tprocessors={}\trate={}\tdecay={}\tgauge={}\n'.format(pattern, dataset_id, amount, N, processors, rate, decay, gauge))

    t = time.time()
    mp.set_start_method('fork', force=True) # For Windows, comment out this line
    g, pool = [], mp.Pool(processes=processors)
    res = pool.imap(partial(genData, id=dataset_id, N=N, b=decay, weight=weight, gauge=gauge), processors * [rate])
    for p in res:
        g += p
    pool.close()
    pool.join()
    t = time.time() - t
    f.write('Total amount={}\tTime={}\n'.format(len(g), t))
    print(len(g), t)
    np.savez(DefaultPath + 'datasets/{}sets/OneDInsulator/N={}/ME_{}.npz'.format(pattern, N, dataset_id), g)
    g = np.array(g)[:, -1]
    f.write('max={}\tmin={}\tmean={}\tstd={}'.format(max(g), min(g), np.mean(g), np.std(g)))
    print(max(g), min(g), np.mean(g), np.std(g))
    f.close()




