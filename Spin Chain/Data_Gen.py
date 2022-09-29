import os
from turtle import update
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import multiprocessing as mp
import random
import time
import numpy as np
import sys
from utils import Optimizer, Scheduler, Network, LoadDynamicalData_gauge, LoadVMCData
import warnings
import mkl
mkl.set_num_threads(1)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DefaultPath = '/home/sijing/ML/MPS/'

TYPE, cheat = 'regression', False
# 'Classification' or 'Regression'; whether introduce cheating strategy;
problem_type = 'regression'
#problem type in the files path!

# structural properties of the training set

update_times = int(sys.argv[4])     #  batch size & update times & epochs

pnt = 32

dataset_id = sys.argv[2]


# physical properties of the training set

L = 6
# The size of the unit cell, which determines the cut-off distance of our operators.
chi = int(sys.argv[1])
# visit make_MPS_data_regression to change the bond dimension
decay = float(sys.argv[3])
# weight decay that applys to the operator expectation velues



# range1 = 4
# range2 = 8
# torch.manual_seed(100)
# np.random.seed(100)
# random.seed(100)


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm1d):
        m.eval()

def main():
    print('Data Generating...')
    for i in range(update_times):
        dataset = LoadDynamicalData_gauge(rate=5, amount=50, chi = chi, L = L, amount_dv = -1, process_number_time=pnt, func=0, decay = decay) #amount = 50, pnt=64
        # dataset = LoadVMCData(rate=5, amount=20, process_number_time=16, func=0)

        if i == 0:
            dataset.saveBlock(
                DefaultPath + 'datasets/trainsets/' + problem_type + '/L=' + str(L) + '/dynamical_Haar_{}.npz'.format(dataset_id))
        else:
            temp = np.concatenate((np.load(
                DefaultPath + 'datasets/trainsets/' + problem_type + '/L=' + str(L) + '/dynamical_Haar_{}.npz'.format(dataset_id),
                allow_pickle=True)['arr_0'], dataset.data), axis=0)
            np.savez(
                DefaultPath + 'datasets/trainsets/' + problem_type + '/L=' + str(L) + '/dynamical_Haar_{}.npz'.format(dataset_id),
                temp)
    print('Data Generation Finished!')



if __name__ == "__main__":
    main()