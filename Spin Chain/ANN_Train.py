import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import multiprocessing as mp
import random
import time
import numpy as np
from utils import Optimizer, Scheduler, LoadDynamicalData_gauge, LoadMPSData_Reg
from NN import *
import warnings
import mkl
mkl.set_num_threads(1)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DefaultPath = '/home/sijing/ML/MPS/'

TYPE, cheat = 'regression', False
# 'Classification' or 'Regression'; whether introduce cheating strategy; scale
problem_type = 'regression'
#problem type in the files path!

Net, no, dataset_id = 'Deep_sig', ['Mix_c02'], ['b0602', 'b0604', 'b0608', 'b0616', 'b0632', 'b0664', 'VMC0001', 'VMC0002']

print(Net, no)

batchsz, num_epoch = 64, 500      #  batch size & number of epochs

L = 6
# The size of the unit cell, which determines the cut-off distance of our operators.
scale = 1.5
# rescaling of the ANN training data labels.

opt, sch, freeze = 'Adam', 'StepLR', False    # optimizer & lr_scheduler & freeze_bn
lr, wd, drop = 1e-3, 1e-5, 0.1
momentum, betas = 0., (0.9, 0.999)        # momentum
gamma, ss = 0.5, 20                     # gamma & step size

# range1 = 4
# range2 = 8
# torch.manual_seed(100)
# np.random.seed(100)
# random.seed(100)


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm1d):
        m.eval()

def main():
    model = Network(Net, input_size=(9 * L - 6), embedding_size = 8 * (9 * L - 6), hidden_size = [2 * (9 * L - 6)], hidden_numbers=(4, 3, 2, 1),
                    block_numbers=(1, 1, 1, 1), drop=[drop]).to(device)
    
    # model = torch.load(
    #         DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}/{}_{}_dynamical.pkl'.format(TYPE, L, opt, Net, no[-1], Net, no[-1])).to(
    #         device)
    criterion = nn.MSELoss().to(device)


    print(model)
    print(model, file=f)
    print(dataset_id)
    print(dataset_id, file=f)

    epoch_loss_list = []
    val_loss_list = []

    # load validation set
    dataload_test = np.empty((0, 9 * L - 6 + 1))
    for id in dataset_id:
        # x = np.load(DefaultPath + 'datasets/testsets/' + problem_type + '/L=' + str(L) + '/dynamical_Haar_{}1.npz'.format(id), allow_pickle=True)['arr_0'][:2000]
        x = np.load(DefaultPath + 'datasets/trainsets/' + problem_type + '/L=' + str(L) + '/dynamical_Haar_{}.npz'.format(id), allow_pickle=True)['arr_0'][:3000]
        dataload_test = np.concatenate((dataload_test, x), axis=0)
    
    valset = LoadMPSData_Reg(dataload=dataload_test, scale=scale)
    valloader = DataLoader(valset, batch_size=batchsz, shuffle=False, num_workers=0)

    print(' size of the valset:', len(valset))
    print(' size of the valset:', len(valset), file=f)

    # load training set
    dataload = np.empty((0, 9 * L - 6 + 1))
    for id in dataset_id:
        x = np.load(DefaultPath + 'datasets/trainsets/' + problem_type + '/L=' + str(L) + '/dynamical_Haar_{}.npz'.format(id), allow_pickle=True)['arr_0'][3000:]
        dataload = np.concatenate((dataload, x), axis=0)

    dataset = LoadMPSData_Reg(dataload=dataload, scale=scale)
    dataloader = DataLoader(dataset, batch_size=batchsz, shuffle=True, num_workers=0)
    
    print(' size of the trainset:', len(dataset))
    print(' size of the trainset:', len(dataset), file=f)

    optimizer = Optimizer(opt, model, lr=lr, weight_decay=wd, momentum=momentum, betas=betas)
    scheduler = Scheduler(sch, optimizer, step_size=ss, gamma=gamma)

    best = 1e5

    if freeze: model.apply(freeze_bn)
    for epoch in range(num_epoch):
        # train
        model.train()
        running_loss = 0.

        for batchidx, (x, label) in enumerate(dataloader):
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, label)
            # print(' label:', label[0], ' output:', logits[0], ' loss:', loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(label)
        epoch_loss = running_loss / len(dataset)
        epoch_loss_list.append(epoch_loss)

        # validation
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            for batchidx, (x, label) in enumerate(valloader):
                x, label = x.to(device), label.to(device)
                logits = model(x)
                loss = criterion(logits, label)
                # print(' label:', label[0], ' output:', logits[0], ' loss:', loss)
                running_loss += loss.item() * len(label)
            val_loss = running_loss / len(valset)

        print(no[0], dataset_id, 'epoch:', epoch, 'epoch loss:', epoch_loss, 'val loss:', val_loss)
        f.write('dataset_id={}\epoch={}\tepoch loss={}\tval loss={}\n'.format(str(dataset_id), epoch, epoch_loss, val_loss))

        scheduler.step()
    
        val_loss_list.append(val_loss)

        # save model
        if best > val_loss:
            best = val_loss
            torch.save(model, DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}/{}_{}_dynamical.pkl'.format(TYPE, L, opt, Net, no[-1], Net, no[-1]))
        np.save(DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}/epoch_loss'.format(TYPE, L, opt, Net, no[-1]), epoch_loss_list)
        np.save(DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}/val_loss'.format(TYPE, L, opt, Net, no[-1]), val_loss_list)
    print('Finished Training')
    print('Finished Training', file=f)


if __name__ == "__main__":
    if not os.path.exists(DefaultPath + 'models/{}/L={}/Training Record'.format(TYPE, L)):
        os.mkdir(DefaultPath + 'models/{}/L={}/Training Record'.format(TYPE, L))
    if not os.path.exists(DefaultPath + 'models/{}/L={}/Training Record/{}'.format(TYPE, L, opt)):
        os.mkdir(DefaultPath + 'models/{}/L={}/Training Record/{}'.format(TYPE, L, opt))
    if not os.path.exists(DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}'.format(TYPE, L, opt, Net, no[-1])):
        os.mkdir(DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}'.format(TYPE, L, opt, Net, no[-1]))

    f = open(DefaultPath + 'models/{}/L={}/Training Record/{}/{}_{}/{}_{}.txt'.format(TYPE, L, opt, Net, no[-1], Net, no[-1]), 'w+')
    print(Net, no, file=f)
    t = time.time()
    main()
    print(time.time() - t)
    f.close()
