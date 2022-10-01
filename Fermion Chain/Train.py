import os
import argparse

parser = argparse.ArgumentParser(description='Trainning ...')
parser.add_argument('--gpuid', dest='gpuid',
                    help="assign the GPU to use")
parser.add_argument('--scale', dest='scale', type=float,
                    help="scale of the labels of regression and cheat data")
parser.add_argument('--N', dest='N', type=int,
                    help="number of truncation")
parser.add_argument('--Net', dest='Net',
                    help="name of the model which will be trained")
parser.add_argument('--init', dest='init', default='',
                    help="initial model")
parser.add_argument('--traindataName', dest='traindataName',
                    help="name of the training data")
parser.add_argument('--valdataName', dest='valdataName',
                    help="name of the validation data")
parser.add_argument('--num_seed', dest='num_seed', type=int, default=0,
                    help="parameter of torch.manual_seed()")
parser.add_argument('--traingauge', dest='traingauge', type=int,
                    help="gauge transformation or not")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
from torch.utils.data import DataLoader
import time
# import visdom
from utils import *
from NN import Network, WEIGHT_INIT
import warnings, mkl

warnings.filterwarnings('ignore')
mkl.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scale, N = args.scale, args.N
Net, init, num_workers, num_seed = args.Net, args.init, 4, args.num_seed
traindataName, valdataName = tuple(args.traindataName.split(',')), tuple(args.valdataName.split(','))
gauge, batchsz, num_epoch = bool(args.traingauge), 512, 1000  # gauge & batch size & epochs
opt, sch = 'Adam', 'StepLR'  # optimizer & lr_scheduler
lr, wd, drop = 1e-3, 1e-6, (0., 0.)  # learning rate & weight decay & dropout
momentum, betas, freeze = 0., (0.9, 0.999), False  # momentum & betas & freeze_bn
gamma, ss, factor, ms = 0.5, 0.1, 200, (10,)  # gamma & factor & step size & milestones
trs, PEVar, k, s, pool = True, None, 1, None, False

traindatafiles, valdatafiles = [], []
for name in traindataName:
    traindatafiles.append('datasets/trainsets/OneDInsulator/N={}/{}.npz'.format(N, name))
for name in valdataName:
    valdatafiles.append('datasets/testsets/OneDInsulator/N={}/{}.npz'.format(N, name))

torch.manual_seed(num_seed)


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm1d):
        m.eval()


def main():
    NetType = Net[:Net.index('.')]
    traindata, valdata = None, None
    for file in traindatafiles:
        if traindata is None:
            traindata = np.load(file, allow_pickle=True)['arr_0']
        else:
            traindata = np.vstack((traindata, np.load(file, allow_pickle=True)['arr_0']))
    for file in valdatafiles:
        if valdata is None:
            valdata = np.load(file, allow_pickle=True)['arr_0']
        else:
            valdata = np.vstack((valdata, np.load(file, allow_pickle=True)['arr_0']))

    trainset = LoadInsulatorData_Reg(traindata, scale)
    valset = LoadInsulatorData_Reg(valdata, scale)
    criterion = nn.MSELoss().to(device)
    if gauge:
        input_size = 6 * N - 1
    else:
        input_size = 6 * N + 1

    try:
        model = torch.load('models/OneDInsulator/N={}/{}.pkl'.format(N, init), map_location=device)
    except (FileNotFoundError, IsADirectoryError):
        model = Network(NetType, input_size=input_size, drop=drop, trs=trs, PEVar=PEVar, k=k, s=s, pool=pool,
                        embedding_size=500, hidden_numbers=(1, 1, 1, 1), hidden_size=(100, ),
                        hidlay_numbers=(1, ), block_numbers=(1, 1, 1, 1), output_size=1).to(device)

    trainloader = DataLoader(trainset, batch_size=batchsz, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batchsz, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(len(trainset), len(trainloader), len(valset), len(valloader), '\n', model)
    f.write(
        'len(trainset)={}\tlen(trainloader)={}\tlen(valset)={}\tlen(valloader)={}\n{}\n'.format(
            len(trainset), len(trainloader), len(valset), len(valloader), model))
    f.flush()

    # # Visdom Setup
    # t = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    # session = 'MLQC_{}'.format(t)
    # vis = visdom.Visdom(env=session, port=8098)
    # vis.text('N={}\tNet={}'.format(N, Net), win='Nets', env=session)
    # losTypes = ["epoch_loss", "val_loss"]
    # losLines = [vis.line(Y=[0], X=[0], env=session, opts=dict(title=losn)) for losn in losTypes]
    # los_x = 0

    optimizer = Optimizer(opt, model, lr=lr, weight_decay=wd, momentum=momentum, betas=betas)
    scheduler = Scheduler(sch, optimizer, gamma=gamma, step_size=ss, factor=factor, milestones=ms)

    best = 1e5
    for epoch in range(num_epoch + 1):
        # train
        model.train()
        if freeze: model.apply(freeze_bn)
        running_loss = 0.
        for batchidx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(label)
        epoch_loss = running_loss / len(trainset)
        # validation
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            for batchidx, (x, label) in enumerate(valloader):
                x, label = x.to(device), label.to(device)
                logits = model(x)
                loss = criterion(logits, label)
                running_loss += loss.item() * len(label)
            val_loss = running_loss / len(valset)
        print(epoch, 'epoch loss:', epoch_loss, 'val loss:', val_loss)
        f.write('epoch={}\tepoch loss={}\tval loss={}\n'.format(epoch, epoch_loss, val_loss))

        # los_x += 1
        # for line, los in zip(losLines, np.array([epoch_loss, val_loss])):
        #     vis.line(Y=[los], X=[los_x], win=line, env=session, update="append")

        if sch == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        # save model
        if best > val_loss:
            best = val_loss
            torch.save(model, 'models/OneDInsulator/N={}/{}.pkl'.format(N, Net))
    print('Finished Training')
    f.write('Finished Training\n')


if __name__ == "__main__":
    f = open('models/OneDInsulator/N={}/{}.txt'.format(N, Net), 'w')
    f.write(
        'scale={}\tN={}\tNet={}\tinit={}\tnum_workers={}\tgauge={}\ttraindataName={}\tvaldataName={}\n'.format(
            scale, N, Net, init, num_workers, gauge, traindataName, valdataName))
    f.write(
        'num_seed={}\topt={}\tsch={}\tlr={}\twd={}\tmomentum={}\tbetas={}\tgamma={}\tss={}\tfactor={}\tms={}\n'.format(
            num_seed, opt, sch, lr, wd, momentum, betas, gamma, ss, factor, ms))
    f.write(
        'freeze={}\tbatchsz={}\tnum_epoch={}\tdrop={}\tweight_init={}\ttrs={}\tPEVar={}\tk={}\ts={}\tpool={}\n'.format(
            freeze, batchsz, num_epoch, drop, WEIGHT_INIT, trs, PEVar, k, s, pool))
    main()
    f.close()
