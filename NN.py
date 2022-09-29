import torch, math
from torch import nn, Tensor
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

WEIGHT_INIT = True


def Network(Net, input_size, output_size=1, embedding_size=128, hidden_size=(256, 256, 256, 256),
            hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
            drop=(0., 0.), trs=True, PEVar=True, k=1, s=None, pool=False):
    if Net == 'Naive':
        return NaiveModel(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Naive_sig':
        return NaiveModel_sig(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Simple':
        return SimpleModel(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Simple_sig':
        return SimpleModel_sig(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Median':
        return MedianModel(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Median_sig':
        return MedianModel_sig(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Deep':
        return Model(input_size, output_size, embedding_size, hidden_size[0], hidden_numbers, block_numbers, drop[0],
                     trs)
    elif Net == 'Deep_sig':
        return Model_sig(input_size, output_size, embedding_size, hidden_size[0], hidden_numbers, block_numbers,
                         drop[0], trs)
    else:
        raise NameError('Wrong Net Type')


class NaiveModel(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(NaiveModel, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size, track_running_stats=trs)
        self.fc2 = nn.Linear(embedding_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop)
        if WEIGHT_INIT:
            self.fc1.apply(self.weights_init)
            self.fc2.apply(self.weights_init)
            self.fc3.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


class NaiveModel_sig(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(NaiveModel_sig, self).__init__()
        self.naivemodel = NaiveModel(input_size, output_size, embedding_size, hidden_size, drop, trs)

    def forward(self, x):
        return F.sigmoid(self.naivemodel(x))


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size, track_running_stats=trs)
        self.fc2 = nn.Linear(embedding_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop)
        if WEIGHT_INIT:
            self.fc1.apply(self.weights_init)
            self.fc2.apply(self.weights_init)
            self.fc3.apply(self.weights_init)
            self.fc4.apply(self.weights_init)
            self.fc5.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(x + self.bn4(self.fc4(self.dropout(F.relu(self.bn3(self.fc3(x))))))))
        return self.fc5(x)


class SimpleModel_sig(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(SimpleModel_sig, self).__init__()
        self.simplemodel = SimpleModel(input_size, output_size, embedding_size, hidden_size, drop, trs)

    def forward(self, x):
        return F.sigmoid(self.simplemodel(x))


class MedianModel(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(MedianModel, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size, track_running_stats=trs)
        self.fc2 = nn.Linear(embedding_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size, track_running_stats=trs)
        self.fc7 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop)
        if WEIGHT_INIT:
            self.fc1.apply(self.weights_init)
            self.fc2.apply(self.weights_init)
            self.fc3.apply(self.weights_init)
            self.fc4.apply(self.weights_init)
            self.fc5.apply(self.weights_init)
            self.fc6.apply(self.weights_init)
            self.fc7.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(x + self.bn4(self.fc4(self.dropout(F.relu(self.bn3(self.fc3(x))))))))
        x = self.dropout(F.relu(x + self.bn6(self.fc6(self.dropout(F.relu(self.bn5(self.fc5(x))))))))
        return self.fc7(x)


class MedianModel_sig(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(MedianModel_sig, self).__init__()
        self.medianmodel = MedianModel(input_size, output_size, embedding_size, hidden_size, drop, trs)

    def forward(self, x):
        return F.sigmoid(self.medianmodel(x))


class Residual(nn.Module):
    def __init__(self, inchannel, outchannel, shortcut=None, drop=0., trs=True):
        super(Residual, self).__init__()
        self.fc3 = nn.Linear(inchannel, outchannel)
        self.fc4 = nn.Linear(outchannel, outchannel)
        self.res_block = nn.Sequential(
            self.fc3,
            nn.BatchNorm1d(outchannel, track_running_stats=trs),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            self.fc4,
            nn.BatchNorm1d(outchannel, track_running_stats=trs)
        )
        self.shortcut = shortcut
        if WEIGHT_INIT:
            self.fc3.apply(self.weights_init)
            self.fc4.apply(self.weights_init)
        # self.dropout = nn.Dropout(drop)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.res_block(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        out = F.relu(out)
        # out = self.dropout(out)
        return out


class Model(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, hidden_numbers=(1, 1, 1, 1),
                 block_numbers=(1, 1, 1, 1), drop=0., trs=True):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, int(hidden_numbers[0] * hidden_size))
        self.pre = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(embedding_size, track_running_stats=trs),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            self.fc2,
            nn.BatchNorm1d(int(hidden_numbers[0] * hidden_size), track_running_stats=trs),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.layer1 = self._make_layer(int(hidden_numbers[0] * hidden_size), int(hidden_numbers[1] * hidden_size),
                                       block_numbers[0], drop, trs=trs)
        self.layer2 = self._make_layer(int(hidden_numbers[1] * hidden_size), int(hidden_numbers[2] * hidden_size),
                                       block_numbers[1], drop, trs=trs)
        self.layer3 = self._make_layer(int(hidden_numbers[2] * hidden_size), int(hidden_numbers[3] * hidden_size),
                                       block_numbers[2], drop, trs=trs)
        self.layer4 = self._make_layer(int(hidden_numbers[3] * hidden_size), hidden_size, block_numbers[3], drop, trs=trs)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.out = nn.Sequential(
            nn.Dropout(drop),
            self.fc5
        )
        if WEIGHT_INIT:
            self.fc1.apply(self.weights_init)
            self.fc2.apply(self.weights_init)
            self.fc5.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _make_layer(self, inchannel, outchannel, block_num, drop, trs):
        if inchannel == outchannel:
            shortcut = None
        else:
            self.fc = nn.Linear(inchannel, outchannel)
            if WEIGHT_INIT: self.fc.apply(self.weights_init)
            shortcut = nn.Sequential(
                self.fc,
                nn.BatchNorm1d(outchannel, track_running_stats=trs)
            )
        layers = []
        layers.append(Residual(inchannel, outchannel, shortcut, drop=drop, trs=trs))
        for i in range(1, block_num):
            layers.append(Residual(outchannel, outchannel, drop=drop, trs=trs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)
        return x


class Model_sig(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20,
                 hidden_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1), drop=0., trs=True):
        super(Model_sig, self).__init__()
        self.model = Model(input_size, output_size, embedding_size, hidden_size, hidden_numbers, block_numbers, drop, trs)

    def forward(self, x):
        return F.sigmoid(self.model(x))


