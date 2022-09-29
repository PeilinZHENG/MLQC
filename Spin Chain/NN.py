import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings("ignore")
WEIGHT_INIT = True

def Network(Net, input_size, output_size=1, embedding_size=128, hidden_size=(256, 256, 256, 256),
            hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
            drop=(0., 0.), trs=True, PEVar=True, k=1, s=None, pool=False):
    if Net == 'TwoLayer':
        return TwoLayerModel(input_size, output_size, embedding_size, drop[0])
    elif Net == 'Naive':
        return NaiveModel(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Naive_sig':
        return NaiveModel_sig(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Naive_relu':
        return NaiveModel_relu(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Simple':
        return SimpleModel(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Simple_sig':
        return SimpleModel_sig(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Simple_relu':
        return SimpleModel_relu(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Median':
        return MedianModel(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Median_sig':
        return MedianModel_sig(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Median_relu':
        return MedianModel_relu(input_size, output_size, embedding_size, hidden_size[0], drop[0], trs)
    elif Net == 'Deep':
        return Model(input_size, output_size, embedding_size, hidden_size[0], hidden_numbers, block_numbers, drop[0],
                     trs)
    elif Net == 'Deep_sig':
        return Model_sig(input_size, output_size, embedding_size, hidden_size[0], hidden_numbers, block_numbers,
                         drop[0], trs)
    elif Net == 'Deep_relu':
        return Model_relu(input_size, output_size, embedding_size, hidden_size[0], hidden_numbers, block_numbers,
                          drop[0], trs)
    elif Net == 'Tran':
        return TranModel(input_size, output_size, hidden_numbers[0], embedding_size, hidden_size[0], block_numbers[0],
                          drop[0], PEVar, k, s)
    elif Net == 'Tran_sig':
        return TranModel_sig(input_size, output_size, hidden_numbers[0], embedding_size, hidden_size[0],
                              block_numbers[0], drop[0], PEVar, k, s)
    elif Net == 'Tran_relu':
        return TranModel_relu(input_size, output_size, hidden_numbers[0], embedding_size, hidden_size[0],
                               block_numbers[0], drop[0], PEVar, k, s)
    elif Net == 'TranS':
        return TranSModel(input_size, output_size, hidden_numbers[0], embedding_size, hidden_size[0], block_numbers[0],
                          drop[0], PEVar, pool)
    elif Net == 'TranS_sig':
        return TranSModel_sig(input_size, output_size, hidden_numbers[0], embedding_size, hidden_size[0],
                              block_numbers[0], drop[0], PEVar, pool)
    elif Net == 'TranS_relu':
        return TranSModel_relu(input_size, output_size, hidden_numbers[0], embedding_size, hidden_size[0],
                               block_numbers[0], drop[0], PEVar, pool)
    elif Net == 'Adv':
        return AdvModel(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                        block_numbers, drop, PEVar, k, s)
    elif Net == 'Adv_sig':
        return AdvModel_sig(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                            block_numbers, drop, PEVar, k, s)
    elif Net == 'Adv_relu':
        return AdvModel_relu(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                             block_numbers, drop, PEVar, k, s)
    elif Net == 'AdvS':
        return AdvSModel(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                        block_numbers, drop, PEVar, pool)
    elif Net == 'AdvS_sig':
        return AdvSModel_sig(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                            block_numbers, drop, PEVar, pool)
    elif Net == 'AdvS_relu':
        return AdvSModel_relu(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                             block_numbers, drop, PEVar, pool)
    else:
        raise NameError('Wrong Net Type')


class TwoLayerModel(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, drop=0.):
        super(TwoLayerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, output_size)
        self.dropout = nn.Dropout(drop)
        if WEIGHT_INIT:
            self.fc1.apply(self.weights_init)
            self.fc2.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


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


class NaiveModel_relu(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(NaiveModel_relu, self).__init__()
        self.naivemodel = NaiveModel(input_size, output_size, embedding_size, hidden_size, drop, trs)

    def forward(self, x):
        return F.relu(self.naivemodel(x))


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


class SimpleModel_relu(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(SimpleModel_relu, self).__init__()
        self.simplemodel = SimpleModel(input_size, output_size, embedding_size, hidden_size, drop, trs)

    def forward(self, x):
        return F.relu(self.simplemodel(x))


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


class MedianModel_relu(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20, drop=0., trs=True):
        super(MedianModel_relu, self).__init__()
        self.medianmodel = MedianModel(input_size, output_size, embedding_size, hidden_size, drop, trs)

    def forward(self, x):
        return F.relu(self.medianmodel(x))


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


class Model_relu(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=100, hidden_size=20,
                 hidden_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1), drop=0., trs=True):
        super(Model_relu, self).__init__()
        self.model = Model(input_size, output_size, embedding_size, hidden_size, hidden_numbers, block_numbers, drop, trs)

    def forward(self, x):
        return F.relu(self.model(x))


class TranModel(nn.Module):
    def __init__(self, input_size, output_size=1, nhead=8, embedding_size=8, hidden_size=256, block_numbers=1,
                 drop=0., PEVar=True, k=1, s=None):
        super(TranModel, self).__init__()
        d_model = nhead * embedding_size
        if input_size % 6 == 1:
            self.N = int((input_size - 1) / 6)
            self.gauge = False
        else:
            self.N = int((input_size + 1) / 6)
            self.gauge = True
        self.pre = nn.Linear(self.N + 1, d_model)
        self.dropout = nn.Dropout(drop)
        if PEVar is True:
            self.pe = nn.Parameter(torch.randn(input_size, 1, d_model))
        elif PEVar is False:
            position = torch.arange(input_size).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(input_size, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = None
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_size,
                                       dropout=drop, norm_first=False), num_layers=block_numbers)
        self.k, self.s = k, s
        if s is None:
            temp_size = int(d_model / k)
        else:
            temp_size = int((d_model - k) / s) + 1
        self.out = nn.Sequential(
            nn.LayerNorm(6 * temp_size),
            nn.Linear(6 * temp_size, output_size)
        )
        if WEIGHT_INIT:
            self.pre.apply(self.weights_init)
            self.transformer_encoder.apply(self.weights_init)
            self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        pad = torch.zeros((x.size(0), 1), device=x.device)
        if self.gauge:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N - 1, self.N - 1], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, pad, x5), dim=1)
        else:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N, self.N], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, x5), dim=1)
        x = self.pre(torch.stack((x0, x1, x2, x3, x4, x5), dim=0))
        # x = self.pre(x.unsqueeze(0).transpose(0, 2))
        if self.pe is not None:
            x += self.pe
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = F.avg_pool1d(x, kernel_size=self.k, stride=self.s)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class TranModel_sig(nn.Module):
    def __init__(self, input_size, output_size=1, nhead=8, embedding_size=8, hidden_size=256, block_numbers=1,
                 drop=0., PEVar=True, k=1, s=None):
        super(TranModel_sig, self).__init__()
        self.model = TranModel(input_size, output_size, nhead, embedding_size, hidden_size, block_numbers, drop, PEVar,
                                k, s)

    def forward(self, x):
        return F.sigmoid(self.model(x))


class TranModel_relu(nn.Module):
    def __init__(self, input_size, output_size=1, nhead=8, embedding_size=512, hidden_size=2048, block_numbers=1,
                 drop=0., PEVar=True, k=1, s=None):
        super(TranModel_relu, self).__init__()
        self.model = TranModel(input_size, output_size, nhead, embedding_size, hidden_size, block_numbers, drop, PEVar,
                                k, s)

    def forward(self, x):
        return F.relu(self.model(x))


class TranSModel(nn.Module):
    def __init__(self, input_size, output_size=1, nhead=8, embedding_size=8, hidden_size=256, block_numbers=1,
                 drop=0., PEVar=True, pool=False):
        super(TranSModel, self).__init__()
        d_model = nhead * embedding_size
        if input_size % 6 == 1:
            self.N = int((input_size - 1) / 6)
            self.gauge = False
        else:
            self.N = int((input_size + 1) / 6)
            self.gauge = True
        self.pre = nn.Linear(self.N + 1, d_model)
        self.dropout = nn.Dropout(drop)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if PEVar is True:
            self.pe = nn.Parameter(torch.randn(input_size + 1, 1, d_model))
        elif PEVar is False:
            position = torch.arange(input_size + 1).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(input_size + 1, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = None
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_size,
                                       dropout=drop, norm_first=True), num_layers=block_numbers)
        self.pool = pool
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_size)
        )
        if WEIGHT_INIT:
            self.pre.apply(self.weights_init)
            self.transformer_encoder.apply(self.weights_init)
            self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        pad = torch.zeros((x.size(0), 1), device=x.device)
        if self.gauge:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N - 1, self.N - 1], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, pad, x5), dim=1)
        else:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N, self.N], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, x5), dim=1)
        x = self.pre(torch.stack((x0, x1, x2, x3, x4, x5), dim=0))
        # x = self.pre(x.unsqueeze(0).transpose(0, 2))
        x = torch.cat((self.cls_token.repeat(1, x.size(1), 1), x), dim=0)
        if self.pe is not None:
            x += self.pe
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = x.mean(dim=1) if self.pool else x[:, 0]
        x = self.out(x)
        return x


class TranSModel_sig(nn.Module):
    def __init__(self, input_size, output_size=1, nhead=8, embedding_size=8, hidden_size=256, block_numbers=1,
                 drop=0., PEVar=True, pool=False):
        super(TranSModel_sig, self).__init__()
        self.model = TranSModel(input_size, output_size, nhead, embedding_size, hidden_size, block_numbers, drop, PEVar,
                                pool)

    def forward(self, x):
        return F.sigmoid(self.model(x))


class TranSModel_relu(nn.Module):
    def __init__(self, input_size, output_size=1, nhead=8, embedding_size=512, hidden_size=2048, block_numbers=1,
                 drop=0., PEVar=True, pool=False):
        super(TranSModel_relu, self).__init__()
        self.model = TranSModel(input_size, output_size, nhead, embedding_size, hidden_size, block_numbers, drop, PEVar,
                                pool)

    def forward(self, x):
        return F.relu(self.model(x))


class SAResidual(nn.Module):
    def __init__(self, inchannel, outchannel, nhead, nmlp=1, hidden_size=0, drop=(0., 0.)):
        super(SAResidual, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=inchannel, num_heads=nhead, dropout=drop[1])
        self.norm1 = nn.LayerNorm(inchannel)
        self.dropout = nn.Dropout(drop[0])
        if nmlp == 1:
            self.fc = nn.Sequential(
                nn.Linear(inchannel, outchannel),
                nn.ReLU(inplace=True),
                nn.Dropout(drop[0]),
            )
        else:
            mlp = []
            for _ in range(nmlp - 2):
                mlp.append(nn.Linear(hidden_size, hidden_size))
                mlp.append(nn.ReLU(inplace=True))
                mlp.append(nn.Dropout(drop[0]))
            self.fc = nn.Sequential(
                nn.Linear(inchannel, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(drop[0]),
                *mlp,
                nn.Linear(hidden_size, outchannel),
                nn.Dropout(drop[0])
            )
        self.norm2 = nn.LayerNorm(inchannel)
        if inchannel == outchannel:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(inchannel, outchannel)
            if WEIGHT_INIT: self.shortcut.apply(self.weights_init)
        if WEIGHT_INIT:
            self.self_attn.apply(self.weights_init)
            self.fc.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_ = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_, x_, x_, need_weights=False)[0])
        out = self.fc(self.norm2(x))
        residual = x if self.shortcut is None else self.shortcut(x)
        out = out + residual
        return out


class AdvModel(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=64, hidden_size=(256, 256, 256, 256),
                 hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
                 drop=(0., 0.), PEVar=True, k=1, s=None):
        '''
            hidden_numbers: nhead
            hidden_numbers * embedding_size: input or output size in each block
            hidden_size: size of mlp layers in each block
            hidlay_numbers: number of mlp layers in each block
            block_numbers: number of repeated blocks
        '''
        super(AdvModel, self).__init__()
        d_model = int(hidden_numbers[0] * embedding_size)
        if input_size % 6 == 1:
            self.N = int((input_size - 1) / 6)
            self.gauge = False
        else:
            self.N = int((input_size + 1) / 6)
            self.gauge = True
        self.pre = nn.Linear(self.N + 1, d_model)
        self.dropout = nn.Dropout(drop[0])
        if PEVar is True:
            self.pe = nn.Parameter(torch.randn(input_size, 1, d_model))
        elif PEVar is False:
            position = torch.arange(input_size).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(input_size, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = None
        self.layer1 = self._make_layer(d_model, int(hidden_numbers[1] * embedding_size),
                                       hidlay_numbers[0], hidden_numbers[0], hidden_size[0], block_numbers[0], drop)
        self.layer2 = self._make_layer(int(hidden_numbers[1] * embedding_size), int(hidden_numbers[2] * embedding_size),
                                       hidlay_numbers[1], hidden_numbers[1], hidden_size[1], block_numbers[1], drop)
        self.layer3 = self._make_layer(int(hidden_numbers[2] * embedding_size), int(hidden_numbers[3] * embedding_size),
                                       hidlay_numbers[2], hidden_numbers[2], hidden_size[2], block_numbers[2], drop)
        self.layer4 = self._make_layer(int(hidden_numbers[3] * embedding_size), embedding_size,
                                       hidlay_numbers[3], hidden_numbers[3], hidden_size[3], block_numbers[3], drop)
        self.k, self.s = k, s
        if s is None:
            temp_size = int(embedding_size / k)
        else:
            temp_size = int((embedding_size - k) / s) + 1
        self.out = nn.Sequential(
            nn.LayerNorm(6 * temp_size),
            nn.Linear(6 * temp_size, output_size)
        )
        if WEIGHT_INIT:
            self.pre.apply(self.weights_init)
            self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _make_layer(self, inchannel, outchannel, nmlp, nhead, hidden_size, block_num, drop):
        layers = []
        for _ in range(block_num - 1):
            layers.append(SAResidual(inchannel, inchannel, nhead, nmlp=nmlp, hidden_size=hidden_size, drop=drop))
        layers.append(SAResidual(inchannel, outchannel, nhead, nmlp=nmlp, hidden_size=hidden_size, drop=drop))
        return nn.Sequential(*layers)

    def forward(self, x):
        pad = torch.zeros((x.size(0), 1), device=x.device)
        if self.gauge:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N - 1, self.N - 1], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, pad, x5), dim=1)
        else:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N, self.N], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, x5), dim=1)
        x = self.pre(torch.stack((x0, x1, x2, x3, x4, x5), dim=0))
        # x = self.pre(x.unsqueeze(0).transpose(0, 2))
        if self.pe is not None:
            x += self.pe
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(0, 1)
        x = F.avg_pool1d(x, kernel_size=self.k, stride=self.s)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class AdvModel_sig(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=64, hidden_size=(256, 256, 256, 256),
                 hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
                 drop=(0., 0.), PEVar=True, k=1, s=None):
        super(AdvModel_sig, self).__init__()
        self.model = AdvModel(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                              block_numbers, drop, PEVar, k, s)

    def forward(self, x):
        return F.sigmoid(self.model(x))


class AdvModel_relu(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=64, hidden_size=(256, 256, 256, 256),
                 hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
                 drop=(0., 0.), PEVar=True, k=1, s=None):
        super(AdvModel_relu, self).__init__()
        self.model = AdvModel(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                              block_numbers, drop, PEVar, k, s)

    def forward(self, x):
        return F.relu(self.model(x))


class AdvSModel(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=64, hidden_size=(256, 256, 256, 256),
                 hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
                 drop=(0., 0.), PEVar=True, pool=False):
        '''
            hidden_numbers: nhead
            hidden_numbers * embedding_size: input or output size in each block
            hidden_size: size of mlp layers in each block
            hidlay_numbers: number of mlp layers in each block
            block_numbers: number of repeated blocks
        '''
        super(AdvSModel, self).__init__()
        d_model = int(hidden_numbers[0] * embedding_size)
        if input_size % 6 == 1:
            self.N = int((input_size - 1) / 6)
            self.gauge = False
        else:
            self.N = int((input_size + 1) / 6)
            self.gauge = True
        self.pre = nn.Linear(self.N + 1, d_model)
        self.dropout = nn.Dropout(drop[0])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        if PEVar is True:
            self.pe = nn.Parameter(torch.randn(input_size + 1, 1, d_model))
        elif PEVar is False:
            position = torch.arange(input_size + 1).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(input_size+ 1, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = None
        self.layer1 = self._make_layer(d_model, int(hidden_numbers[1] * embedding_size),
                                       hidlay_numbers[0], hidden_numbers[0], hidden_size[0], block_numbers[0], drop)
        self.layer2 = self._make_layer(int(hidden_numbers[1] * embedding_size), int(hidden_numbers[2] * embedding_size),
                                       hidlay_numbers[1], hidden_numbers[1], hidden_size[1], block_numbers[1], drop)
        self.layer3 = self._make_layer(int(hidden_numbers[2] * embedding_size), int(hidden_numbers[3] * embedding_size),
                                       hidlay_numbers[2], hidden_numbers[2], hidden_size[2], block_numbers[2], drop)
        self.layer4 = self._make_layer(int(hidden_numbers[3] * embedding_size), embedding_size,
                                       hidlay_numbers[3], hidden_numbers[3], hidden_size[3], block_numbers[3], drop)
        self.pool = pool
        self.out = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, output_size)
        )
        if WEIGHT_INIT:
            self.pre.apply(self.weights_init)
            self.out.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _make_layer(self, inchannel, outchannel, nmlp, nhead, hidden_size, block_num, drop):
        layers = []
        for _ in range(block_num - 1):
            layers.append(SAResidual(inchannel, inchannel, nhead, nmlp=nmlp, hidden_size=hidden_size, drop=drop))
        layers.append(SAResidual(inchannel, outchannel, nhead, nmlp=nmlp, hidden_size=hidden_size, drop=drop))
        return nn.Sequential(*layers)

    def forward(self, x):
        pad = torch.zeros((x.size(0), 1), device=x.device)
        if self.gauge:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N - 1, self.N - 1], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, pad, x5), dim=1)
        else:
            x0, x1, x2, x3, x4, x5 = x.split([self.N + 1, self.N, self.N, self.N, self.N, self.N], dim=1)
            x1 = torch.cat((pad, x1), dim=1)
            x2 = torch.cat((pad, x2.flip(dims=(1,))), dim=1)
            x3 = torch.cat((pad, x3), dim=1)
            x4 = torch.cat((pad, x4.flip(dims=(1,))), dim=1)
            x5 = torch.cat((pad, x5), dim=1)
        x = self.pre(torch.stack((x0, x1, x2, x3, x4, x5), dim=0))
        # x = self.pre(x.unsqueeze(0).transpose(0, 2))
        x = torch.cat((self.cls_token.repeat(1, x.size(1), 1), x), dim=0)
        if self.pe is not None:
            x += self.pe
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(0, 1)
        x = x.mean(dim=1) if self.pool else x[:, 0]
        x = self.out(x)
        return x


class AdvSModel_sig(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=64, hidden_size=(256, 256, 256, 256),
                 hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
                 drop=(0., 0.), PEVar=True, pool=False):
        super(AdvSModel_sig, self).__init__()
        self.model = AdvSModel(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                               block_numbers, drop, PEVar, pool)

    def forward(self, x):
        return F.sigmoid(self.model(x))


class AdvSModel_relu(nn.Module):
    def __init__(self, input_size, output_size=1, embedding_size=64, hidden_size=(256, 256, 256, 256),
                 hidden_numbers=(1, 1, 1, 1), hidlay_numbers=(1, 1, 1, 1), block_numbers=(1, 1, 1, 1),
                 drop=(0., 0.), PEVar=True, pool=False):
        super(AdvSModel_relu, self).__init__()
        self.model = AdvSModel(input_size, output_size, embedding_size, hidden_size, hidden_numbers, hidlay_numbers,
                               block_numbers, drop, PEVar, pool)

    def forward(self, x):
        return F.relu(self.model(x))