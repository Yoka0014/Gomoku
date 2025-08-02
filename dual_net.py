import torch.nn as nn

class DualNet(nn.Module):
    """
    局面sを入力として, 状態価値V(s)と, 次元数が行動の数と一致するベクトルpを出力するデュアルネットワーク.
    すなわち, 9路番であれば, pは9x9=81次元のベクトルとなる.

    ベクトルpが意味する値は, DualNetの用途によって以下の2種類がある.
    
    1. DQNにおいては, pをアドバンテージA(s, a)とみなす．
       アドバンテージとは, 状態価値と行動価値の差分であり, 行動価値はQ(s, a) = V(s) + A(s, a)と表される．

    2. PV-MCTSにおける探索方策として用いる際は, pを次の手の確率分布P(s, a)とみなす．
    """
    NUM_LAYERS = 3

    def __init__(self, board_size):
        super(DualNet, self).__init__()

        kernel_size = 5
        self.conv_0 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn_0 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.hidden_layers = nn.ModuleList()

        for _ in range(self.NUM_LAYERS):
            self.hidden_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2, bias=False))
            self.hidden_layers.append(nn.BatchNorm2d(128))
            self.hidden_layers.append(nn.ReLU(inplace=True))

        # action head
        self.action_conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.action_bn = nn.BatchNorm2d(128)
        self.action_head_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, bias=False)
        self.action_head_bn = nn.BatchNorm2d(1)
        self.action_head = nn.Flatten()

        # value head
        self.value_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_flatten = nn.Flatten()
        self.value_linear = nn.Linear(board_size**2, 256)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = self.relu(x)

        for layer in self.hidden_layers:
            x = layer(x)

        # action head
        p = self.action_conv(x)
        p = self.action_bn(p)
        p = self.relu(p)
        p = self.action_head_conv(p)
        p = self.action_head_bn(p)
        p = self.action_head(p)

        # value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.value_flatten(v)
        v = self.relu(v)
        v = self.value_linear(v)
        v = self.relu(v)
        v = self.value_head(v)

        return p, v

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)