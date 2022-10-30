import torch
import torch.nn as nn

class ResidualModule(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels

        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels//2,kernel_size=(1,1))
        self.bn1=nn.BatchNorm2d(num_features=out_channels//2)

        self.conv2=nn.Conv2d(in_channels=out_channels//2,out_channels=out_channels//2,kernel_size=(3,3),padding=(1,1))
        self.bn2=nn.BatchNorm2d(num_features=out_channels//2)
        self.conv3=nn.Conv2d(in_channels=out_channels//2,out_channels=out_channels,kernel_size=(1,1))
        self.bn3=nn.BatchNorm2d(num_features=out_channels)


        if in_channels==out_channels:
            self.skip_layers=nn.Identity()
        else:
            self.skip_layers=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1)),
                nn.BatchNorm2d(num_features=out_channels),
            )

        self.relu=nn.ReLU(inplace=True)


    def forward(self,feature:torch.Tensor):
        identity=feature

        out=self.conv1(feature)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        out+=  self.skip_layers(identity)
        out=self.relu(out) # reluいらない？　hgのレポジトリではない？

        return out


def __test_in_out_same_size():
    b=ResidualModule(256,256)
    print(b)
    x=torch.randn(5,256,64,64)
    print(x.size())
    out=b(x)
    print(out.size())
def __test_in_out_diff_size():
    b=ResidualModule(256,128)
    print(b)
    x=torch.randn(5,256,64,64)
    print(x.size())
    out=b(x)
    print(out.size())

# __test_in_out_diff_size()
