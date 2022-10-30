import torch
import torch.nn as nn
from .residual import ResidualModule

# fig.3
# 層の深さ(depthは4で固定の実装)
# 元は再帰構造
class HourglassModule(nn.Module):
    def __init__(self,num_features:int):
        super().__init__()
        nf=num_features
        self.num_features=num_features

        # num_modules=1
        # 入力サイズは64x64を想定
        # 最小サイズは 4x4


        self.depth=4

        self.branch_skip1=ResidualModule(nf,nf)
        self.branch_skip2=ResidualModule(nf,nf)
        self.branch_skip3=ResidualModule(nf,nf)
        self.branch_skip4=ResidualModule(nf,nf)

        self.down1=self.downblock(nf,nf)
        self.down2=self.downblock(nf,nf)
        self.down3=self.downblock(nf,nf)
        self.down4=self.downblock(nf,nf)
        
        self.middle=ResidualModule(nf,nf)

        self.up4=self.upblock(nf,nf)
        self.up3=self.upblock(nf,nf)
        self.up2=self.upblock(nf,nf)
        self.up1=self.upblock(nf,nf)



    def forward(self,x):

        skip1=self.branch_skip1(x)
        out=self.down1(x)

        skip2=self.branch_skip2(out)
        out=self.down2(out)

        skip3=self.branch_skip3(out)
        out=self.down3(out)

        skip4=self.branch_skip4(out)
        out=self.down4(out)

        # (4,4)のサイズになっているはず
        out=self.middle(out)

        out=self.up4(out)
        out= out+ skip4

        out=self.up3(out)
        out= out+skip3

        out=self.up2(out)
        out=out+skip2

        out=self.up1(out)
        out=out+skip1


        return out

    def downblock(self,in_features,out_features):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            ResidualModule(in_features,out_features)
        )

    def upblock(self,in_features,out_features):
        return nn.Sequential(
            ResidualModule(in_features,out_features),
            nn.Upsample(scale_factor=2,mode='nearest')
        )



def __test_hourglass_module():
    # TODO: 確認する
    m=HourglassModule(256)

    
    x=torch.randn(4,256,64,64)
    print("in:",x.size())
    out=m(x)
    print("out",out.size())

# __test_hourglass_module()