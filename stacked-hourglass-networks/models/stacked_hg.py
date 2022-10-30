import torch
import torch.nn as nn
from .hg_module import HourglassModule
from .residual import ResidualModule


class StackedHourglassNetwork(nn.Module):
    def __init__(self,heatmap_channels,in_channels=3,num_features=256,num_stacks=8):
        super().__init__()
        self.in_channels=in_channels
        self.num_features=num_features
        self.num_stacks=num_stacks

        # jointの数。heatmapのチャネル数になる
        self.heatmap_channels=heatmap_channels

        # conv(ks=7x7,stride=2) =>residual module => a round of max plloing で256から64へ小さくする
        self.first_layers=nn.Sequential(
            #  maxppl2dまでで  (3,256,256)=>(128,64,64)　になる
            nn.Conv2d(in_channels,64,kernel_size=(7,7),stride=(2,2),padding=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualModule(64,128),

            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),

            ResidualModule(128,128),

            ResidualModule(128,num_features)

        )

        # TODO: stacked hourglass
        self.layers=nn.ModuleList(
            nn.Sequential(
                HourglassModule(num_features),

                nn.Conv2d(in_channels=num_features,out_channels=num_features,kernel_size=(1,1),stride=(1,1),padding=0),
                nn.BatchNorm2d(num_features), 
                nn.ReLU(inplace=True),
                

            ) for i in range(num_stacks)
        )

        self.to_heatmap=nn.ModuleList(
                nn.Conv2d(in_channels=num_features,out_channels=heatmap_channels,kernel_size=(1,1),stride=(1,1),padding=0) for i in range(num_stacks)
        )

        # layersのあとに適用
        self.skip2_layers=nn.ModuleList(
                # resmoudle???
                # ReLUは？？
                nn.Conv2d(in_channels=num_features,out_channels=num_features,kernel_size=(1,1),stride=(1,1),padding=0) for i in range(num_stacks)
        )

        # 
        self.skip3_layers=nn.ModuleList(
                # resmoudle???
                # ReLUは？？
                nn.Conv2d(in_channels=heatmap_channels,out_channels=num_features,kernel_size=(1,1),stride=(1,1),padding=0) for i in range(num_stacks)
        )

    def forward(self,x):
        out=self.first_layers(x)

        # 各stackごとにheatmapを保存する
        heatmaps=[]
        for i in range(self.num_stacks):
            skip1=out
            out=self.layers[i](out)
            skip2=out

            heatmap=self.to_heatmap[i](out)
            heatmaps.append(heatmap)

            # 最終層ではskip connectionの処理は必要ない
            if i< self.num_stacks:
                skip2=self.skip2_layers[i](out)
                skip3=self.skip3_layers[i](heatmap)
                out=skip1+skip2+skip3
        
        # list => tensor
        
        # list=>tensorにして、次元をbatch_sizeとstackのdimの位置変更 
        heatmaps=torch.stack(heatmaps).permute((1,0,2,3,4))
        return heatmaps



def __test_stacked_hourglass_network():
    net=StackedHourglassNetwork(heatmap_channels=16)

    x=torch.randn(2,3,256,256)
    print("in:",x.size())
    heatmaps=net(x)
    print("out")
    print("size",heatmaps.size()) # スタック数8   (batch_size,stack_size,num_joints,64,64)のはず


# __test_stacked_hourglass_network()