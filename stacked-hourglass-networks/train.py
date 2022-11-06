import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from models.stacked_hg import StackedHourglassNetwork
from datasets.mpii import MPIIDataset
from utils import collate_fn,CustomTransforms,generate_batched_heatmaps


def get_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('--lr',type=float,default=2.5e-4,help='learning rate')
    parser.add_argument('--lr-decay',type=float,default=0.0,help='learning rate decay')
    parser.add_argument('--momentum',type=float,default=0.0,help='momentum')
    parser.add_argument('--weight-decay',type=float,default=0.0,help='weight decay')

    # criterion MSE
    # optimizer: rmsprop  ~ sgd | nag | adadelta

    # threshold : ???
    parser.add_argument('--num-epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('--batch-size',type=int,default=6,help='batch size')
    parser.add_argument('--num-workers',type=int,default=8,help='number of workers')

    
    parser.add_argument('--dataset',type=str,default='mpii',help='mpii')
    parser.add_argument('--data-root',type=str,default='./data/MPII',help='dataset root directory path')
    parser.add_argument('--seed',type=int,default=0,help='seed')
    parser.add_argument('--result-dir',type=str,default="../results",help='result directory')
    
    # gpu
    parser.add_argument('--cudnn-benchmark', action='store_true',help='cudnn benchmark')
    parser.add_argument('--amp', action='store_true',help='enable amp')


    # TODO: netType 'hg' | 'hg-stacked' が何か？
    # TODO: -task 'post' | 'post-int' 何？

    parser.add_argument('--checkpoint-path',type=str,default='../results/checkpoint.pth')
    parser.add_argument('--checkpoint-interval',type=int,default=100)


    # stack size 8
    # nModules: 1
    # n feats :256
    return parser.parse_args()


if __name__=="__main__":
    args=get_args()
    print(args)

    os.makedirs(args.result_dir,exist_ok=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: datasetによって変える
    num_joints=16 
    num_stacks=8
    heatmap_res=64

    shuffle=True

    # seed
    torch.manual_seed(args.seed)

    # スピードアップ
    torch.backends.cudnn.benchmark=args.cudnn_benchmark
    
    # model
    model=StackedHourglassNetwork(heatmap_channels=num_joints)
    model=model.to(device)

    # criterion
    criterion=nn.MSELoss()

    # optimizer
    optimizer=optim.RMSprop(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)

    my_transforms=CustomTransforms()
    #  dsatset
    dataset=MPIIDataset(args.data_root, os.path.join(args.data_root,'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1'),transforms=my_transforms)
    # drop last
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers,drop_last=True,collate_fn=collate_fn)


    # lr_scheduler=None

    scaler=GradScaler()
    iteration=1
    for epoch in range(1,args.num_epochs):
        for images,annotations in tqdm(dataloader):
            # TODO: heatmap作成はdataset側でやる datasetが返すのを image, gt_heatmapにする
            gt_heatmaps=generate_batched_heatmaps(num_joints,num_stacks,annotations,heatmap_res)

            # size : (batch_size,3,256,256)
            images=images.to(device)
            gt_heatmaps=gt_heatmaps.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type,enabled=args.amp):
                output=model(images)
                # (batch_size, stack_size, channel, height, width) だけど大丈夫？？？ stack_size増えてるけど
                loss=criterion(output,gt_heatmaps)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # TODO: 定期的にlossを表示するようにする or tensorboardに保存

            if iteration% args.checkpoint_interval == 0:
                print(iteration,args.checkpoint_interval)
                print("save checkpoint")
                # checkpoint保存
                torch.save({
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'iteration':iteration,
                    'epoch':epoch,
                    "gradscaler":scaler.state_dict()
                },args.checkpoint_path)


            iteration+=1

        torch.save(model.state_dict(),os.path.join(args.result_dir,f"epoch-{epoch}.pth"))
        print(f"epoch[{epoch}] finished")

    print("DONE")

    torch.save(model.state_dict(),os.path.join(args.result_dir,'last.pth'))




