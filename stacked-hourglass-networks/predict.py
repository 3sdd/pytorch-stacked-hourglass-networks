import argparse
import os
import torch
import torchvision.transforms as transforms
from models.stacked_hg import StackedHourglassNetwork
from utils import CustomTransforms,draw_circle
from PIL import ImageDraw,Image
from datasets.mpii import MPIIDataset

def get_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('--model-path',type=str,default='../results/last.pth',help='model path')
    parser.add_argument('--image-path',type=str,default='../sample-images/marianna-smiley-1I4hATI7idg-unsplash.jpg',help='input image path')
    parser.add_argument('--outdir',type=str,default='../results/predicts/',help='outdir ')

    return parser.parse_args()


@torch.no_grad()
def predict():
    args=get_args()
    os.makedirs(args.outdir,exist_ok=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_joints=16
    image_size=256

    # load model
    model=StackedHourglassNetwork(heatmap_channels=num_joints)
    model.load_state_dict(torch.load( args.model_path,map_location='cpu'))
    model=model.to(device)
    model.eval()

    # my_transforms=CustomTransforms()
    transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    # dataset=MPIIDataset(args.data_root,os.path.join(args.data_root,"mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"),transforms=my_transforms)
    # load image (test???)
    # とりあえず　trainのデータを読み取る
    # image,anno=dataset[0]

    
    to_pil=transforms.functional.to_pil_image

    # 画像読み込み
    pil_image=Image.open(args.image_path).convert('RGB')
    # リサイズ
    pil_image=transforms.functional.resize(pil_image,(256,256))

    pil_image.save(os.path.join(args.outdir,'input.png'))
    # display(pil_image)

    image=transforms.functional.to_tensor(pil_image)
    image=image.unsqueeze(dim=0).to(device)
    print(image.size())

    # 推論
    heatmap=model(image)

    heatmap=heatmap[0]
    # (8,16,64,64)
    # (num_stacks, num_joints, heigth, width)
    print(heatmap.size())

    last_heatmap=heatmap[-1,:,:,:]
    # (16,64,64)

    # TODO: save heatmap

    c,h,w=last_heatmap.size()

    print(last_heatmap)
    print(last_heatmap.view(c,-1).size())
    max_idx=torch.argmax(last_heatmap.view(c,-1),dim=1)
    print(max_idx)
    def to_xy_index(index:int,image_size:int):
        # image_size: 256
        # 0-255
        y_idx=index//image_size
        x_idx=index%image_size
        return x_idx,y_idx


    x_idx,y_idx=to_xy_index(max_idx,last_heatmap.size(1))  
    print("size:",x_idx.size())
    # TODO: heatmapを保存
    
    # TODO: heatmapのサイズは？　64x64???
    # リサイズして画像上にheatmapのポイントを表示
    ratio=image_size/h
    draw=ImageDraw.Draw(pil_image)

    for i in range(x_idx.size(0)):
        x=x_idx[i]
        y=y_idx[i]

        x=x*ratio
        y=y*ratio
        draw_circle(draw,(x,y),color=(255,0,0))

        
    print(x_idx,y_idx)

    # display(pil_image)
    pil_image.save(os.path.join(args.outdir,'predict_out.png'))


if __name__=="__main__":
    predict()