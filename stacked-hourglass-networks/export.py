from multiprocessing import dummy
import torch
import argparse
from models.stacked_hg import StackedHourglassNetwork


def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-path',type=str,default='../results/last.pth')
    parser.add_argument('--image-size',type=int,default=256)
    parser.add_argument('--output-path',type=str,default="../results/stacked-hg.onnx")
    parser.add_argument('--verbose',action='store_true')
    return parser

def export_as_onnx():
    args=get_parser().parse_args()

    model=StackedHourglassNetwork(heatmap_channels=16)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()


    dummy_input=torch.randn(1,3,args.image_size,args.image_size)

    input_names=['input']
    output_names=['output']

    torch.onnx.export(model,dummy_input,args.output_path,verbose=args.verbose,input_names=input_names,output_names=output_names)


    # TODO: 元モデルと同じになるか判定する


if __name__=="__main__":
    export_as_onnx()