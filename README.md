

# stacked hourglass networks の再実装


[公式実装](https://github.com/princeton-vl/pose-hg-train)を真似してpytorchで実装したもの。


- arxiv: https://arxiv.org/abs/1603.06937
- 2016年



```
cd stacked-hourglass-networks
```

```sh
python train.py --data-root="E:\Datasets\MPII" --cudnn-benchmark --amp
```


```sh
python predict.py --model-path="../results/epoch-32.pth" --data-root="E:\Datasets\MPII"
```

## ライブラリ

- python 3.10
- pytorch
- torchvision
- tqdm
- scipy

## データ

[MPIIデータセット](http://human-pose.mpi-inf.mpg.de/#download)