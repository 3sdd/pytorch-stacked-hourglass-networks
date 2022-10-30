import math
import torch
import torchvision.transforms as transforms
import random
from typing import  Tuple
# import seaborn as sns
# import matplotlib.pyplot as plt


def generate_batched_heatmaps(num_joints:int,num_stacks:int,batched_annotations,heatmap_res:int,gaussian_sigma=1):
    list_heatmap=[]
    for i in range(len(batched_annotations)):
        anno=batched_annotations[i]

        heatmap=generate_heatmap(num_joints,num_stacks,anno,heatmap_res,gaussian_sigma)
        list_heatmap.append(heatmap)
    # batch分をstackして　(8,16,64,64)=>(batch_size,8,16,64,64)
    gt_heatmaps=torch.stack(list_heatmap)

    return gt_heatmaps

def clamp(value,min_value,max_value):
    return max(min(value, max_value), min_value)

# torchvision.transforms.functional.gaussian_blur でいける？？？？
# position(x,y)
def generate_heatmap(num_joints:int,num_stacks,annotations,heatmap_res:int ,gaussian_sigma=1):
    # heatmap_width,hegith=64

    # TODO: image resolutionも入力させる？？？
    image_resolution=256

    # print(annotations)

    gt_heatmaps=torch.zeros((num_joints,heatmap_res,heatmap_res))
    annopoints=annotations['annopoints']
    # test_img=torch.zeros((1,64,64))
    for i in range(len(annopoints)):
        point=annopoints[i]
        pos_x,pos_y=point['x'],point['y']
        id=point['id']

        # 画像サイズとヒートマップのサイズが違うので座標変換する。 256x256=>64x64
        ratio=heatmap_res/image_resolution
        pos_x,pos_y=pos_x*ratio, pos_y*ratio

        # TODO: [0,image_size-1]の範囲でclampする必要ある？
        # TODO: 切り捨て(floor) or 四捨五入 (round) どっちにする？　今は切り捨てている
        pos_x=math.floor(pos_x)
        pos_y=math.floor(pos_y)

        # TODO: はみ出ていることあるので、どのようなところがはみ出ているのか見てみたい。もしかしたらバグ？
        # はみ出て姉妹っているものは、とりあえず無視してheatmap上には置かない
        if pos_x<0 or pos_x>=heatmap_res-1 or pos_y<0 or pos_y>=heatmap_res-1:
            continue

        # TODO: sigma引数で入れる
        gt_heatmaps[id]=draw_gaussian(gt_heatmaps[id],(pos_x,pos_y))

    

        # test_img[0,pos_y,pos_x]=1

        # to_pil=torchvision.transforms.functional.to_pil_image
        # display(to_pil(gt_heatmaps[i].unsqueeze(dim=0)))

        # img=torchvision.transforms.functional.gaussian_blur(img,kernel_size=size,sigma=sigma)
    # display(to_pil(test_img).resize((256,256)))

    gt_heatmaps=gt_heatmaps.unsqueeze(dim=0).repeat(num_stacks,1,1,1)

    return gt_heatmaps


def __test_gt_hm():
    annos=[
        {'image_filename': '015601864.jpg', 'scale': 3.021046176409755, 'objpos': {'x': 594, 'y': 257}, 'x1': 627, 'y1': 100, 'x2': 706, 'y2': 198, 'annopoints': [{'id': 6, 'x': 134.7791085617694, 'y': 98.341400042259, 'is_visible': 0}, {'id': 7, 'x': 150.45579711086106, 'y': 93.68076290604255, 'is_visible': 1}, {'id': 8, 'x': 146.2273705148859, 'y': 99.53549764598615, 'is_visible': 1}, {'id': 9, 'x': 171.20830082614907, 'y': 64.94653677012731, 'is_visible': 1}, {'id': 0, 'x': 139.01605141287524, 'y': 186.04611706015027, 'is_visible': 1}, {'id': 1, 'x': 137.3212742724329, 'y': 133.08433142132705, 'is_visible': 1}, {'id': 2, 'x': 119.1024200126777, 'y': 97.49401147203783, 'is_visible': 1}, {'id': 3, 'x': 150.45579711086106, 'y': 98.76509432736958, 'is_visible': 0}, {'id': 4, 'x': 156.38751710240925, 'y': 112.74700573601892, 'is_visible': 1}, {'id': 5, 'x': 154.26904567685634, 'y': 116.98394858712477, 'is_visible': 1}, {'id': 10, 'x': 133.08433142132705, 'y': 111.05222859557657, 'is_visible': 1}, {'id': 11, 'x': 110.62853431046598, 'y': 87.32534862938377, 'is_visible': 1}, {'id': 12, 'x': 130.9658599957741, 'y': 89.86751434004728, 'is_visible': 1}, {'id': 13, 'x': 169.52203994083743, 'y': 97.49401147203783, 'is_visible': 1}, {'id': 14, 'x': 169.945734225948, 'y': 120.79719715312005, 'is_visible': 1}, {'id': 15, 'x': 167.8272628003951, 'y': 151.72687996619283, 'is_visible': 1}]},
        {'image_filename': '043194502.jpg', 'scale': 3.6099861495579177, 'objpos': {'x': 338, 'y': 258}, 'x1': 299, 'y1': 86, 'x2': 391, 'y2': 205, 'annopoints': [{'id': 6, 'x': 118.42655606747071, 'y': 181.18579962516267, 'is_visible': 0}, {'id': 7, 'x': 117.71741207246855, 'y': 110.27140012494577, 'is_visible': 0}, {'id': 8, 'x': 119.83406506875102, 'y': 106.59665140004428, 'is_visible': 1}, {'id': 9, 'x': 141.12994289626414, 'y': 69.62464916221168, 'is_visible': 1}, {'id': 1, 'x': 177.28550765265075, 'y': 186.50437958767893, 'is_visible': 1}, {'id': 2, 'x': 114.52626409495878, 'y': 184.02237560517133, 'is_visible': 1}, {'id': 3, 'x': 121.97227604248155, 'y': 178.349223645154, 'is_visible': 0}, {'id': 4, 'x': 192.17753154769628, 'y': 181.18579962516267, 'is_visible': 1}, {'id': 10, 'x': 171.6123556926334, 'y': 176.57636365764859, 'is_visible': 1}, {'id': 11, 'x': 126.58171200999566, 'y': 171.2577836951323, 'is_visible': 1}, {'id': 12, 'x': 117.71741207246855, 'y': 116.2991240824642, 'is_visible': 1}, {'id': 13, 'x': 117.71741207246855, 'y': 104.24367616742732, 'is_visible': 0}, {'id': 14, 'x': 114.1716920974577, 'y': 57.08560049978308, 'is_visible': 1}, {'id': 15, 'x': 100.3433841949154, 'y': 91.83365625488936, 'is_visible': 1}]},
        # 3 エラーになる
        {'image_filename': '052475643.jpg', 'scale': 1.7618354066143638, 'objpos': {'x': 316, 'y': 220}, 'x1': 316, 'y1': 105, 'x2': 361, 'y2': 163, 'annopoints': [{'id': 6, 'x': 132.3590905093446, 'y': 167.95832966899238, 'is_visible': 1}, {'id': 7, 'x': 135.9916659337985, 'y': 93.12727592524301, 'is_visible': 1}, {'id': 8, 'x': 138.16721535550388, 'y': 85.93848181175731, 'is_visible': 1}, {'id': 9, 'x': 150.52596346458077, 'y': 45.100923587030124, 'is_visible': 1}, {'id': 0, 'x': 128.72651508489076, 'y': 267.49089629902795, 'is_visible': 1}, {'id': 1, 'x': 129.45303016978153, 'y': 185.39469170637088, 'is_visible': 1}, {'id': 2, 'x': 109.11060779283996, 'y': 167.2318145841016, 'is_visible': 1}, {'id': 3, 'x': 154.88105814095852, 'y': 167.95832966899238, 'is_visible': 1}, {'id': 4, 'x': 191.20681238549705, 'y': 185.39469170637088, 'is_visible': 1}, {'id': 5, 'x': 184.66817662148011, 'y': 265.31135104435566, 'is_visible': 1}, {'id': 10, 'x': 113.46969830218458, 'y': 66.2462177842845, 'is_visible': 1}, {'id': 11, 'x': 102.57197202882303, 'y': 119.28181898131075, 'is_visible': 1}, {'id': 12, 'x': 113.46969830218458, 'y': 89.49470050078915, 'is_visible': 1}, {'id': 13, 'x': 158.51363356541236, 'y': 96.75985134969686, 'is_visible': 1}, {'id': 14, 'x': 179.5825710272447, 'y': 139.62424135825233, 'is_visible': 1}, {'id': 15, 'x': 183.21514645169856, 'y': 89.49470050078915, 'is_visible': 1}]},
        # 4
        # {'image_filename': '052475643.jpg', 'scale': 2.095709903588758, 'objpos': {'x': 499, 'y': 232}, 'x1': 495, 'y1': 105, 'x2': 551, 'y2': 172, 'annopoints': [{'id': 6, 'x': 147.54468981124668, 'y': 152.4308622640584, 'is_visible': 1}, {'id': 7, 'x': 145.71237514144232, 'y': 95.0183359435212, 'is_visible': 1}, {'id': 8, 'x': 145.337544637156, 'y': 92.05731543711731, 'is_visible': 1}, {'id': 9, 'x': 139.97949007971403, 'y': 49.728403478409774, 'is_visible': 1}, {'id': 0, 'x': 115.17379731136934, 'y': 234.8850224052554, 'is_visible': 1}, {'id': 1, 'x': 112.11993952836204, 'y': 173.19709518850797, 'is_visible': 1}, {'id': 2, 'x': 125.55691377359415, 'y': 151.20931915085546, 'is_visible': 1}, {'id': 3, 'x': 169.53246584889922, 'y': 153.04163382065985, 'is_visible': 1}, {'id': 4, 'x': 177.4724960847182, 'y': 174.4186383017109, 'is_visible': 1}, {'id': 5, 'x': 179.91558231112404, 'y': 233.66347929205247, 'is_visible': 1}, {'id': 10, 'x': 124.94614221699268, 'y': 99.90450839633287, 'is_visible': 1}, {'id': 11, 'x': 109.06608174535475, 'y': 132.27540089621021, 'is_visible': 1}, {'id': 12, 'x': 115.7845688679708, 'y': 97.46142216992703, 'is_visible': 1}, {'id': 13, 'x': 175.64018141491383, 'y': 91.96447816051389, 'is_visible': 1}, {'id': 14, 'x': 181.74789698092843, 'y': 126.77845688679709, 'is_visible': 1}, {'id': 15, 'x': 160.9816640564788, 'y': 96.23987905672412, 'is_visible': 1}]},
    ]

    num_joints=16
    num_stacks=8
    heatmaps=generate_batched_heatmaps(num_joints,num_stacks,annos,64)

    print(heatmaps.size())

    # hm=generate_heatmap(16,64,64,pos)

    # for i in range(hm.size(0)):
    #     # TODO: to_pil
    #     print(hm[i].unsqueeze(dim=0).size())
    #     heatmap_img=torchvision.transforms.functional.to_pil_image(hm[i].unsqueeze(dim=0))
    #     display(heatmap_img)
# __test_gt_hm()





class CustomTransforms():
    def __init__(self):
        # 画像を[0,1]へ
        self.to_tensor=transforms.ToTensor()

        # TODO: crop
        self.resize_target_res=256 # resolution
        self.resize=transforms.Resize((self.resize_target_res,self.resize_target_res))
        # TODO: resize (256,256)
        # TODO: rotation (+/- 30 degree)
        # TODO: scale (.75 -1.25)  1/4

    def __call__(self,pil_image,anno):

        # 中心
        objpos=anno['objpos']
        objpos_x=objpos['x']
        objpos_y=objpos['y']
        # サイズ
        scale=anno['scale']
        h=200*scale
        h_half=h/2

        # 人ごとのbounding box
        upper_left= (objpos_x-h_half,objpos_y - h_half)
        lower_right= (objpos_x + h_half , objpos_y + h_half )
        # 人の領域付近でcrop
        pil_img=pil_image.crop((*upper_left,*lower_right))
        # tmp_img=pil_img.copy()
        # draw=ImageDraw.Draw(tmp_img)
        # cropしたのでkeypointの更新
        # TODO: objpos,scaleの更新する？？？？　or もう必要ないので消しておく？？？
        annopoints=anno['annopoints']
        for i in range(len(annopoints)):
            
            point=annopoints[i]
            key_x=point['x']
            key_y=point['y']
            # id=point['id']
            # is_visible=point['is_visible']

            key_x=key_x - upper_left[0]
            key_y=key_y - upper_left[1]

            annopoints[i]['x']=key_x
            annopoints[i]['y']=key_y

            # draw_circle(draw,(key_x,key_y))
        # display(tmp_img)

        # [0,1]のtensorに変換
        img=self.to_tensor(pil_img)

        # 256x256に画像をリサイズ
        img=self.resize(img)
        # annotationのkeypointの座標変更
        ratio=self.resize_target_res/h
        for i in range(len(annopoints)):
            point=annopoints[i]
            annopoints[i]['x']=point['x'] * ratio
            annopoints[i]['y']=point['y'] * ratio
            

        # print("IMG")
        # display(transforms.functional.to_pil_image(img))
        # img2=transforms.functional.to_pil_image(img)
        # draw=ImageDraw.Draw(img2)

        # for i in range(len(annopoints)):
            
        #     point=annopoints[i]
        #     key_x=point['x']
        #     key_y=point['y']
        #     draw_circle(draw,(key_x,key_y))
        # display(img2)


        
        # x軸右側が正、y軸下側が正、反時計回りの角度が正

        angle = random.randint(-30, 30)
        angle_radian=math.radians(angle)
        # print("Angle",angle)
        
        # 画像回転
        img=transforms.functional.rotate(img,angle)
        # 座標回転
        rotatino_matrix=torch.tensor([
            math.cos(angle_radian),math.sin(angle_radian),
            -math.sin(angle_radian),math.cos(angle_radian),
        ]).view(2,2)
        for i in range(len(annopoints)):
            point=annopoints[i]
            
            key_x=point['x'] - self.resize_target_res//2 # 中央を中心に回転させているので中央中心になるように移動させる
            key_y=point['y'] - self.resize_target_res//2

            # keypointの座標を回転させる            
            result= torch.matmul(rotatino_matrix, torch.tensor([key_x,key_y],dtype=torch.float32))
            key_x=result[0] + self.resize_target_res//2 # 中央中心に変更していたのを元に戻す
            key_y=result[1] + self.resize_target_res//2

            annopoints[i]['x']=key_x
            annopoints[i]['y']=key_y


        return img,anno


def collate_fn(batch):
    images, annos= list(zip(*batch))
    images=torch.stack(images)

    return images,annos




def draw_circle(image_draw,position,color=(255,0,0)):
    # position はlen=2のtuple (x,y)
    circle_size=8
    size=circle_size//2
    upper_left=(position[0]-size,position[1]-size)
    lower_right=(position[0]+size,position[1]+size)
    image_draw.ellipse((*upper_left,*lower_right),fill=color)



def draw_gaussian(image:torch.Tensor,location:Tuple[int,int]):
    # image size: (height,width)　を想定
    delta=3
    size=2*delta+1
    x0=y0=size//2
    sigma=1

    image_width=image.size(1)
    image_height=image.size(0)
    # TODO: 範囲外の時の処理

    x=torch.arange(0,size,1,dtype=torch.float)
    y=x[:,None]
    # 中心が1になるようなgaussian func
    g_torch=torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


    # sns.heatmap(g_torch.numpy(),vmin=0,vmax=1.0)
    # plt.show()

    # 左上、右下のx,y  
    # TODO: int変換の方法あってる？floor or ceil
    upper_left_x= int(location[0])-delta
    upper_left_y=int(location[1])-delta

    lower_right_x=int(location[0])+delta+1
    lower_right_y= int(location[1])+delta+1

    # 画像の変更部分の範囲
    image_range_x_start= max(0,upper_left_x)
    image_range_x_end= min(image_width,lower_right_x)
    image_range_y_start=max(0,upper_left_y)
    image_range_y_end=min(image_height,lower_right_y)
    # print("g size:",g_torch.size())
    # print("image size:",image.size())
    # gaussianを画像に書き込み


    # print(image_range_x_start,image_range_x_end)
    # print(image_range_y_start,image_range_y_end)
    # print(image[image_range_y_start:image_range_y_end,image_range_x_start:image_range_y_end].size())

    
    gaussian_range_x_start=max(0,-upper_left_x)   # upper left xがマイナスの時は、マイナス部分だけずらしたところからスタート
    gaussian_range_x_end= min(image_width,lower_right_x) - upper_left_x
    gaussian_range_y_start=max(0,-upper_left_y)
    gaussian_range_y_end= min(image_height,lower_right_y) - upper_left_y
    # print(gaussian_range_x_start,gaussian_range_x_end)
    # print(gaussian_range_y_start,gaussian_range_y_end)

    # print("pos_x,y",location[0],location[1])
    # print("g size:",g_torch.size())

    # print("image slice")
    # print(image[image_range_y_start:image_range_y_end,image_range_x_start:image_range_x_end])

    # 画像範囲を作る
    image[image_range_y_start:image_range_y_end,image_range_x_start:image_range_x_end] \
        =g_torch[gaussian_range_y_start:gaussian_range_y_end,gaussian_range_x_start:gaussian_range_x_end]
    return image