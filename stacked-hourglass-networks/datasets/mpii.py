from torchvision.datasets import VisionDataset
from PIL import Image, ImageDraw
import os
import scipy.io
from numpy.lib import recfunctions as rfn
# from IPython.display import display
import numpy as np


# フォルダー構造
# datasetroot
#     |- mpii_human_pose_v1
#     |              |- images
#     |                    |- xxx.jpg
#     |- mpii_human_pose_v1_u12_2
#                    |- mpii_human_pose_v1_u12_1.mat 
# 注意: annotationのファイルの拡張子は.matでwindowsだと展開したら消えてることがあった???
# 
# データ数は約30K?

# データセットのREADME.md間違っている？
# - joint
class MPIIDataset(VisionDataset):
    def __init__(self,root:str,annotation_path:str,train=True,transforms=None):
        super().__init__(root)

        self.train=train
        self.transforms=transforms
        # annotationファイルは matlabの形式？なので　scipy.io　を使ったら読み込める

        # TODO: 確認
        # TODO: dict形式の方がよさそう 0:r-ankle
        self.classes=[
            'r ankle', # 0:右　足首
            'r knee', # 右　膝
            'r hip', # 右 臀部(腰)
            'l hip', # 左 臀部(腰)
            'l knee', # 左　膝
            'l ankle', # 左　足首
            'pelvis', # 骨盤
            'thorax', # 胸部
            'upper neck', # 首上
            'head top', # 頭上
            'r wrist', # 右手首
            'r elbow', # 右ひじ
            'r shoulder', # 右肩
            'l shoulder', # 左肩
            'l elbow', # 左ひじ
            'l wrist' # 左手首
        ]
        self.num_joints=16

        # 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist


        self.root=root
        self.annotation_path=annotation_path

        # assert exists image dir and annotation file
        
        # dict
        loaded_anno=scipy.io.loadmat(annotation_path)

        # ndarray  (structured )
        anno=loaded_anno['RELEASE']

        # anno['annolist']
        # anno['img_train']
        # anno['single_person']
        # anno['act']
        # anno['video_list']
        annolist=anno["annolist"]
        obj=annolist[0,0]

        self.length=anno['img_train'][0][0][0].shape[0]

        # TODO: jointのアノテーション情報保存　x,y,id   person-centeric body joint annotations

        def istrain(idx):
            # Return true if image is in training set
            return (anno['img_train'][0][0][0][idx] and
                    anno['annolist'][0][0][0]['annorect'][idx].size > 0 and
                    'annopoints' in anno['annolist'][0][0][0]['annorect'][idx].dtype.fields)
            

        # TODO: istrainでannopoints in ~の部分があるときと無い時の差は？確認する

        # TODO: normalize
        # TODO: torsoangle
        data=[]
        for i in range(self.length):

            # テストデータはkeypointなどの情報がないのでとばす
            # TODO: testだけのデータも作成したい
            # istrainは0を返したりする
            if istrain(i) != True:
                continue

            # ここからtrain
            annorect=annolist[0][0][0]["annorect"][i]
            annorect_length=annorect.shape[1]

            item=annolist[0][0][0]

            # TODO: データの取得方法でもっといい方法ありそう？  [0,0]とか本当に必要？
            image_name=item['image'][i][0]['name'][0][0]

            # body annotation for a person
            annorect=item['annorect'][i]
            annorect_length=annorect.shape[1]
            # a_rect=[]
            for annorect_idx in range(annorect_length):
                # annorect_idxは実質、人ごとのindex

                rect=annorect[0][annorect_idx]

                #  元コードの        if not c[0] == -1: の部分気になる

                # データがない時もあるのでとばす
                if rect['scale'].size == 0 or rect['objpos'].size == 0:
                    # データがない？？？
                    continue


                objpos_x=rect['objpos'][0,0]['x'][0,0]
                objpos_y=rect['objpos'][0,0]['y'][0,0]

                scale=rect['scale'][0,0]
                # TODO: ndarrayになっているがいいか？

                objpos={
                    "x":objpos_x,
                    "y":objpos_y,
                }

                kp_list=rect['annopoints'][0][0][0][0]
                kp_length=kp_list.shape[0]
                keypoints=[]
                for kp_idx in range(kp_length):

                    kp=kp_list[kp_idx]
                    id=kp['id'][0][0]
                    x=kp['x'][0][0]
                    y=kp['y'][0][0]

                    is_visible=None
                    
                    # https://stackoverflow.com/questions/59772847/mpii-pose-estimation-dataset-visibility-flag-not-present
                    # TODO: is_visibleのフィールドがないときはどうなっているか表示してみてみる
                    if 'is_visible' in kp.dtype.fields:

                        if  kp['is_visible'].shape[0] == 0:
                            # size=0のときは is_visible=1　であってるか？？
                            is_visible=1
                        else:
                            is_visible=int(kp['is_visible'])
                    else:
                        # keyとしてis_visibleがないときもある
                        # TODO: is_visibleがないときは、1？？？・
                        is_visible=1


                    keypoints.append({
                        'id':id,
                        'x':x,
                        'y':y,
                        'is_visible':is_visible
                    })


                x1=rect['x1'][0][0]
                y1=rect['y1'][0][0]
                x2=rect['x2'][0][0]
                y2=rect['y2'][0][0]

                data.append({
                    'image_filename':image_name,
                    # erson scale w.r.t. 200 px height
                    "scale":scale,
                    # rough human position in the image
                    "objpos":objpos,

                    'x1':x1,
                    'y1':y1,
                    'x2':x2,
                    'y2':y2,
                    'annopoints':keypoints
                })


            # TODO: check length

  
            # data.append({
            #     'image_filename':image_name,
            #     'anno_rect':a_rect,
            # })

            # TODO: あとで消す
            # break

        self.data=data
        # print("data:",data)
        # training / test  

        # print(rfn.flatten_descr(anno['annolist'][0].dtype))




        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html#scipy.io.loadmat





    def __getitem__(self, index: int) :
        # about 25k(training:28k,test:11k)


        data=self.data[index]
        annotation=data

        filename=data['image_filename']

        # ファイル構造によって変わる
        image_path=os.path.join(self.root,'mpii_human_pose_v1','images',filename)
        image=Image.open(image_path)
        if self.transforms is not None:
            image,annotation=self.transforms(image,annotation)

        # key:
        # image_filename, anno_rect,annopoints

        return image,annotation

    def __len__(self) -> int:
        return self.length

def draw_pair_line(image_draw,keypoints):
    red=(255,0,0)
    purple=(192,48,192)
    blue=(0,0,255)
    pairs=[
        # TODO: thorax 使わない？？？？

        # 足
        {
            'pair':(0,1), # r-ankle ---  r-knee
            'color':red
        },
        {
            'pair':(1,2), # r-knee  --- r-hip
            'color':red
        },
        {
            'pair':(2,6),# r-hip --- pelvis
            'color':red
        },
        {
            'pair':(5,4),# l-ankle  --- l-knee
            'color':blue
        },     
        {
            'pair':(4,3),# l-knee --- l-hip
            'color':blue
        },          
        {
            'pair':(3,6), # l-hip --- pelvis
            'color':blue
        },          
        
        # 手、腕
        {
            'pair':(10,11),# r-wrist --- r-elbow
            'color':red
        },          
        {
            'pair':(11,12),# r-elbow --- r-shoulder
            'color':red
        },           
        {
            'pair':(12,8), # r-shoulder --- upper-neck
            'color':red
        },              
        {
            'pair':(15,14),# l-writs --- l-elbow
            'color':blue
        },          
        {
            'pair':(14,13),# l-elbow --- l-shoulder
            'color':blue
        },          
        {
            'pair':(13,8), # r-shoulder --- neck
            'color':blue
        },    

        # 体
        {
            'pair':(6,8), #  pelvis ---  upper-neck
            'color':purple
        },    

        # 頭
        {
            'pair':(8,9), # upper-neck --- head-top
            'color':purple
        },    
    ]

    
    def get_keypoint_from_id(keypoints,id):
        keypoints=list(filter(lambda keypoint: keypoint['id'] == id,keypoints))
        if len(keypoints)==0:
            return None
        elif len(keypoints)==1:
            return keypoints[0]
        else:
            raise Exception('エラー')

    for pair in pairs:
        id0=pair['pair'][0]
        id1=pair['pair'][1]
        color=pair['color']

        keypoint0=get_keypoint_from_id(keypoints,id0)
        keypoint1=get_keypoint_from_id(keypoints,id1)

        if (keypoint0 is not None) and ( keypoint1 is not None):
            image_draw.line((keypoint0['x'],keypoint0['y'],keypoint1['x'],keypoint1['y']),fill=color,width=5)

        print("keypoint 0,1",keypoint0,keypoint1)
        

        # id0とid1を探す
    classes=[
        'r ankle', # 0:右　足首
        'r knee', # 右　膝
        'r hip', # 右 臀部(腰)
        'l hip', # 左 臀部(腰)
        'l knee', # 左　膝
        'l ankle', # 左　足首
        'pelvis', # 骨盤
        'thorax', # 胸部
        'upper neck', # 首上
        'head top', # 頭上
        'r wrist', # 右手首
        'r elbow', # 右ひじ
        'r shoulder', # 右肩
        'l shoulder', # 左肩
        'l elbow', # 左ひじ
        'l wrist' # 左手首
    ]

def draw_circle(image_draw,position,color=(255,0,0)):
    # position はlen=2のtuple (x,y)
    circle_size=8
    size=circle_size//2
    upper_left=(position[0]-size,position[1]-size)
    lower_right=(position[0]+size,position[1]+size)
    image_draw.ellipse((*upper_left,*lower_right),fill=color)
    
# def add_annotation(pil_image,anno):
#     img=pil_image.copy()

#     draw=ImageDraw.Draw(img)

#     print(anno)
#     rect_list=anno['anno_rect']
#     for rect in rect_list:
#         print(rect)
#         objpos=rect['objpos']
#         objpos_x=objpos['x']
#         objpos_y=objpos['y']
#         scale=rect['scale']

#         print("objpos",objpos)
#         print("scale:",scale)

#         # 中心
#         draw_circle(draw,(objpos_x,objpos_y),color=(0,0,255))
#         # 全体
#         h=200*scale
#         h_half=h/2
#         upper_left= (objpos_x-h_half,objpos_y - h_half)
#         lower_right= (objpos_x + h_half , objpos_y + h_half )
#         draw.rectangle([upper_left,lower_right],outline=(0,0,255),width=2)

#         # cropした画像表示 & keypointの座標変換
#         print("upper_left",upper_left)
#         print("lower right",lower_right)
#         cropped=pil_image.crop((*upper_left,*lower_right))
#         draw_cropped=ImageDraw.Draw(cropped)
#         display(cropped)
#         print("crop image size:",cropped.size)
#         # crop用 annopoints
#         crop_annopoints=[]
#         for point in rect['annopoints']:
#             key_x=point['x']
#             key_y=point['y']

#             key_x=key_x - upper_left[0]
#             key_y=key_y - upper_left[1]

#             crop_annopoints.append({
#                 'x':key_x,
#                 'y':key_y,
#                 'id':point['id'],
#                 'is_visible':point['is_visible']
#             })

#             draw_circle(draw_cropped,(key_x,key_y))

#             # TODO: 座標を変換する  元の画像の座標 => クロップした画像の座標へ
#             # x,y=transform((key_x,key_y))
#         draw_pair_line(draw_cropped, crop_annopoints)
#         display(cropped)


#         # keypointの表示
#         x1,y1=rect['x1'],rect['y1']
#         x2,y2=rect['x2'],rect['y2']
#         draw.rectangle((x1,y1,x2,y2),outline=(255,0,0))




#         annopoints=rect['annopoints']
#         for point in annopoints:
#             print(point)
#             key_x=point['x']
#             key_y=point['y']
#             draw_circle(draw,(key_x,key_y))


#     # TODO: cropした画像を表示

#     return img



def __test_mpii_dataset():

    ds=MPIIDataset("./data/MPII","./data/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat")
    print("len",len(ds))



    # display(img)
    num_imgs=1
    for i in range(num_imgs):
        img,anno=ds[i]


        # display(img)


        # print(type(img))
        # print(ds[])
        # img_anno=add_annotation(img,anno)
        # display(img_anno)


    
# __test_mpii_dataset()