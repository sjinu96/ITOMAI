from re import S
import pandas as pd
from PIL import Image
import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from utils import convert_rgb_to_label
from utils import convert_label_to_rgb
from segment import segment_single_image
import cv2
from finetune import start_adaptation
from finetune import generate_image
from finetune import Args
import numpy as np
from trainers.pix2pix_trainer import Pix2PixTrainer
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel

# 트리거 저장소 
class Trigger:
    def __init__(self):
        self.upload=None
        self.gpu=None
        self.cpu=None

        self.init=True
        self.init2=True

        self.sa=True
        self.gi=True
        self.fg=True

        self.canvas=True
        self.save_image=False

# 변수 저장소(Variable)
class Var:
    def __init__(self):
        self.model = None
        self.origin_image=None
        self.origin_shape=None
        self.origin_label_map=None
        self.rgb_label_map=None

        self.width=None
        self.height=None

        self.uploaded_file=None

        self.output=None

# 같은 이미지에 대해서 Segmentation은 한 번만 진행.
@st.cache(allow_output_mutation=True)
def segment(origin_image, device):
    # Segment Image
    origin_label_map = segment_single_image(origin_image, device)   
    print('origin_label_map .shape(segment)', origin_label_map.shape)
    # print('origin :', np.unique(origin_label_map)) # coco-stuff label
    # 배경용 라벨
    rgb_label_map=convert_label_to_rgb(origin_label_map) # label to rgb
    # print('rgb합:',  np.unique(rgb_label_map[:, :, 0]+rgb_label_map[:, :, 1]+rgb_label_map[:, :, 2]))
    # rgb_label_map=Image.fromarray(rgb_label_map)
    origin_label_map = convert_rgb_to_label(rgb_label_map, origin_label_map) # 결측치 처리
    rgb_label_map = Image.fromarray(rgb_label_map)
    return origin_label_map, rgb_label_map

def load_segment(origin_image, model_num):
    # Segment Image
    origin_label_map = Image.open(f'required_data/Flickr/00{model_num}.png') #.resize((512,512), resample=0)
    origin_label_map = np.array(origin_label_map)

    print('origin_labael_map:', origin_label_map.shape)
    print('uniquqe : ', np.unique(origin_label_map))
    # 배경용 라벨
    rgb_label_map=convert_label_to_rgb(origin_label_map)
    print('rgb_label_map : ', rgb_label_map.shape)
    origin_label_map = convert_rgb_to_label(rgb_label_map, origin_label_map)
    rgb_label_map=Image.fromarray(rgb_label_map)
    # print('dsfsdf')
    return origin_label_map, rgb_label_map

# 최초 실행시에만
def initialize(model_num=False):
    uploaded_file = Image.open(st.session_state.var.uploaded_file)
    height, width, _= np.array(uploaded_file)[:, :, :3].shape
    # print('initi')
    if max(height, width) > 640 : 
        if height>width:
            ratio = 640 / height
            width = round(width * ratio)
            height = round(height * ratio)
        else:
            ratio = 640 / width
            width = round(width * ratio)
            height = round(height * ratio)
    
    uploaded_file=uploaded_file.resize((width, height), resample=0)
    
    # assert 1==0
    # [:, :, :3] (??) 마지막이 rgba 형식이면 따로 함수 불러와야 하는 걸로 알고 있긴 한데..


    origin_image = np.array(uploaded_file)[:, :, :3]
    origin_shape = origin_image.shape

    origin_image_512=np.array(uploaded_file.resize((512,512)))[:, :, :3]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # segment
    data_load_state = st.text('Segmenting...')
    if st.session_state.trigger.gpu:
        origin_label_map, rgb_label_map = segment(origin_image, device)
    if st.session_state.trigger.cpu:
        origin_label_map, rgb_label_map = load_segment(origin_image, model_num)
        
    data_load_state.text("Done!")
    print('origin_label_map.shape', origin_label_map.shape)
    st.session_state.trigger.uploaded=True
    st.session_state.var.origin_image=origin_image
    st.session_state.var.origin_image_512=origin_image_512
    st.session_state.var.origin_shape=origin_shape
    st.session_state.var.origin_label_map=origin_label_map
    st.session_state.var.rgb_label_map=rgb_label_map

    st.session_state.trigger.init=False


def canvas():
    # print('canvas 실행')

    if st.session_state.trigger.init:  
        # print('(test) initialize !! ') 
        initialize() # 최초 1회만

    # Original Image를 최상단에.
    st.image(st.session_state.var.origin_image, "Original Image")

    label_key_dict = {8: 'boat', 14: 'bench', 96:'bush', 105: 'clouds',  110: 'dirt', 112: 'fence', 118: 'flower', 119: 'fog',
    123: 'grass',  124: 'gravel', 125: 'ground_other', 126: 'hill', 127: 'house', 128: 'leaves', 134: 'mountain', 135: 'mud',
    139: 'pavement', 141: 'plant_other', 147: 'river', 148: 'road', 149: 'rock', 153: 'sand', 154: 'sea', 156: 'sky_other', 
    158: 'snow', 161: 'stone', 168: 'tree', 177: 'water_other',  181: 'wood'}
    key_label_dict = {label_key_dict[k]:k for k in label_key_dict}
    
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 75, 50)

    label = st.sidebar.selectbox("Choose a semantic label.", options = key_label_dict.keys())
   
    if label == "boat":
        stroke_color = st.sidebar.color_picker("boat", value = "#7C8E8A")
    elif label == "bench":
        stroke_color = st.sidebar.color_picker("bench", value = "#FF0600")
    elif label == "fence":
        stroke_color = st.sidebar.color_picker("fence", value = '#7D0200')
    elif label == "house":
        stroke_color = st.sidebar.color_picker("house", value = '#DFFF00')
    elif label == "pavement":
        stroke_color = st.sidebar.color_picker("pavement", value = '#7F7F7F')
    elif label == "road":
        stroke_color = st.sidebar.color_picker("road", value = '#AF4BD8')
    elif label == "bush":    
        stroke_color = st.sidebar.color_picker("bush", value = '#6AA450')
    elif label == "clouds":
        stroke_color = st.sidebar.color_picker("clouds", value = '#19F7FF')
    elif label == "dirt":
        stroke_color = st.sidebar.color_picker("dirt", value = '#CBB138')
    elif label == "flower":
        stroke_color = st.sidebar.color_picker("flower", value = '#D86D6D')
    elif label == "fog":    
        stroke_color = st.sidebar.color_picker("fog", value = '#3A4481')
    elif label == "grass":    
        stroke_color = st.sidebar.color_picker("grass", value = '#17AA17')
    elif label == "gravel":    
        stroke_color = st.sidebar.color_picker("gravel", value = '#751A23')
    elif label == "ground_other":    
        stroke_color = st.sidebar.color_picker("ground_other", value = '#C3C074')
    elif label == "hill":    
        stroke_color = st.sidebar.color_picker("hill", value = '#5FD87D')
    elif label == "leaves":    
        stroke_color = st.sidebar.color_picker("leaves", value = '#00B92F')
    elif label == "mountain":    
        stroke_color = st.sidebar.color_picker("mountain", value = '#2CEC91')
    elif label == "mud":    
        stroke_color = st.sidebar.color_picker("mud", value = '#965103')
    elif label == "plant_other":    
        stroke_color = st.sidebar.color_picker("plant_other", value = '#618141')
    elif label == "river":
        stroke_color = st.sidebar.color_picker("river", value = '#5F62B9')
    elif label == "rock":
        stroke_color = st.sidebar.color_picker("rock", value = '#5C4114')
    elif label == "sand":
        stroke_color = st.sidebar.color_picker("sand", value = '#E6C691')
    elif label == "sea":
        stroke_color = st.sidebar.color_picker("sea", value = '#362CF7')
    elif label == "sky_other":
        stroke_color = st.sidebar.color_picker("sky_other", value = '#3EE8F5')
    elif label == "snow":
        stroke_color = st.sidebar.color_picker("snow", value = '#D0D9E2')
    elif label == "tree":
        stroke_color = st.sidebar.color_picker("tree", value = '#1A691A')
    elif label == "water_other":
        stroke_color = st.sidebar.color_picker("water_other", value = '#4A7A81')
    elif label == "wood":
        stroke_color = st.sidebar.color_picker("wood", value = '#AF5E22')
    
    # Create a canvas component
    
    # if st.session_state.trigger.init_label:
    #    st.session_state.canvas_result

    # if not st.session_state.trigger.init2: 
        # st.image(st.session_state.var.canvas_result.image_data, "output1")
        # st.session_state.var.canvas_result.image_data.save('test3.jpg')
    # print(st.session_state.var.origin_shape) 
    height, width, _ = st.session_state.var.origin_shape
    
    st.session_state.var.height=height
    st.session_state.var.width=width
    if st.session_state.trigger.canvas: 
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=st.session_state.var.rgb_label_map,
            update_streamlit=True,
            width = width,
            height = height,
            drawing_mode="freedraw",
            key="canvas"
        )

        # if not st.session_state.trigger.init2: 
        #     st.image(canvas_result.image_data, "output_last")
        #     st.session_state.var.canvas_result = canvas_result
        #     canvas_result.image_data.save('test0.jpg')
        # canvas_result.image_data = st.session_state.var.rgb_label_map
        st.session_state.var.canvas_result = canvas_result
        # st.session_state.var.canvas_result.image_data.save('test1.jpg')
        # st.image(st.session_state.var.canvas_result, output1)
        
        # if st.session_state.trigger.init2:  
        #     # print('sdfasdfasdfasdfasdfasdfasdfsad')
        #     # canvas_result.image_data = st.session_state.var.rgb_label_map
        #     # st.image(canvas_result.image_data)
        #     st.session_state.trigger.init2 = False
        
        if not st.session_state.trigger.init2:
            rgb_label_map_copy=np.array(st.session_state.var.rgb_label_map).copy()
            # st.image(rgb_label_map_copy, 'rgb')
            # print(rgb_label_map_copy[200:300, 1:3])
            # print(np.unique((np.array(canvas_result.image_data)[:, :, :3])))#.astype(np.uint8))))
            temp=np.array(Image.fromarray((np.array(canvas_result.image_data)[:, :, :3]).astype(np.uint8)).resize((512,512), resample=0))
            # st.image(temp, 'temp')
            # print(temp.shape)
            # print(np.unique(temp))
            mask = (temp) !=0

            # print(mask)
            # print(mask.shape)
            # print(rgb_label_map_copy.shape
            # print(np.unique(rgb_label_map_copy))     # print(canvas_result.image_data[:, :, :3])
            rgb_label_map_copy[mask]=0
            # print(np.unique(rgb_label_map_copy))
            canvas_result.image_data = rgb_label_map_copy + temp
            # st.image(rgb_label_map_copy, 'rgb')
            canvas_result.image_data = np.array(Image.fromarray(canvas_result.image_data).resize((width, height), resample=0))

        st.session_state.var.canvas_result=canvas_result
        # st.image(st.session_state.var.canvas_result.image_data, "output2")
        # st.image(canvas_result.image_data, 'output')
        # Image.fromarray(canvas_result.image_data).save('label.png')
        if st.session_state.trigger.init2:
            st.session_state.trigger.init2=False
    else:
        pass

def load_cpu_model(model_num):
        initialize(model_num)

        opt = Args()
        
        opt.gpu_ids=[]
        # opt = TestOptions().parse()
        opt.isTrain=False
        opt.phase='test'
        opt.which_epoch=model_num # 3 if 003.jpg
        model = Pix2PixModel(opt)
        
        model.eval()
        
        # trainer = Pix2PixTrainer(opt)
        x = {"label" : torch.tensor(st.session_state.var.origin_label_map).unsqueeze(0).unsqueeze(0).float(),
            "instance" : torch.tensor([0]),
            "image" : st.session_state.var.origin_image,
            "path" : ""}

        # trainer.pix2pix_model.eval()
        # st.session_state.var.model=trainer
        st.session_state.var.model=model
        st.session_state.var.x=x

        st.session_state.trigger.upload=True 

def generate_output():
     # 모델 불러오기
    model = st.session_state.var.model
    x = st.session_state.var.x
    # st.image(st.session_state.var.canvas_result.image_data, 'output3')
    user_label_map = np.array(Image.fromarray(st.session_state.var.canvas_result.image_data[:, :, :3]).resize((512, 512), resample=0))
    # st.image(st.session_state.var.origin_label_map, 'output')
    user_label_map = convert_rgb_to_label(user_label_map, st.session_state.var.origin_label_map)
    # st.image(user_label_map, 'user_label_map')
    x['label']=user_label_map
    print(x['image'].shape, user_label_map.shape)
    output = generate_image(model, x, user_label_map, st.session_state.var.origin_shape)
    
    output=np.array(Image.fromarray(output).resize((st.session_state.var.width, st.session_state.var.height ), resample=0))
    st.image(output, caption = "Generated Image")
    Image.fromarray(output).save('image.jpg')
    # print(output.shape)
    st.session_state.var.output = output
    st.session_state.trigger.save_image = True
    # 테스트용
    # st.image(st.session_state.var.canvas_result.image_data[:, :, :3], "canvas_result")  

########### 상시 실행 코드 

if 'trigger' not in st.session_state:
    st.session_state.trigger=Trigger()

if 'var' not in st.session_state:
    st.session_state.var=Var()

st.title("Like a Bob Ross.")


# gpu / cpu mode 안 골랐을 때.
if (not st.session_state.trigger.gpu) and (not st.session_state.trigger.cpu):
    # if st.button("GPU 사용"):
    #     st.session_state.trigger.gpu = True
    if st.button("CPU 사용(샘플이미지)"):
        st.session_state.trigger.cpu = True


if st.session_state.trigger.gpu:
    uploaded_file = st.file_uploader("Upload the image you want to manipulate", type=["jpg", "jpeg", "png"])

    # 파일 업로드 될 경우 trigger on.
    if uploaded_file is not None:
        st.session_state.var.uploaded_file=uploaded_file
        st.session_state.trigger.upload=True 


if st.session_state.trigger.cpu:
   # 파일 업로드 될 경우 trigger on
    if st.button("Model 01"):
        st.session_state.var.uploaded_file ='required_data/Flickr/001.jpg'
        load_cpu_model(1)
    if st.button("Model 02"):
        st.session_state.var.uploaded_file ='required_data/Flickr/002.jpg'
        load_cpu_model(2)
    if st.button("Model 03"):
        st.session_state.var.uploaded_file ='required_data/Flickr/003.jpg'
        load_cpu_model(3)
    if st.button("Model 04"):
        st.session_state.var.uploaded_file ='required_data/Flickr/004.jpg'
        load_cpu_model(4)
        

if st.session_state.trigger.upload:

    canvas() # 항상 실행
   
    # original image는 항상 띄어놓기
    if st.session_state.trigger.gpu:
        
        if st.session_state.trigger.sa:
            if st.button("Start Adaptation"):

                adaptation_state=st.text("Adaptation...")
                model, output, x= start_adaptation(st.session_state.var.origin_image, st.session_state.var.origin_label_map,
                                                st.session_state.var.origin_shape, adaptation_state)
                adaptation_state.text("Done!")

                # 모델 & x 저장
                st.session_state.var.model=model
                st.session_state.var.x=x
                output = np.array(Image.fromarray(output).resize((st.session_state.var.width, st.session_state.var.height), resample=0))
                st.image(output, "Adapated Image")
                st.session_state.trigger.sa = False
        

    # Label을 기반으로 이미지 생성
    if st.session_state.trigger.gi:
        if st.button("Generate Image"):
            generate_output()
            st.session_state.trigger.gi=False


    # # session이 초기화 될 때 마다 output은 유지
    if st.session_state.var.output is not None:
        
        if not st.session_state.trigger.fg:
            # print('generate_output 도입 전.')
            # st.image(st.session_state.var.canvas_result.image_data, 'output2')
            generate_output()
        if st.session_state.trigger.fg:
            st.session_state.trigger.fg=False
        # st.image(st.session_state.var.output, "Generated Image")

    


    if st.button("End Paintings ! (click 2 times)"):
        st.session_state.trigger.canvas=False


    # if st.button('Save segmentation'):
    #     print(st.session_state.var.origin_label_map.shape)
    #     # assert max(st.session_state.vfar.origin_label_map)<2
    #     Image.fromarray(st.session_state.var.origin_label_map.astype(np.uint8)).save('./required_data/Flickr/004.png')

    # if st.button("Save Model(테스트용)"): 
    #     st.session_state.var.model.opt.checkpoint='./required_data'
    #     st.session_state.var.model.opt.name='Flickr'
    #     st.session_state.var.model.save(4)
    
    if st.session_state.trigger.save_image:
        with open("image.jpg", "rb") as fp:
            btn = st.download_button(
            label="Download Image",
            data=fp,
            file_name="blue-jay1.jpg",
            mime="image/jpeg"
        )
        