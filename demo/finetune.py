from collections import OrderedDict
import data
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st

class Args:
    D_steps_per_G=1
    aspect_ratio=1.0
    batchSize=1
    beta1=0.5
    beta2=0.999
    cache_filelist_read=False
    cache_filelist_write=False
    checkpoints_dir='./required_data'
    contain_dontcare_label=False
    continue_train=True
    crop_size=512
    dataroot='./datasets/cityscapes/'
    dataset_mode='custom'
    debug=False
    display_freq=10
    display_winsize=256
    gan_mode='hinge'
    gpu_ids=[0]
    image_dir='gm_sample/train1_img'
    init_type='xavier'
    init_variance=0.02
    instance_dir=''
    isTrain=True
    label_dir='gm_sample/train1_labels'
    label_nc=182
    lambda_feat=10.0
    lambda_kld=0.05
    lambda_vgg=10.0
    load_from_opt_file=True
    load_size=512
    lr=0.0005
    max_dataset_size=9223372036854775807
    model='pix2pix'
    nThreads=0
    n_layers_D=3
    name='Flickr'
    ndf=64
    nef=16
    netD='multiscale'
    netD_subarch='n_layer'
    netG='spade'
    ngf=64
    niter=300
    niter_decay=0
    no_TTUR=False
    no_flip=False
    no_ganFeat_loss=False
    no_html=False
    no_instance=True
    no_pairing_check=False
    no_vgg_loss=False
    norm_D='spectralinstance'
    norm_E='spectralinstance'
    norm_G='spectralspadesyncbatch3x3'
    num_D=2
    num_upsampling_layers='normal'
    optimizer='adam'
    output_nc=3
    phase='train'
    preprocess_mode='resize_and_crop'
    print_freq=100
    save_epoch_freq=150
    save_latest_freq=5000
    semantic_nc=182
    serial_batches=False
    skip_losses=False
    tf_log=False
    use_vae=False
    which_epoch="latest"
    z_dim=256

def denorm(x):
    x = (x * 127.5) + 127.5
    x = x[0].permute(1, 2, 0).detach().cpu().numpy()
    x = x.astype(np.uint8)
    return x


def start_adaptation(image, label, origin_shape, state):
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    image = (image - 127.5) / 127.5
    image = F.interpolate(image, size = (512,512), mode = "bilinear", align_corners = True)
    opt = Args()
    
    # opt.gpu_ids=[]

    state.text('Load Model...')
    trainer = Pix2PixTrainer(opt)
    x = {"label" : torch.tensor(label).unsqueeze(0).unsqueeze(0).float(),
         "instance" : torch.tensor([0]),
         "image" : image,
         "path" : ""}

    epoch=130
    for i in range(epoch):
        state.text(f"Adaptation... ({round(i/90*100, 2)}%)")
        trainer.run_generator_one_step(x)
        trainer.run_discriminator_one_step(x)
        trainer.update_learning_rate(i)

    # x["label"] = torch.tensor(user_label).unsqueeze(0).unsqueeze(0).float()

    trainer.pix2pix_model.eval()
    output = trainer.pix2pix_model(x, "inference")

    # output = F.interpolate(output, size = (origin_shape[0], origin_shape[1]), mode = "bilinear", align_corners = True)
    output = denorm(output)

    return trainer, output, x


def generate_image(trainer, x, user_label, origin_shape):
    if st.session_state.trigger.cpu:
        # trainer.pix2pix_model.eval()
        x["label"] = torch.tensor(user_label).unsqueeze(0).unsqueeze(0).float()
        # # print(torch.tensor(x["image"].permute(2,0,1).shape))
        # image = torch.tensor(x["image"]).permute(2, 0, 1).unsqueeze(0).float()
        # image = (image - 127.5) / 127.5
        # image = F.interpolate(image, size = (512,512), mode = "bilinear", align_corners = True)
        # x["image"] = image 
        output = trainer(x, 'inference') # 사실 trainer가 아니라 pix2pix
        output = denorm(output)
    else:
        
        trainer.pix2pix_model.eval()
        x["label"] = torch.tensor(user_label).unsqueeze(0).unsqueeze(0).float()
        output = trainer.pix2pix_model(x, "inference")
        # st.session_state.trigger.cpu
        # reshape 잠시 생략
        # output = F.interpolate(output, size = (origin_shape[0], origin_shape[1]), mode = "bilinear", align_corners = True)
        output = denorm(output)

    return output
    

