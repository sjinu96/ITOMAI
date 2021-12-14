# ITOMAI
Interactive Tools for Object Manipulation in Art Images

## 2021-11-21

create repo

## 2021-11-27
create branch: segmentation  
create dir: segmentation




# Use SPADE


## 1. 설치

Clone this repo.
```bash
git clone https://github.com/sjinu96/ITOMAI
cd ITOMAI/SPADE/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

## 2. Flickr-Landscape pre-trained model 다운로드하기

https://drive.google.com/drive/folders/1NBgc1ziGOhG9RzoMQpftq8oetUzp_n9D?usp=sharing

위의 구글 드라이브에서 아래와 같이 7개의 파일을 받을 수 있다. 
Files for Train & Test  : `classes_list.text, iter.text, loss_log.text, opt.pkl, opt.text`  (총 5개)
Pretrained model : `latest_net_D.pth, latest_net_G.pth` (총 2개)

해당 repo의 `SPADE/checkpoints/Flickr`에 이미 5개의 파일은 올라가 있으며, 해당 경로에 latest_net_D.pth, latest_net_G.pth 파일만 넣어주면 학습과 테스트를 실행시킬 수 있다. 

-  `latest_net_D.pth, latest_net_G.pth` 의 파일 이름을 `50_net_D.pth, 50_net_G.pth`으로 바꾸는 것을 추천한다.
  - (학습할 때마다 latest_net_D(G).pth는 새로운 모델로 대체되므로, 50 epoch을 original pre-trained model로 사용)
  - 이는 이후에 `--which-epoch 50` 인자를 사용하면 학습과 테스트를 수행할 수 있다.  
- 또한, 학습을 새로 할 경우 iter.text, opt.pkl, opt.text <-- 이 3개의 파일은 항상 원본으로 유지시켜주어야 한다(꼬임).


## 3. 학습하는법 (for single image)

코드 자체가 **폴더** 단위로 학습을 진행하기 때문에, 우선 single image Adaptation 또한 아래와 같은 이미지 경로가 필요하다.

|- val_img (original image - 3 channel)
  |- 0001.jpg
|- val_label (segmentation mask - 1channel)
  |- 0001.png 


이 때 아래와 같이 학습을 수행하면 된다.
```
$ python train.py --name Flickr --dataset_mode custom --label_dir gm_TTTT/val_label --image_dir gm_TTTT/val_img --continue_train --load_from_opt_file --gpu_ids 0 --niter 300 --save_epoch_freq 150 --batchSize 1 --lr 0.0005 -preprocess_mode resize_and_crop --which_epoch 50 --print_freq 10
```

다른 건 필수적이지만, 아래는 선택이다.

`--label_dir` : 학습용 라벨이 있는 폴더경로
`--image_dir` : 학습용 이미지가 있는 폴더경로
`--niter 300` : 학습횟수(50epoch이 시작이므로 고려해서 사용,  1epoch = 1iter)
`--save_epoch_freq` : 저장 빈도(50epoch이 시작이므로 고려해서 사용).
`--lr 0.0005` : 학습률을 0.0005로 사용할 시 `--niter 150` 정도면 충분함(100번학습).
`--which_epoch 50` : 위에서 pre-trained model의 이름을 `50_net_D(G).pth`로 바꿨을 경우, 이 모델의 parameter를 초기 parameter로 사용(사실상 필수).
`--gpu_ids 0` : 학습 환경에 맞춰사용.
`--print_frep 10` : 사실 뭔지 기억이 잘 안남. 당장은 필요 없을듯




