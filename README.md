# ITOMAI
Interactive Tools for Object Manipulation in Art Images

## 2021-11-21

create repo

## 2021-11-27
create branch: segmentation  
create dir: segmentation


---

# SPADE

![image](https://user-images.githubusercontent.com/71121461/145967478-5ae79b3e-eadb-4a88-abda-f51a5981af45.png)


실행 예시 : https://velog.io/@sjinu/Test-Time-Train-Landscape-Image-using-Flickr-Segmenter-Flickr-SPADE-2-2021-12-05  
(우선) 사용할 샘플 : https://velog.io/@sjinu/Sample-dataset-for-SPADE  


## 1. 설치


Clone this repo.
```bash
git clone https://github.com/sjinu96/ITOMAI
cd ITOMAI/SPADE/
```

Create env
```
conda create --name spade python=3.7 
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

*  `latest_net_D.pth, latest_net_G.pth` 의 파일 이름을 `50_net_D.pth, 50_net_G.pth`으로 바꾸는 것을 추천한다.
   * latest_net_D(G).pth는 학습할 때마다 새로운 모델로 대체되기 때문에 기존 모델이 사라짐.
   * 위와 같이 pth파일 명을 바꿨다면 추후에 `--which-epoch 50` 인자를 사용해서 학습과 테스트를 수행할 수 있다.  
* 또한, 학습을 새로 할 경우 iter.text, opt.pkl, opt.text <-- 이 3개의 파일은 항상 원본으로 유지시켜주어야 한다(꼬임).


## 3. 학습 (for single image)

### 3.1. 데이터 구조
코드 자체가 **폴더** 단위로 학습을 진행하기 때문에, 우선 single image Adaptation 또한 아래와 같은 이미지 경로가 필요하다.

|- val_img (original image - 3 channel)  
　|- 0001.jpg  
|- val_label (segmentation mask - 1channel)  
　|- 0001.png   
 

(변경 전)
> sample들은 `SPADE/gm_sample` folder에 업로드(지속적으로 추가 예정).
![image](https://user-images.githubusercontent.com/71121461/145966272-94f10a6f-00e3-4417-88f8-ad34408531db.png)

- image folder와 label folder 내의 이미지 이름은 같아야 함(위에서 0001).
- 이미지가 `val_img2`이라면,  이에 해당하는 original label map은 `val_label2`이고, 여러가지 변형을 준 label map은 `val_label2_1, val_label2_2, ...`등으로 네이밍
- 해당 Label map은 **coco-stuff164k 라벨에 맞게 정제해서 올려놓은 것.**(다른 Label map은 혼용 안 됨)
- 지속적으로 sample 만들어 업로드하겠음.


### 3.2. 학습 실행

이 때 아래와 같이 학습을 수행하면 된다.
```
$ python train.py --name Flickr --dataset_mode custom --label_dir gm_sample/train_labels --image_dir gm_sample/train_img --continue_train --load_from_opt_file --gpu_ids 0 --niter 300 --save_epoch_freq 150 --batchSize 1 --lr 0.0005 -preprocess_mode resize_and_crop --which_epoch 50 --display_freq 10
```

다른 건 필수적이지만, 아래는 선택이다.

`--label_dir` : 학습용 라벨이 있는 폴더경로  
`--image_dir` : 학습용 이미지가 있는 폴더경로  
`--niter 300` : 학습횟수(50epoch이 시작이므로 고려해서 사용,  1epoch = 1iter)  
`--save_epoch_freq` : 저장 빈도(50epoch이 시작이므로 고려해서 사용).   
`--lr 0.0005` : 학습률을 0.0005로 사용할 시 `--niter 150` 정도면 충분함(100번학습).  
`--which_epoch 50` : 위에서 pre-trained model의 이름을 `50_net_D(G).pth`로 바꿨을 경우, 이 모델의 parameter를 초기 parameter로 사용(사실상 필수).  
`--gpu_ids 0` : 학습 환경에 맞춰사용.  
`--display_frep 10` : (아마) 10 epoch마다 `checkpoints/Flickr/web/images/..`에 학습 추이 저장해줄 것. 



### 3.3. 학습 결과 시각화.


학습에 따라 특정 epoch마다 실행 결과가 'SPADE/checkpoints/Flickr/web/images'에 저장될 것.    
(model을 저장할 때마다 생성하는 것 같음 - `--save_epoch_freq 50` : 50 epoch마다 생성 결과 저장.).  

[gm_TestTimeTrainingTest.ipynb](https://github.com/sjinu96/ITOMAI/blob/main/SPADE/gm_TestTimeTrainingTest.ipynb)    
- 위 주피터노트북 파일의 중간을 보면 학습에 따른 결과 추이를 프린트하는 cell이 있긴 한데, 파일 더러우니까 그냥 직접 코드 짜거나 폴더 들어가서 보는 게 나을듯  

![image](https://user-images.githubusercontent.com/71121461/145965808-1e0a1fe1-19c2-4cc3-8b45-4b26ae03fc3a.png)




## 4. Test

```
%python test.py --name Flickr --dataset_mode custom --load_from_opt_file --gpu_ids -1 --which_epoch [사용할 model의 epoch] --image_dir gm_sample/test_img --label_dir gm_sample/test_labels
```

Test를 진행할 때에는 아래의 인자만 변경해주면 된다.  

`--which epoch` : 어떤 모델을 사용할지. ex) `--which epoch 150` : `150_net_G.pth` 모델 사용  
`--image_dir` : 이미지 폴더. ex) `SPADE/gm_sample/test3_img`  
`--label_dir` : 이미지 폴더에 맞는 라벨. ex) `SPADE/gm_TTTT/test3_labels`

실행 결과는 아래에 저장될 것.  
`SPADE/results/Flickr/train_[epoch]/images/input_label`  
`SPADE/results/Flickr/train_[epoch]/images/synthesized_image`  

### 4.1. 테스트 결과 시각화

[gm_Visualize.ipynb](https://github.com/sjinu96/ITOMAI/blob/main/SPADE/gm_Visualize.ipynb) 참고.  


![image](https://user-images.githubusercontent.com/71121461/145965533-d7342f4c-a420-4fcc-9792-7c8be1f4f684.png)



