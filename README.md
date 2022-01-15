# Interactive Tools for Object Manipulation in Art Images
> **ITOMAI: Tobigs 15&16 기 이미지 생성 프로젝트**
- 15기 : 김현지, 박진수, 오주영, 이윤정
- 16기 : 김경준

## IDEA
<img width="852" alt="image" src="https://user-images.githubusercontent.com/71121461/149619636-6ecadba0-119e-468b-95a7-9305d9ffdc68.png">


## RESULT
<img width="852" alt="image" src="https://user-images.githubusercontent.com/71121461/149609803-4e5508f8-f5a2-44e1-bdfa-c116e9a66c24.png">

## DEMO
https://user-images.githubusercontent.com/71121461/149610182-84c2af7b-b38a-42dd-8aed-853cd437e49c.mp4




---

# On your computer(참고용, 환경 테스트 후 재배포 예정)

## 1. Enrionments Settings

Clone this repo.
```bash
git clone https://github.com/sjinu96/ITOMAI
cd ITOMAI
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
cd SPADE/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../..
```

## 2. Download Flickr-Landscape pre-trained SPADE 

https://drive.google.com/drive/folders/1NBgc1ziGOhG9RzoMQpftq8oetUzp_n9D?usp=sharing

Files for Train & Test  : `classes_list.text, iter.text, loss_log.text, opt.pkl, opt.text`  (총 5개)  
Pretrained model : `latest_net_D.pth, latest_net_G.pth` (총 2개)  



## 3. Run Demo

```
cd demo
streamlit run demo.py
```



https://user-images.githubusercontent.com/71121461/149610078-7fc8bc80-a9d2-439f-a9f5-46021b36d751.mov

> enjoy ! 

# Other Example

https://user-images.githubusercontent.com/71121461/149610086-abff3dfc-7dcd-429a-80de-eed19371a07f.mp4



<img width="1394" alt="image" src="https://user-images.githubusercontent.com/71121461/149609805-46ba39e6-93dd-4d5c-8499-d80b36281224.png">

<img width="1393" alt="image" src="https://user-images.githubusercontent.com/71121461/149609812-4257c877-9cf2-451e-9ad5-7e15dd8e81dd.png">



