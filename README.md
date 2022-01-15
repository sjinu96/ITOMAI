# ITOMAI - Tobigs
Interactive Tools for Object Manipulation in Art Images

<img width="1394" alt="image" src="https://user-images.githubusercontent.com/71121461/149609803-4e5508f8-f5a2-44e1-bdfa-c116e9a66c24.png">

<img width="1394" alt="image" src="https://user-images.githubusercontent.com/71121461/149609805-46ba39e6-93dd-4d5c-8499-d80b36281224.png">

<img width="1393" alt="image" src="https://user-images.githubusercontent.com/71121461/149609812-4257c877-9cf2-451e-9ad5-7e15dd8e81dd.png">


---

# SPADE

## 1. Environments


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

## 2. Download Flickr-Landscape pre-trained SPADE 

https://drive.google.com/drive/folders/1NBgc1ziGOhG9RzoMQpftq8oetUzp_n9D?usp=sharing

위의 구글 드라이브에서 아래와 같이 7개의 파일을 받을 수 있다.   
Files for Train & Test  : `classes_list.text, iter.text, loss_log.text, opt.pkl, opt.text`  (총 5개)  
Pretrained model : `latest_net_D.pth, latest_net_G.pth` (총 2개)  


