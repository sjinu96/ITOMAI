
## 학습(1장의 이미지 - 1장의 라벨)


```
python train.py --image_dir gm_sample/train1_img --label_dir gm_sample/train1_label --lr 0.0005 --niter 150 --save_epoch_freq 150 --gpu_ids=0 --name Flickr --dataset_mode custom --continue_train --load_from_opt_file --batchSize 1 --preprocess_mode resize_and_crop --which_epoch 50
```

- epoch : 150 (update는 100번)
- lr : 0.0005
- gpu_ids : 0
- image_dir : gm_sample/train[1~4]_img
- label_dir : gm_sample/train[1~4]_labels
## 테스트(1장의 이미지 - 2~4장의 라벨)

```
python test.py  --image_dir gm_sample/test1_img --label_dir gm_sample/test1_labels --which_epoch 150 --name Flickr --dataset_mode custom --load_from_opt_file --gpu_ids -1 
```

- epoch : 150 (150_net_G.pth를 이용할 경우) 
- image_dir : gm_sample/test[1~4]_img
- label_dir : gm_sample/test[1~4]_labels
- 단, 당연히 gm_sample/test[번호] 는 train 할 때랑 동일하게 맞춰야 성능이 나올 것 


