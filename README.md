## Yolact_minimal
This is the same implementation made [here](https://github.com/feiyuhuahuo/Yolact_minimal/) with just small adjustements to be used in Colab with a single GPU.

## Environments  
Works fine on Colab with T4 GPU.

## Prepare
1. Create a colab netbook and make sure to adjuste the ressources to GPU.
In a cell : 
```shell
git clone https://github.com/mohcenaouadj/SSL-YOLACT/
```
2. Install dependencies
```Shell
# Build cython-nms 
python setup.py build_ext --inplace
```
Also (if needed) : 
```shell
! pip install tensorboardX TensorRT terminaltables onnxruntime-gpu -q
```
3. Modify `self.data_root` in 'res101_coco' in `config.py` according to your data folder. 
4. Download weights.

Yolact trained weights.

|Backbone   | box mAP  | mask mAP | number of parameters | Google Drive                                                                                                             |
|:---------:|:--------:|:--------:|:--------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|PixPro-Resnet50   | 40.53     | 38.66     |       30.7 M        |[best_38.66_res50_pascal_12000.pth](https://drive.google.com/file/d/1H9u_unCJUWWGEc9J7E8szbDFh5KVl4gl/view?usp=sharing)  |
|VicRegL-Resnet50  | 28.55     | 29.89     |       30.7 M        |[best_29.89_res50_pascal_11000.pth](https://drive.google.com/file/d/1Gn-G9m80XuW9K-4Wz-M_n9hrzEskEzrK/view?usp=sharing)  |


5. Detect
   
![Example](https://github.com/mohcenaouadj/SSL-YOLACT/blob/master/readme_imgs/2011_002930.jpg)
  

```Shell
# To detect images, pass the path of the image folder, detected images will be saved in `results/images`.
python detect.py --best_38.66_res50_pascal_12000.pth --image=images
```
```Shell
# To detect videos, pass the path of video, detected video will be saved in `results/videos`:
python detect.py --weight=weights/best_38.66_res50_pascal_12000.pth --video=videos/1.mp4
# Use --real_time to detect real-timely.
python detect.py --weight=weights/best_38.66_res50_pascal_12000.pth --video=videos/1.mp4 --real_time
```
6. Use tensorboard
```Shell
tensorboard --logdir=tensorboard_log/res50_pascal
```
7. In case you want to retrain :

You can download the pascal segmentation bounderies dataset from [here](https://drive.google.com/drive/folders/155KEfQj93gYNWLBOXmevqipf2S_Nd3h_?usp=sharing)  

```shell
!torchrun --nproc_per_node=1 --master_port=$((RANDOM)) train.py --cfg=res50_pascal
```
Note that the configuration here is for pascal voc dataset, for more details you should check the original repo.  

## Citation
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

