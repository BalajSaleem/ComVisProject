# Instance Shadow Segmentation for Image Object Removal

Our work is fundametally built on top of:
* [Instance Shadow Segmentation](https://github.com/stevewongv/InstanceShadowDetection)
* [Image Inpainting via Generative Multi-column Convolutional Neural Networks](https://github.com/shepnerd/inpainting_gmcnn)

By using a model which detects objects together with their respective shadows (LISA), we can improve on the object removal from images by removing the shadow associated with the shadow as well. For object removel, we employ GM-CNN, a GAN which does image inpainting. 

# Enviroment
The notebook should handle the enviroment setup of the models. Access to a GPU is assumed. The code was written to configure a Google Colab instance so if you are running in your own machine you might miss some modules that Colab has by default. Please refer to the two linked repos for more detailed explanations on the dependencies of each model.

# Datasets & Training
The LISA model is trained with the [SOBA](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP) dataset. While the GMCNN Model is trained with pictures of places. This model has three different variations, but the `Places2` and `paris_streetview` are more appropriate for our use case (refer to the GM-CNN repo for more detail). 

We use pictures from the SOBA dataset to display our results, since they consist of objects with their images, which is convenient to display our project's capabilities.

The repos we linked go in greater detail on the procedures required to train the models. For the sake of convenience, our code uses the pretrained weights for both models in order to make predictions.

* [LISA Weights](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP)
* GM-CNN Weights: [Places2](https://drive.google.com/file/d/1wgesxSUfKGyPwGQMw6IXZ9GLeZ7YNQxu/view?usp=sharing), [paris_streetview](https://drive.google.com/file/d/1aakVS0CPML_Qg-PuXGE1Xaql96hNEKOU/view?usp=sharing)

# Running Experiments
All our tests can be run through the `Clean ComVis Project.ipynb`. Sequentially run the cells following any instruction in the form of comments or text cells. We performed our tests on Google Colab.

**Note:** Make sure to have the rest of the files of the repo in the same directory as your notebook. When working with Google Colab, upload these files in the local path (`./`) of the session. 

# Group Members
* Gledis Zeneli 
* Balaj Saleem

# Citations
```
@InProceedings{Wang_2020_CVPR,
author = {Wang, Tianyu and Hu, Xiaowei and Wang, Qiong and Heng, Pheng-Ann and Fu, Chi-Wing},
title = {Instance Shadow Detection},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

@inproceedings{wang2018image,
title={Image Inpainting via Generative Multi-column Convolutional Neural Networks},
author={Wang, Yi and Tao, Xin and Qi, Xiaojuan and Shen, Xiaoyong and Jia, Jiaya},
booktitle={Advances in Neural Information Processing Systems},
pages={331--340},
year={2018}
}
```