# Instance Shadow Segmentation for Image Object Removal

![Example](example.png)

Our work is fundametally built on top of:
* [Instance Shadow Segmentation](https://github.com/stevewongv/InstanceShadowDetection)
* [Partial Convolutions for Image Inpainting using Keras](https://github.com/MathiasGruber/PConv-Keras)

By using a model which detects objects together with their respective shadows (LISA), we can improve on the object removal from images by removing the shadow associated with the object as well. For object removal, we employ a model which uses partial convolution to do image inpainting.

# Environment
The notebooks should handle their own enviroment setup for their respective tasks. Access to a GPU is assumed in both cases. The demo notebook is built and tested on Google Colab so all the library installations are done accordingly. The training notebook was run on a 2 GPU machine in order to simulate the original training process. For convenience of those who do not have access to such processing power, training can be done with `LISA-colab-power.yml`, but that setup might not produce as effective weights as the full training. Similarly to the demo, the notebook does not handle the installation of common libaries. Please refer to the two linked repos for more detailed explanations on the dependencies of each model.

* Files need to run the demo and the dataset for training can be found here: [comvis project](https://drive.google.com/drive/folders/11pal1EmQLp_1FND37bCpP3qUTIh_3jEr?usp=sharing)

# Datasets & Training
The LISA model is trained with the [SOBA](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP) (original link) dataset. The training notebook in `./notebooks` was used to train, and yield the weights used for the demo.

The PConv Model is trained on a modified version of ImageNet. However, we did not train the model ourselves since the inpainter is the not the focus of the project, and use pretrained weights also provided in the *comvis project* link above.


# Running Experiments
All our tests can be run through the `ComVis_Project_Demo.ipynb`. Sequentially run the cells following any instruction in the form of comments or text cells. We performed our tests on Google Colab. 

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

@inproceedings{liu2018partialpadding,
author = {Guilin Liu and Kevin J. Shih and Ting-Chun Wang and Fitsum A. Reda and Karan Sapra and Zhiding Yu and Andrew Tao and Bryan Catanzaro},
title = {Partial Convolution based Padding},
booktitle = {arXiv preprint arXiv:1811.11718},   
year = {2018},
}

@inproceedings{liu2018partialinpainting,
author = {Guilin Liu and Fitsum A. Reda and Kevin J. Shih and Ting-Chun Wang and Andrew Tao and Bryan Catanzaro},
title = {Image Inpainting for Irregular Holes Using Partial Convolutions},
booktitle = {The European Conference on Computer Vision (ECCV)},   
year = {2018},
}
```