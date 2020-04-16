# deepcell.model_zoo

'deepcell-tf' features several model architectures for cell segmentation. All of the models in this library are `keras.models` and can easily fit into any `keras` deep learning workflow. These include:

* [FeatureNets](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005177) - This architecture was introduced in the first DeepCell paper and are quite effective for analyzing images with a single relevant length scale. This architecture is governed by a single control parameter, the receptive field, which controls the size of the spatial region around each pixel used to make predictions for that pixel. These models can perform instance segmentation by predicting a transform of the instance masks - e.g. whether pixels are edge/interior/background or a distance transform that can be used with watershed. While these are no longer the state-of-the-art, they're easy to train and are still effective - particularly for bacterial images. 

* PanopticNets - These architectures use [feature pyramid networks] for segmentation tasks. They consist of a backbone (e.g. resnet50), a feature pyramid, and one (or more) semantic head(s). We have found these to be very effective for cell segmentation. Our current approach to segmentation with this architecture borrows from the [DeepDistance](https://arxiv.org/abs/1908.11211) method as well as [location aware](https://www.nature.com/articles/s41598-017-05300-5) deep learning models. By including a location layer and predicting the inner (the distance of every pixel in a cell to the cell's center of mass) and outer distance (distance of every pixel in a cell to the background) transforms, these models can perform instance segmentation in a manner that is just as accurate as bounding box based approaches, but is faster and easier to deploy on large images.

* [RetinaNet](https://arxiv.org/abs/1708.02002) - These models find bounding boxes that encapsulate cells within an image. These models only return a bounding box and do not provide an instance mask. This is effectively a port of the [fizyr retinanet library](https://github.com/fizyr/keras-retinanet) that was adapted to work with tf.keras and offer greater control of image augmentation operations. These models include the option to attach a [semantic segmentation head](http://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.html) for panoptic segmentation tasks.

* [RetinaMask](https://arxiv.org/abs/1901.03353) - These models use retinanet to find bounding boxes and then predict instance masks within each bounding boxes. These models are essentially the same as the popular [MaskRCNN](https://arxiv.org/abs/1703.06870) method, except it uses RetinaNet to generatte the bounding boxes as opposed to FasterRCNN. This is effectively a port of the [fizyr maskrcnn library](https://github.com/fizyr/keras-maskrcnn). These models include the option to attach a semantic segmentation head.

* [Siamese Model](https://www.biorxiv.org/content/10.1101/803205v2) This architecture, contained within featurenet.py, is meant for cell tracking. This architecture, which was inspired by [previous](http://openaccess.thecvf.com/content_iccv_2017/html/Sadeghian_Tracking_the_Untrackable_ICCV_2017_paper.html) object tracking models, 




