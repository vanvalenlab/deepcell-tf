The augmented microscopy model uses the `retinanet_feature_extractor` model defined in `retinanet.py`, and is based off of retinanet.

Changes from DeepCell's segmentation retinanet:

- classification/regression heads removed and replaced with `default_regression_model` (in `retinanet.py`) which just condeses the z-dimension into 1 with convolutions and `tf.reduce_sum`. As a result, the output is a 2D image
- Backbone is the `featurenet_3D_backbone` in `backbone_utils.py`, except with the change that a full resolution backbone layer is included, in order to try to develop finer resolution features in the output. To save on computation and avoid out of memory, this full resolution layer actually was actually half resolution in the z-dimension, but this can be removed if needed.
- The pyramid is created through the method described in the original FPN paper, by upsampling the previous pyramid feature and then merging with the current backbone layer, rather than upsampling only the previous backbone layer and then merging with the current backbone layer, as it is done in the regular DeepCell retinanet. This option is enabled through the `fully_chained` parameter, and makes it so that each pyramid feature contains information from all lower resolution pyramid features. This was necessary to get good localization of the cells.
- Upsampling was changed from using `UpsampleLike` to `Conv3DTranspose`. This allows the upsampling to be learned; otherwise the output was very low resolution. Note that this only works because the featurenet_3D_backbone has each backbone layer be half the resolution of the previous backbone layer. This potentially creates off by one errors when the input size is odd, so a `UpsampleLike` layer is added at the end to fix the shape. Small note: for tensorflow versions < 1.12, the `use_bias` option on `Conv3DTranspose` layers must be set to `False` to work with `None` shapes.
- `UpsampleLike` for 3D inputs does not work for `None` shapes (see below), so in order to handle variable input sizes, reflection padding is used. This prodces slightly worse results when trained, however, for some reason. In `create_pyramid_am` in `fpn.py`, `variable_input` can be set to true to allow for dynamic shaped inputs. The default model was trained on a fixed 61 x 256 x 256 input size.
- Training is done using `train_model_am()` defined in `training.py`, which uses mean squared error.
- `StackDataGenerator` is the data generator used. An `identity` transform is defined in `image_generators.py` to perform no transformation, which is used for the augmented microscopy model.

Other changes from master:

- In `normalization.py`, `ImageNormalization3D` was changed to use a `K.constant` kernel instead of being a layer with weights. This created a bug for all 3D segmentation models that used the layer, since using non-trainable weights  would introduce a shape-dependent number of parameters in the model, which would prevent weights from being loaded across models with different input sizes. Also, the `kernel_shape` was changed to a constant size to allow for variable input sizes. 

- In `upsample.py`, `UpsampleLike.call`, resizing using the dynamic `K.shape()` did not work for 3D inputs, so it was changed to use the static .shape attribute instead. This makes it not work for dynamic input shapes; something else might fix this. 

Other notes:

- A regular 3D featurenet was tried, but it didn't produce the level of cell localization that the retinanet did.
- The model seems to be a bit unstable since different initial weights of the model produced relatively large variances in the accuracy of the trained model. Multiple training attempts were done to get a good result. 
- Fine resolution features still seem to mostly disappear by the end. A retinanet model with a featurenet backbone with dilated convolutions was trained too, in order to try and perseve more spatial information, but it didn't show better results.
- Currently, the z-stack is treated like a third dimension, but I also tried treating them like channels for a 2D input, but this produced worse results. 