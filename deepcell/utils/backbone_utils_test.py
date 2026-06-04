"""Tests for backbone_utils"""


from absl.testing import parameterized

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras import keras_parameterized

from deepcell.utils import backbone_utils


class TestBackboneUtils(keras_parameterized.TestCase):

    @keras_parameterized.run_with_all_model_types
    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            data_format=[
                # 'channels_first',
                'channels_last']))
    def test_get_featurenet_backbone(self, data_format):
        backbone = 'featurenet'
        input_shape = (256, 256, 3)
        inputs = Input(shape=input_shape)
        with self.cached_session():
            K.set_image_data_format(data_format)
            model, output_dict = backbone_utils.get_backbone(
                backbone, inputs, return_dict=True)
            assert isinstance(output_dict, dict)
            assert all(k.startswith('C') for k in output_dict)
            assert isinstance(model, Model)

            # No imagenet weights for featurenet backbone
            with self.assertRaises(ValueError):
                backbone_utils.get_backbone(backbone, inputs, use_imagenet=True)

    # @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            data_format=[
                # 'channels_first',
                'channels_last']))
    def test_get_featurenet3d_backbone(self, data_format):
        backbone = 'featurenet3d'
        input_shape = (40, 256, 256, 3)
        inputs = Input(shape=input_shape)
        with self.cached_session():
            K.set_image_data_format(data_format)
            model, output_dict = backbone_utils.get_backbone(
                backbone, inputs, return_dict=True)
            assert isinstance(output_dict, dict)
            assert all(k.startswith('C') for k in output_dict)
            assert isinstance(model, Model)

            # No imagenet weights for featurenet backbone
            with self.assertRaises(ValueError):
                backbone_utils.get_backbone(backbone, inputs, use_imagenet=True)

    # @keras_parameterized.run_with_all_model_types
    # @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            backbone=[
                'resnet50',
                'resnet101',
                'resnet152',
                'resnet50v2',
                'resnet101v2',
                'resnet152v2',
                # 'resnext50',
                # 'resnext101',
                'vgg16',
                'vgg19',
                'densenet121',
                'densenet169',
                'densenet201',
                'mobilenet',
                'mobilenetv2',
                'efficientnetb0',
                'efficientnetb1',
                'efficientnetb2',
                'efficientnetb3',
                'efficientnetb4',
                'efficientnetb5',
                'efficientnetb6',
                'efficientnetb7',
                'nasnet_large',
                'nasnet_mobile']))
    def test_get_backbone(self, backbone):
        with self.cached_session():
            K.set_image_data_format('channels_last')
            inputs = Input(shape=(256, 256, 3))
            model, output_dict = backbone_utils.get_backbone(
                backbone, inputs, return_dict=True)
            assert isinstance(output_dict, dict)
            assert all(k.startswith('C') for k in output_dict)
            assert isinstance(model, Model)

    def test_invalid_backbone(self):
        inputs = Input(shape=(4, 2, 3))
        with self.assertRaises(ValueError):
            backbone_utils.get_backbone('bad', inputs, return_dict=True)


if __name__ == '__main__':
    test.main()
