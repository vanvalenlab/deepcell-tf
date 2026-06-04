"""Tests for CytoplasmSegmentationModel"""


from tensorflow.python.platform import test
import numpy as np

from deepcell.model_zoo import PanopticNet
from deepcell.applications import CytoplasmSegmentation


class TestCytoplasmSegmentation(test.TestCase):

    def test_cytoplasm_app(self):
        with self.cached_session():
            model = PanopticNet(
                'resnet50',
                input_shape=(128, 128, 1),
                norm_method='whole_image',
                num_semantic_heads=2,
                num_semantic_classes=[1, 1],
                location=True,
                include_top=True,
                lite=True,
                use_imagenet=False,
                interpolation='bilinear')
            app = CytoplasmSegmentation(model)

            # test output shape
            shape = app.model.output_shape
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 2)
            self.assertEqual(len(shape[0]), 4)
            self.assertEqual(len(shape[1]), 4)

            # test predict
            x = np.random.rand(1, 500, 500, 1)
            y = app.predict(x)
            self.assertEqual(x.shape, y.shape)
