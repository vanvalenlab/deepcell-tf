import numpy as np
from skimage.measure import label
from tensorflow.python.platform import test

from deepcell import metrics


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def _generate_test_masks():
    img_w = img_h = 30
    mask_images = []
    for _ in range(8):
        imarray = np.random.randint(2, size=(img_w, img_h, 1))
        mask_images.append(imarray)
    return mask_images


def _generate_stack_3d():
    img_w = img_h = 30
    imarray = np.random.randint(0, high=1, size=(40, img_w, img_h))
    return imarray


def _generate_stack_4d():
    img_w = img_h = 30
    imarray = np.random.randint(0, high=1, size=(40, img_w, img_h, 2))
    return imarray


class TransformUtilsTest(test.TestCase):
    def test_reshape_3d(self):
        stack = _generate_stack_3d()
        out = metrics.reshape_padded_tiled_2d(stack)

        # Expected width must accommodate padding
        exp_width = stack.shape[0] * (stack.shape[1] + 2)
        self.assertEqual(out.shape[2], exp_width)

    def test_reshape_4d(self):
        stack = _generate_stack_4d()
        out = metrics.reshape_padded_tiled_2d(stack)

        # Expected width must accommodate padding
        exp_width = stack.shape[0] * (stack.shape[1] + 2)
        self.assertEqual(out.shape[2], exp_width)

        # Check that channel dimension is preserved
        self.assertEqual(out.shape[-1], 2)

    def test_calc_objects_ious(self):
        y_true = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))
        y_pred = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))

        iou = metrics.calc_object_ious_fast(y_true, y_pred)

        # Check that output dimensions are 2d
        self.assertEqual(len(iou.shape), 2)

    def test_calc_cropped_ious(self):
        y_true = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))
        y_pred = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))

        iou_in = np.zeros((y_true.max(), y_pred.max()))
        iou_out = metrics.calc_cropped_ious(y_true, y_pred, 0.5, iou_in)

        # Input and output shape should be equal
        self.assertEqual(iou_in.shape, iou_out.shape)
        self.assertEqual(len(iou_out.shape), 2)

    def test_dice_jaccard_value(self):
        y_true = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))
        y_pred = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))

        iou = metrics.calc_object_ious_fast(y_true, y_pred)

        # Dice and jaccard values should be between 0 and 1
        d, j = metrics.get_dice_jaccard((iou > 0.5).astype('int'))
        # self.assertAllInRange([d, j], 0, 1)
        self.assertTrue((d >= 0) & (d <= 1))
        self.assertTrue((j >= 0) & (j <= 1))

    def test_2d_object_stats(self):
        y_true = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))
        y_pred = label(metrics.reshape_padded_tiled_2d(_generate_stack_3d()).astype('int'))
        iou = metrics.calc_object_ious_fast(y_true, y_pred)

        stats = metrics.calc_2d_object_stats((iou > 0.5).astype('int'))
        self.assertEqual(type(stats), dict)

        for k in stats.keys():
            self.assertFalse(np.isnan(stats[k]))

    def test_pixelstats_output(self):
        y_true = _get_image()
        y_pred = _get_image()

        out1 = metrics.stats_pixelbased(y_true, y_pred)
        self.assertEqual(type(out1), type(None))

        out2 = metrics.stats_pixelbased(y_true, y_pred, return_stats=True)
        self.assertEqual(type(out2), dict)

    def run_object_stats(self):
        y_true = label(_generate_stack_3d())
        y_pred = label(_generate_stack_3d())

        metrics.stats_objectbased(y_true, y_pred)
