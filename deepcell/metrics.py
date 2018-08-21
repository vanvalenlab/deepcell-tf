import numpy as np
import skimage.io
import skimage.measure
import matplotlib.pyplot as plt

PREDICTION = './wshed_pred.tif'
TRUTH = 'wshed_mask.tif'
WIN_SIZE = 15                    # if
IM_SIZE = 2048
ROUND_TO = 4
CROP_SIZE = 32
DICE_IOU_THRESH = 0.5
MERGE_IOU_THRESH = 0.01

# reads images into ndarrays, and trims them to fit each other.
def im_prep(prediction, mask, win_size):
#    prediction = skimage.io.imread(str_pred)
#    mask = skimage.io.imread(str_mask)

    # if trimming not needed, return
    if win_size == 0 or prediction.shape == mask.shape:
        return prediction, mask

    # otherwise, pad prediction to imsize and zero out the outer layer of mask
    else:
        mask = mask[win_size:-win_size, win_size:-win_size]
        mask = np.pad(mask, (win_size, win_size), 'constant')
        prediction = np.pad(prediction, (win_size, win_size), 'constant')

    return prediction, mask

# calculates pixel-based dice and jaccard scores, and prints them out
def stats_pixelbased(pred_input, truth_input):

    pred = np.copy(pred_input)
    truth = np.copy(truth_input)

    pred[pred != 0] = 1
    truth[truth != 0] = 1

    if pred.shape != truth.shape:
        raise ValueError('shape of inputs need to match. Shape of prediction is: ', pred.shape, ' Shape of mask is: ', truth.shape)

    if((pred.sum() + truth.sum()) == 0):
        print('arrays are empty. Dice score is technically 1.0 but something aint right')
        return 1.0

    intersection = np.logical_and(pred, truth)

    dice_pixel = (2*intersection.sum() / (pred.sum() + truth.sum()))
    jaccard_pixel = dice_pixel/(2-dice_pixel)

    false_pos = np.sum( np.logical_or(pred, truth) & np.logical_not(truth) )
    false_neg = np.sum( np.logical_or(pred, truth) & np.logical_not(pred) )

    false_pos_perc_err = false_pos/(false_pos+false_neg)
    false_neg_perc_err = false_neg/(false_pos+false_neg)

    false_pos_perc_pred = false_pos/np.count_nonzero(pred)
    false_neg_perc_truth = false_neg/np.count_nonzero(truth)

    dice_pixel = round(dice_pixel, ROUND_TO)
    jaccard_pixel = round(jaccard_pixel, ROUND_TO)
    false_pos = round(false_pos, ROUND_TO)
    false_neg = round(false_neg, ROUND_TO)
    false_pos_perc_err = round(false_pos_perc_err, ROUND_TO)
    false_neg_perc_err = round(false_neg_perc_err, ROUND_TO)
    false_pos_perc_pred = round(false_pos_perc_pred, ROUND_TO)
    false_neg_perc_truth = round(false_neg_perc_truth, ROUND_TO)

    print('')
    print('____________________Pixel-based statistics____________________')
    print('')
    print('dice/F1 index:', dice_pixel)
    print('jaccard index:', jaccard_pixel)

    print('#false positives:', false_pos, '  % of total error:', false_pos_perc_err, '  % of predicted incorrect:', false_pos_perc_pred)
    print('#false negatives:', false_neg, '  % of total error:', false_neg_perc_err, '  % of ground truth missed:', false_neg_perc_truth)
    print('')
    print('')

# Calculate all Intersection over Union values within the cropped input.
# If values are > a threshold, mark them as a hit.
def calc_cropped_ious(crop_pred, crop_truth, threshold, iou_matrix):
    # for each unique cellID in the given mask...
    for n in np.unique(crop_truth):
        if n == 0: continue #excluding background

        # for each unique cellID in the given prediction...
        for m in np.unique(crop_pred):
            if m == 0: continue #excluding backgrond

            #calculate the intersection over union for
            intersection = np.logical_and( crop_pred==m, crop_truth==n)
            union = np.logical_or( crop_pred==m, crop_truth==n)
            iou = np.sum(intersection) / np.sum(union)

            if iou > threshold:
            #    print('Hit!')
                iou_matrix[n-1][m-1] = 1


def gen_iou_matrix_quick(pred, truth, threshold, crop_size, im_size=IM_SIZE):

    # label ground truth masks, neccesary if not already tagged with cellID numbers
    truth = skimage.measure.label(truth, connectivity = 1)

    # create empty intersection over union matrix, with shape n(truth) by m(prediction)
    iou_matrix = np.zeros((truth.max(), pred.max()))
    crop_counter = 1

    # crop input images and calculate the iou's for the cells present
    for x in range(0, im_size, crop_size):
        for y in range(0, im_size, crop_size):
#            print('crop # is: ', crop_counter)
#            crop_counter+=1
            crop = np.zeros((crop_size, crop_size))
            crop_pred = pred[x:x+crop_size, y:y+crop_size]
            crop_truth = truth[x:x+crop_size, y:y+crop_size]
            plt.imshow(crop_pred)
            calc_cropped_ious(crop_pred, crop_truth, threshold, iou_matrix)
    return iou_matrix


def dice_jaccard_object(pred, truth, threshold=DICE_IOU_THRESH, crop_size=256):

    iou_matrix = gen_iou_matrix_quick(pred, truth, threshold, crop_size=crop_size)
    iou_sum = np.sum(iou_matrix)
    pred_max = iou_matrix.shape[1]-1
    truth_max = iou_matrix.shape[0]-1


    dice_object = 2*iou_sum / (2*iou_sum + pred_max-iou_sum + truth_max-iou_sum)
    jaccard_object = dice_object/(2-dice_object)

#    print('object-based dice/F1 score is: ', dice_object)
    return iou_matrix, dice_object, jaccard_object

def count_false_pos_neg(iou_matrix):

    ## Count the number of cellID's in the ground truth mask without a corresponding prediction
    false_neg = 0

    # for each ground truth cellID
    for n in range(0, iou_matrix.shape[0]):
        counter = 0

        # check all masks
        for m in range(0, iou_matrix.shape[1]):

            # if any of the mask predictions match the cellID, move on to the next cell
            if iou_matrix[n,m] == 1:
                counter+=1
                continue

        # Otherwise, if no matches are found, then a false negative has occurred.
        if counter == 0:
            false_neg += 1

    ## Count the number of predicted masks without a corresponding ground-truth cell
    false_pos = 0

    # for each predicted cell
    for m in range(0, iou_matrix.shape[1]):
        counter = 0

        # check all ground truth cells
        for n in range(1, iou_matrix.shape[0]):

            # if any of the ground truth cells match the predicted mask, move on to the next
            if iou_matrix[n,m] == 1:
                counter+=1
                continue

        # Otherwise, if no matches are found, then a false positive has occured
        if counter == 0:
            false_pos += 1

    return false_pos, false_neg



def count_merg_div(iou_matrix):

    # for each unique cell in the ground truth mask, count the number of overlapping predicted masks.
    # every predicted mask beyond the first represents an incorrect division.
    divided = 0
    for n in range(0, iou_matrix.shape[0]):
        counter = 0
        for m in range(0, iou_matrix.shape[1]):
            if iou_matrix[n,m] == 1:
                counter+=1
        if counter > 1:
            divided += (counter-1)

    # for each predicted mask, count the # of overlapping cells in the ground truth.
    # every overlapping cell beyond the first represents an incorrect merge
    merged = 0
    for m in range(0, iou_matrix.shape[1]):
        counter = 0
        for n in range(1, iou_matrix.shape[0]):
            if iou_matrix[n,m] == 1:
                counter+=1
        if counter > 1:
            merged += (counter-1)

    return merged, divided

def stats_objectbased(pred_input, truth_input):

    # copy inputs so original arrays are not modified
    wshed_pred = np.copy(pred_input)
    wshed_truth = np.copy(truth_input)

    stats_iou_matrix, dice_object, jaccard_object = dice_jaccard_object(wshed_pred, wshed_truth, crop_size=CROP_SIZE)
    false_pos, false_neg = count_false_pos_neg(stats_iou_matrix)

    false_pos_perc_err = false_pos/(false_pos+false_neg)
    false_neg_perc_err = false_neg/(false_pos+false_neg)

    false_pos_perc_pred = false_pos/stats_iou_matrix.shape[1]
    false_neg_perc_truth = false_neg/stats_iou_matrix.shape[0]

    merge_div_iou_matrix = gen_iou_matrix_quick(wshed_pred, wshed_truth, threshold=MERGE_IOU_THRESH, crop_size=CROP_SIZE)
    merged, divided = count_merg_div(merge_div_iou_matrix)

    perc_merged = merged/stats_iou_matrix.shape[0]
    perc_divided = divided/stats_iou_matrix.shape[0]

    # round all print percentages to a given limit
    dice_object = round(dice_object, ROUND_TO)
    jaccard_object = round(jaccard_object, ROUND_TO)
    false_pos = round(false_pos, ROUND_TO)
    false_neg = round(false_neg, ROUND_TO)
    false_pos_perc_err = round(false_pos_perc_err, ROUND_TO)
    false_neg_perc_err = round(false_neg_perc_err, ROUND_TO)
    false_pos_perc_pred = round(false_pos_perc_pred, ROUND_TO)
    false_neg_perc_truth = round(false_neg_perc_truth, ROUND_TO)
    perc_merged = round(perc_merged, ROUND_TO)
    perc_divided = round(perc_divided, ROUND_TO)

    print('____________________Object-based statistics____________________')
    print('')
    print('Intersection over Union thresholded at:', DICE_IOU_THRESH)
    print('dice/F1 index:', dice_object)
    print('jaccard index:', jaccard_object)

    print('#false positives:', false_pos, '  % of total error:', false_pos_perc_err, '  % of predicted incorrect:', false_pos_perc_pred)
    print('#false negatives:', false_neg, '  % of total error:', false_neg_perc_err, '  % of ground truth missed:', false_neg_perc_truth)

    print('')
    print('Intersection over Union thresholded at:', MERGE_IOU_THRESH)
    print('#incorrect merges:', merged, '     % of ground truth merged:', perc_merged)
    print('#incorrect divisions:', divided, '  % of ground truth divided:', perc_divided)
    print('')


#if __name__ == '__main__':
#    wshed_pred, wshed_truth = im_prep(PREDICTION, TRUTH, WIN_SIZE)
#    stats_pixelbased(wshed_pred, wshed_truth)
#    stats_objectbased(wshed_pred, wshed_truth)
