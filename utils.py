import os
import cv2
import collections
import numpy as np
from skimage import color, img_as_float

mask_map = [('left_lung', 1), ('right_lung', 2)]


resize_width = 512
resize_height = 512


def fill_hole(mask):
    # print('mask in fill hole {}'.format(mask[:-1]))
    h, w = mask.shape[:2]
    mask_ret = np.zeros((h + 2, w + 2), np.uint8)
    mask_ret[1:-1, 1:-1] = mask

    # Floodfill from point (0, 0)
    mask_floodfill = mask.copy()
    # Give background pixel 255
    mask_floodfill = 255 - mask_floodfill
    # The flood area will have value 1
    cv2.floodFill(mask_floodfill, mask_ret, (0, 0), 1)
    # Pick out the flood area.
    flood_area = np.zeros(mask_ret.shape, np.uint8)
    flood_area[mask_ret == 1] = 255
    mask_floodfill = flood_area[1:-1, 1:-1]

    # Invert floodfilled image
    mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)

    # Combine the two images to get the foreground.
    return mask | mask_floodfill_inv


def clean_mask(mask):
    mask = fill_hole(mask)
    contours, _ = cv2.findContours(np.copy(mask), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)

    if len(contours) == 1:
        return mask, contours[0]

    max_area = -1
    idx = -1

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)

        if area > max_area:
            max_area = area
            idx = i

    new_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(new_mask, contours, idx, color=(255),
                     thickness=cv2.FILLED)

    return new_mask, contours[idx]


def process(mask, mask_map=mask_map):
    """Post process mask.
    """
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    for organ, idx in mask_map:
        bin_mask = np.zeros(mask.shape, dtype=np.uint8)
        bin_mask[mask == idx] = 255
        filled_mask = fill_hole(bin_mask)
        m, _ = clean_mask(filled_mask)
        new_mask[m == 255] = idx
    return new_mask


def get_border(contour):
    lower_right_corner = np.amax(contour, axis=0)
    upper_left_corner = np.amin(contour, axis=0)

    left, top = upper_left_corner[0]
    right, bot = lower_right_corner[0]

    return top, bot, left, right


def get_row_border(row):
    nonzero = np.nonzero(row)
    left = np.min(nonzero)
    right = np.max(nonzero)
    return left, right


def get_central(ind_left, ind_right, top):
    centrals = []
    for row in range(top + 1, top + 21):
        ll, lr = get_row_border(ind_left[row, :])
        rl, rr = get_row_border(ind_right[row, :])

        cen = (ll + rr) / 2
        centrals.append(cen)
        # print(cen)

    return int(np.mean(centrals))


def find_location(contour, extremum):
    # return absolute coordinates
    possible_locations = np.argwhere(contour == extremum)

    if possible_locations.shape[0] == 1:
        return (extremum, contour[possible_locations[0, 0]][0, 1])
    else:
        return (extremum, np.amax(contour[possible_locations[:, 0]][:, 0, 1]))


def extract_right_lung_corner(contour_right):
    """Extract right lung corner.

    :param contour_right: contour for the right lung mask
    :return: (width, height) in absolute value
    """
    # Find Convex Hull
    hull = cv2.convexHull(contour_right, returnPoints=True)

    vert_num = len(hull)

    # Compute Lung Corner
    left = hull[0][0][0]
    right = left
    for i in range(0, vert_num):
        left = hull[i][0][0] if hull[i][0][0] < left else left
        right = hull[i][0][0] if hull[i][0][0] > right else right

    threshold = left + (right - left) / 3

    res = [0, 0]
    for i in range(0, vert_num):
        if hull[i][0][0] > threshold:
            if res[0] == 0 and res[1] == 0:
                res = hull[i][0]
            elif res[1] < hull[i][0][1]:
                res = hull[i][0]
    return tuple(res)


def compute_ctr(left_lung, right_lung, alpha_1=0.5, alpha_2=0.8, beta_1=0.75,
                beta_2=0.85):
    ind_left = left_lung > 0
    ind_right = right_lung > 0

    left_lung, contour_left = clean_mask(left_lung)
    right_lung, contour_right = clean_mask(right_lung)

    top_l, bot_l, left_l, right_l = get_border(contour_left)
    top_r, bot_r, left_r, right_r = get_border(contour_right)

    left = left_r
    right = right_l

    top = max(top_l, top_r)
    bot = min(bot_l, bot_r)

    coor_lung_left = find_location(contour_right, left)
    coor_lung_right = find_location(contour_left, right)

    mask_left = ind_left.astype(np.int32)
    mask_right = ind_right.astype(np.int32)

    mask_left[np.nonzero(mask_left)] = 3
    mask_right[np.nonzero(mask_right)] = 2

    mask = mask_left + mask_right

    central = get_central(ind_left, ind_right, top)  # central line

    height = (bot - top + 1)

    left_row_max = 0
    right_row_max = 0
    row_left = 0
    row_right = 0
    right_loc = 0
    left_loc = 0

    for row in range(top + int(beta_1 * height), top + int(beta_2 * height)):
        # print(row)
        ll, lr = get_row_border(ind_left[row, :])
        dis_left = ll - central

        if dis_left < left_row_max:
            break

        left_row_max = dis_left
        row_left = row
        right_loc = ll

    right_lung_corner_column, right_lung_corner_row = extract_right_lung_corner(contour_right)
    # print(right_lung_corner_row, top + int(alpha_1 * height))
    for row in range(right_lung_corner_row - 4, top + int(alpha_1 * height), -1):
        rl, rr = get_row_border(ind_right[row, :])
        dis_right = central - rr

        if dis_right > right_row_max:
            right_row_max = dis_right
            row_right = row
            left_loc = rr

    coor_lung_left = coor_lung_left[0] * 1.0 / resize_width, coor_lung_left[1] * 1.0 / resize_height
    coor_lung_right = coor_lung_right[0] * 1.0 / resize_width, coor_lung_right[1] * 1.0 / resize_height
    coor_heart_right = (right_loc * 1.0 / resize_width, row_left * 1.0 / resize_height)
    coor_heart_left = (left_loc * 1.0 / resize_width, row_right * 1.0 / resize_height)

    od = collections.OrderedDict()
    od['lung_left'] = coor_lung_left
    od['lung_right'] = coor_lung_right
    od['heart_left'] = coor_heart_left
    od['heart_right'] = coor_heart_right

    ctr = (od['heart_right'][0] - od['heart_left'][0]) / (od['lung_right'][0] - od['lung_left'][0])
    return ctr


def add_seg(img, mask):
    img_color = np.dstack((img, img, img))

    # expand to 3 RGB channels.
    img_float = img.astype(np.float32) / 255
    img_float = img_as_float(img_float)

    color_mask = np.zeros((img_float.shape[0], img_float.shape[1], 3))

    color_map = [[0, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0.5, 1]]

    for i in range(1, len(color_map)):
        color_mask[mask == i] = color_map[i]

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # replace the hue and saturation of the original image
    # with that of the color mask
    alpha = 0.4
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv) * 256
    img_color = img_masked.astype(np.uint8)

    return img_color
