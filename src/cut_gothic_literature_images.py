from typing import List
from glob import glob
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2 as cv

from scipy.stats import linregress
from skimage.measure import block_reduce
from scipy.signal import find_peaks


def rotate_and_cut_off(image: np.ndarray, angle: float, center: (int, int)) -> np.ndarray:
    """
        Function rotates an image and cuts off borders

        Args:
            image (np.array): tensor representing an image.
            angle (float): the angle of rotation is measured in degrees.
            center ((int, int)): center of rotation.
    """
    height, width = image.shape[:2]
    x, y = center

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(-height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t - width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    rotated = cv.warpAffine(image, M, (new_width, new_height))
    return rotated


def cut_horizontal_image(image: np.ndarray) -> List[np.ndarray]:

    x = np.mean(1 - image, axis=1) / np.mean(1 - image)
    kernel = image.shape[0] // 20
    x = np.convolve(x, np.ones(kernel) / kernel, mode='same')

    peaks, _ = find_peaks(x, height=0.75, prominence=0.3, width=image.shape[0] // 20)
    cut_indices = (peaks[1:] + peaks[:-1]) // 2
    cut_indices = np.insert(cut_indices, 0, 0)
    cut_indices = np.append(cut_indices, image.shape[0])

    single_line_images = []
    for i1, i2 in zip(cut_indices[:-1], cut_indices[1:]):
        single_line_images.append(image[i1: i2])

    return single_line_images


def bfs(img: List, x: int, y: int, used: List, h: int, w: int):
    q = set()
    q.add((x, y))

    while len(q) != 0:
        i, j = q.pop()

        used[i][j] = True

        if i > 0 and img[i - 1][j] and not used[i - 1][j]:
            q.add((i - 1, j))

        if i < h - 1 and img[i + 1][j] and not used[i + 1][j]:
            q.add((i + 1, j))

        if j > 0 and img[i][j - 1] and not used[i][j - 1]:
            q.add((i, j - 1))

        if j < w - 1 and img[i][j + 1] and not used[i][j + 1]:
            q.add((i, j + 1))


def cut_image_into_text_lines(image: np.array):
    img = (image < np.mean(image) - np.std(image)).astype(float)
    img = cv.dilate(img, np.ones((1, img.shape[1] // 10)))
    img = cv.erode(img, np.ones((img.shape[0] // 40, img.shape[1] // 40)))

    img = img.astype(bool)
    img = block_reduce(img, (3, 3), np.max)

    h, w = img.shape
    used = np.zeros((h, w), dtype=bool)
    used_list = used.tolist()
    img_list = img.tolist()

    slopes = []

    for x, y in zip(*np.where(~used & img)):

        if not used_list[x][y]:
            prev_used = used

            bfs(img_list, x, y, used_list, h, w)

            used = np.array(used_list)
            _x, _y = np.where(prev_used != used)

            if len(_x) > w * h // 100:
                res = linregress(_y, _x)
                slopes.append(res.slope)

    s = np.mean(slopes)
    s = np.arctan(s)

    rot_image = rotate_and_cut_off(image, s, (image.shape[1] // 2, image.shape[0] // 2))

    single_line_images = cut_horizontal_image(rot_image)

    return single_line_images


src_dir = '../dataset/old-degraded/'
dist_dir = Path('../dataset/old-degraded-single-line')
dist_dir.mkdir()

for p in tqdm(glob(src_dir + '*.png')):
    img = cv.imread(p, cv.IMREAD_GRAYSCALE)
    single_line_images = cut_image_into_text_lines(img)

    d = dist_dir / Path(p).stem
    d.mkdir(exist_ok=True)
    for i, x in enumerate(single_line_images):
        cv.imwrite(str(d / f'{i}.png'), x)
