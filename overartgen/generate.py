""" Script for the generation of an artificial dataset.

This script generates a new dataset with syntethic images. This images are formed by three
superposed ellipses. This scripts accepts multiples paremeters that defines the kind of elements to
generate.

100 45 610 610 45 85 3 2000

Arguments:
    path (str): output folder for the dataset.
    max_r (int): maximum size of the diameters of the ellipse. Default: 100
    min_r (int): minimum size of the diameters of the ellipse. Default: 45
    width (int): horizontal size of the image. Default: 610
    height (int): vertical size of the image. Default: 610
    min_dist (int): minimum distance between the ellipses. Default: 45
    max_dist (int): maximum distance between the ellipses. Default: 85
    number_cells (int): number of cells per image. Default: 3
    number_images (int): number of images to generate. Default: 2000

"""

import argparse
from typing import Tuple, Union
import os
import random

import numpy as np
import cv2
from tqdm import tqdm

random.seed(42)


def parse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("path", help="Folder to save the images")
    parser.add_argument("max_r", help="Maximum radius")
    parser.add_argument("min_r", help="Minimum radius")
    parser.add_argument("width")
    parser.add_argument("height")
    parser.add_argument("min_dist")
    parser.add_argument("max_dist")
    parser.add_argument("number_cells", help="Number of cells per image")
    parser.add_argument("number_images", help="Number of images to generate")

    return parser


def random_cell(max_r, min_r, shape: Tuple[int, int]):
    """
    Generates randomly the parameters of the ellipse.

    Args:
        max_r:
        min_r:
        shape:

    Returns:

    """
    rand1 = np.random.randint(min_r, max_r)
    rand2 = np.random.randint(min_r, max_r)

    a, b = max(rand1, rand2), min(rand1, rand2)
    phi = random.randint(0, 360)

    return {"shape": shape, "radius": (a, b), "phi": np.deg2rad([phi])[0]}


def set_centers(cells, rang_dist: Tuple[Union[int, float], Union[int, float]]):
    """ Defines the center of the cells depending on the constrains passed as parameters.

    The initial cell is set on the center of the image. The rest of cells are then located on the
    area that lies between the minimum and maximum distance from the center. This area is defined
    by the initial cell and the next addtions

    Args:
        cells: List of dictionaries [{"shape": shape, "radius": (a,b), "phi": phi }]
        rang_dist: Tuple that defines the constrains of the cell distance.

    Returns:

    """
    min_dist, max_dist = rang_dist
    width, height = cells[0]["shape"]

    cells[0]["center"] = int(width / 2), int(height / 2)

    checked = np.zeros((len(cells)))
    checked[0] = 2

    i = 1
    while checked.min() < 2:
        img_c = np.zeros((width, height)).astype(int)
        if checked[i] == 1:
            checked[i] = 0

        if checked[i] == 0:
            for j in range(1, len(cells)):
                pos_c = (j + i) % 3
                cell = cells[pos_c]
                if "center" in cells[pos_c]:
                    checked[i] += 1
                    if img_c.max() == 0:
                        img_c[cell["center"][0] + min_dist:cell["center"][0] + max_dist,
                        cell["center"][1] + min_dist:cell["center"][1] + max_dist] = 1
                    else:
                        aux_img = np.copy(img_c)
                        aux_img[cell["center"][0] + min_dist:cell["center"][0] + max_dist,
                        cell["center"][1] + min_dist:cell["center"][1] + max_dist] = 1
                        img_c = cv2.bitwise_and(img_c, aux_img)
            indexs = np.argwhere(img_c == 1)
            center = indexs[np.random.randint(0, len(indexs))]
            cells[i]["center"] = center

        i = (i + 1) % len(cells)

    return cells


def run():
    parser = parse_arguments(argparse.ArgumentParser())
    args = parser.parse_args()

    max_r = int(args.max_r)
    min_r = int(args.min_r)

    min_dist = int(args.min_dist)
    max_dist = int(args.max_dist)
    path = args.path

    height, width = int(args.height), int(args.width)

    n_images = int(args.number_images)
    n_cells = int(args.number_cells)

    s_cells = []
    for i in tqdm(range(n_images), desc="Image generation"):
        img = np.zeros((width, height, 3))
        cells = []
        for j in range(n_cells):
            cells.append(random_cell(max_r, min_r, (width, height)))
        set_centers(cells, (min_dist, max_dist))
        for c in cells:
            c_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            major_radius, minor_radius = c["radius"]
            center = c["center"]
            c_img = cv2.ellipse(c_img, center=(center[0], center[1]),
                                axes=(int(major_radius), int(minor_radius)),
                                angle=int(np.rad2deg(c['phi'])),
                                startAngle=0.0, endAngle=360, color=(1, 1, 1),
                                thickness=-1)

            img = img + c_img
        img[img > 1] = 1
        cv2.imwrite(os.path.join(path, str(i) + ".jpg"), img * 255)

        s_cells = s_cells + cells

    out = [np.array([c["center"][0], c["center"][1], max(c["radius"]), min(c["radius"]), c["phi"]])
           for c in s_cells]

    out = np.array(out)

    np.savetxt(os.path.join(path, "res.csv"), out, delimiter=",")


run()
