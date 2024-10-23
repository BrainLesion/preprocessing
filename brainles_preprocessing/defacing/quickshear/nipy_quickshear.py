# Code adapted from: https://github.com/nipy/quickshear/blob/master/quickshear.py (23.10.2024)
# Minor adaptions in terms of parameters and return values
# Original Author': Copyright (c) 2011, Nakeisha Schimke. All rights reserved.

import argparse
import logging

#!/usr/bin/python
import sys

import nibabel as nb
import numpy as np
from numpy.typing import NDArray

try:
    from duecredit import BibTeX, due
except ImportError:
    # Adapted from
    # https://github.com/duecredit/duecredit/blob/2221bfd/duecredit/stub.py
    class InactiveDueCreditCollector:
        """Just a stub at the Collector which would not do anything"""

        def _donothing(self, *args, **kwargs):
            """Perform no good and no bad"""
            pass

        def dcite(self, *args, **kwargs):
            """If I could cite I would"""

            def nondecorating_decorator(func):
                return func

            return nondecorating_decorator

        cite = load = add = _donothing

        def __repr__(self):
            return self.__class__.__name__ + "()"

    due = InactiveDueCreditCollector()

    def BibTeX(*args, **kwargs):
        pass


citation_text = """@inproceedings{Schimke2011,
abstract = {Data sharing offers many benefits to the neuroscience research
community. It encourages collaboration and interorganizational research
efforts, enables reproducibility and peer review, and allows meta-analysis and
data reuse. However, protecting subject privacy and implementing HIPAA
compliance measures can be a burdensome task. For high resolution structural
neuroimages, subject privacy is threatened by the neuroimage itself, which can
contain enough facial features to re-identify an individual. To sufficiently
de-identify an individual, the neuroimage pixel data must also be removed.
Quickshear Defacing accomplishes this task by effectively shearing facial
features while preserving desirable brain tissue.},
address = {San Francisco},
author = {Schimke, Nakeisha and Hale, John},
booktitle = {Proceedings of the 2nd USENIX Conference on Health Security and Privacy},
title = {{Quickshear Defacing for Neuroimages}},
year = {2011},
month = sep
}
"""
# __version__ = "1.3.0.dev0"


def edge_mask(mask):
    """Find the edges of a mask or masked image

    Parameters
    ----------
    mask : 3D array
        Binary mask (or masked image) with axis orientation LPS or RPS, and the
        non-brain region set to 0

    Returns
    -------
    2D array
        Outline of sagittal profile (PS orientation) of mask
    """
    # Sagittal profile
    brain = mask.any(axis=0)

    # Simple edge detection
    edgemask = (
        4 * brain
        - np.roll(brain, 1, 0)
        - np.roll(brain, -1, 0)
        - np.roll(brain, 1, 1)
        - np.roll(brain, -1, 1)
        != 0
    )
    return edgemask.astype("uint8")


def convex_hull(brain):
    """Find the lower half of the convex hull of non-zero points

    Implements Andrew's monotone chain algorithm [0].

    [0] https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain

    Parameters
    ----------
    brain : 2D array
        2D array in PS axis ordering

    Returns
    -------
    (2, N) array
        Sequence of points in the lower half of the convex hull of brain
    """
    # convert brain to a list of points in an n x 2 matrix where n_i = (x,y)
    pts = np.vstack(np.nonzero(brain)).T

    def cross(o, a, b):
        return np.cross(a - o, b - o)

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    return np.array(lower).T


@due.dcite(
    BibTeX(citation_text),
    description="Geometric neuroimage defacer",
    path="quickshear",
)
def run_quickshear(bet_img: nb.nifti1.Nifti1Image, buffer: int = 10) -> NDArray:
    """Deface image using Quickshear algorithm

    Parameters
    ----------
    bet_img : Nifti1Image
        Nibabel image of skull-stripped brain mask or masked anatomical
    buffer : int
        Distance from mask to set shearing plane

    Returns
    -------
    defaced_mask: NDArray
        Defaced image mask
    """
    src_ornt = nb.io_orientation(bet_img.affine)
    tgt_ornt = nb.orientations.axcodes2ornt("RPS")
    to_RPS = nb.orientations.ornt_transform(src_ornt, tgt_ornt)
    from_RPS = nb.orientations.ornt_transform(tgt_ornt, src_ornt)

    mask_RPS = nb.orientations.apply_orientation(bet_img.dataobj, to_RPS)

    edgemask = edge_mask(mask_RPS)
    low = convex_hull(edgemask)
    xdiffs, ydiffs = np.diff(low)
    slope = ydiffs[0] / xdiffs[0]

    yint = low[1][0] - (low[0][0] * slope) - buffer
    ys = np.arange(0, mask_RPS.shape[2]) * slope + yint
    defaced_mask_RPS = np.ones(mask_RPS.shape, dtype="bool")

    for x, y in zip(np.nonzero(ys > 0)[0], ys.astype(int)):
        defaced_mask_RPS[:, x, :y] = 0

    defaced_mask = nb.orientations.apply_orientation(defaced_mask_RPS, from_RPS)

    # return anat_img.__class__(
    #     np.asanyarray(anat_img.dataobj) * defaced_mask,
    #     anat_img.affine,
    #     anat_img.header,
    # )

    return defaced_mask


# def main():
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     logger.addHandler(ch)

#     parser = argparse.ArgumentParser(
#         description="Quickshear defacing for neuroimages",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("anat_file", type=str, help="filename of neuroimage to deface")
#     parser.add_argument("mask_file", type=str, help="filename of brain mask")
#     parser.add_argument(
#         "defaced_file", type=str, help="filename of defaced output image"
#     )
#     parser.add_argument(
#         "buffer",
#         type=float,
#         nargs="?",
#         default=10.0,
#         help="buffer size (in voxels) between shearing plane and the brain",
#     )

#     opts = parser.parse_args()

#     anat_img = nb.load(opts.anat_file)
#     bet_img = nb.load(opts.mask_file)

#     if not (
#         anat_img.shape == bet_img.shape
#         and np.allclose(anat_img.affine, bet_img.affine)
#     ):
#         logger.warning(
#             "Anatomical and mask images do not have the same shape and affine."
#         )
#         return -1

#     new_anat = quickshear(anat_img, bet_img, opts.buffer)
#     new_anat.to_filename(opts.defaced_file)
#     logger.info(f"Defaced file: {opts.defaced_file}")


# if __name__ == "__main__":
#     sys.exit(main())
