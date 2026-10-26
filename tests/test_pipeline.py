"""Pipeline tests on synthetic images -- no model needed.

The core claim under test: paste_back must reproduce the original
bit-for-bit everywhere outside the (guard-dilated) mask, even when the
'edit' is a hostile full-frame re-render that changes every pixel.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_edit.core import paste_back
from image_edit.masks import mask_from_boxes, mask_from_dark_region, mask_from_diff
from image_edit.verify import verify_outside


def make_original() -> np.ndarray:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (160, 200, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)  # pale background
    img[40:120, 120:180] = (10, 10, 10)  # a black "coat" block
    return img


class TestGuarantee(unittest.TestCase):
    def test_hostile_edit_cannot_leak_outside_mask(self):
        orig = make_original()
        rng = np.random.default_rng(99)
        edit = rng.integers(0, 255, orig.shape, dtype=np.uint8)  # changes ~everything
        mask = mask_from_boxes(orig.shape, [(550, 200, 950, 800)])
        final = paste_back(orig, edit, mask)
        report = verify_outside(orig, final, mask)
        self.assertTrue(report["guarantee_holds"], report)
        self.assertEqual(report["changed_outside_guard"], 0)
        self.assertEqual(report["max_diff_outside_guard"], 0)
        self.assertGreater(report["changed_inside_mask"], 0)

    def test_dark_region_traces_the_object_not_the_box(self):
        orig = make_original()
        mask = mask_from_dark_region(orig, (550, 200, 950, 800), luminance_threshold=95)
        self.assertGreater(int((mask > 0).sum()), 1000)
        # mask must not spill onto the pale background corner of the box
        self.assertEqual(int(mask[60, 150]), 255)  # inside the black block
        self.assertEqual(int(mask[35, 115]), 0)  # inside the box, but pale background
        report = verify_outside(orig, paste_back(orig, np.zeros_like(orig), mask), mask)
        self.assertTrue(report["guarantee_holds"], report)

    def test_diff_mask_ignores_drift_outside_regions(self):
        orig = make_original()
        edit = orig.copy()
        edit[60:80, 130:150] = (0, 0, 200)  # the intended change, inside the region
        edit[5:15, 5:15] = (0, 200, 0)  # hostile drift far outside any region
        mask = mask_from_diff(orig, edit, [(550, 200, 950, 800)])
        final = paste_back(orig, edit, mask)
        report = verify_outside(orig, final, mask)
        self.assertTrue(report["guarantee_holds"], report)
        # the drift patch must NOT be present in the composite
        self.assertTrue((final[5:15, 5:15] == orig[5:15, 5:15]).all())
        # the intended change must be present
        self.assertTrue((final[65:75, 135:145] != orig[65:75, 135:145]).any())


if __name__ == "__main__":
    unittest.main()
