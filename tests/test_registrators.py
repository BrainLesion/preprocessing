from pathlib import Path
import tempfile

import numpy as np
from registrator_base import RegistratorBase

from brainles_preprocessing.registration.ANTs.ANTs import ANTsRegistrator
from brainles_preprocessing.registration.niftyreg.niftyreg import NiftyRegRegistrator
from brainles_preprocessing.registration.elastix.elastix import ElastixRegistrator
from brainles_preprocessing.registration.greedy.greedy import GreedyRegistrator

import unittest


class TestANTsRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return ANTsRegistrator()

    def get_method_and_extension(self):
        return "ants", "mat"


class TestNiftyRegRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return NiftyRegRegistrator()

    def get_method_and_extension(self):
        return "niftyreg", "txt"

    def test_invert_affine_transform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Erstelle eine Testmatrix
            mat = np.array(
                [
                    [1, 2, 3, 4],
                    [0, 1, 4, 5],
                    [0, 0, 1, 6],
                    [0, 0, 0, 1],
                ]
            )
            path = Path(tmpdir) / "matrix.txt"
            output_path = Path(tmpdir) / "inverted.txt"

            np.savetxt(path, mat, fmt="%.12f")

            registrator = self.get_registrator()
            # Invertieren
            registrator._invert_affine_transform(path, output_path)

            # Ergebnis prüfen
            inverted_result = np.loadtxt(output_path)

            # Produkt aus Original * Inverse muss Identity ergeben
            identity = mat @ inverted_result
            np.testing.assert_array_almost_equal(identity, np.eye(4), decimal=5)

    def test_compose_affine_transforms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Zwei absichtlich nicht-kommutative Matrizen
            mat1 = np.array(
                [
                    [1, 2, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            mat2 = np.array(
                [
                    [1, 0, 3, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

            path1 = Path(tmpdir) / "t1.txt"
            path2 = Path(tmpdir) / "t2.txt"
            output_path = Path(tmpdir) / "composed.txt"

            np.savetxt(path1, mat1, fmt="%.12f")
            np.savetxt(path2, mat2, fmt="%.12f")

            # Compose (Erwartet: T2 @ T1)
            registrator = self.get_registrator()
            # Compose the transforms
            registrator._compose_affine_transforms([path1, path2], output_path)

            composed_result = np.loadtxt(output_path)

            expected = mat2 @ mat1  # Richtige Reihenfolge

            np.testing.assert_array_almost_equal(composed_result, expected, decimal=6)


class TestElastixRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return ElastixRegistrator()

    def get_method_and_extension(self):
        return "elastix", "txt"


class TestGreedyRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return GreedyRegistrator()

    def get_method_and_extension(self):
        return "greedy", "mat"
