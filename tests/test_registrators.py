from registrator_base import RegistratorBase

from brainles_preprocessing.registration.ANTs.ANTs import ANTsRegistrator
from brainles_preprocessing.registration.eReg.eReg import eRegRegistrator
from brainles_preprocessing.registration.niftyreg.niftyreg import NiftyRegRegistrator
from brainles_preprocessing.registration.elastix.elastix import elastixRegistrator
from brainles_preprocessing.registration.greedy.greedy import greedyRegistrator

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


class TestEregRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return eRegRegistrator()

    def get_method_and_extension(self):
        return "ereg", "mat"


class TestElastixRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return elastixRegistrator()

    def get_method_and_extension(self):
        return "elastix", "txt"


class TestGreedyRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return greedyRegistrator()

    def get_method_and_extension(self):
        return "greedy", "mat"
