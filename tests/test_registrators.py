from registrator_base import RegistratorBase

from brainles_preprocessing.registration.ANTs.ANTs import ANTsRegistrator
from brainles_preprocessing.registration.eReg.eReg import eRegRegistrator
from brainles_preprocessing.registration.niftyreg.niftyreg import NiftyRegRegistrator

import unittest


class TestANTsRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return ANTsRegistrator()

    def get_method_and_extension(self):
        return "ants", "mat"


class TestNiftyRegRegistratorRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return NiftyRegRegistrator()

    def get_method_and_extension(self):
        return "niftyreg", "txt"


class TestEregRegistratorRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return eRegRegistrator()

    def get_method_and_extension(self):
        return "ereg", "mat"
