from abc import ABC, abstractmethod


class Registrator(ABC):
    def __init__(self, backend):
        self.backend = backend

    @abstractmethod
    def register(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        pass

    @abstractmethod
    def transform(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        pass
