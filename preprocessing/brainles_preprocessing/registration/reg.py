from abc import ABC, abstractmethod, abstractproperty


class Registrator(ABC):
    @abstractproperty
    def backend(self):
        pass

    @abstractmethod
    def register(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        pass

    @abstractmethod
    def transform(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        pass
