from abc import ABC, abstractmethod


class BrainExtractor(ABC):
    def __init__(self, backend):
        self.backend = backend

    @abstractmethod
    def extract(self, input_image, output_image, log_file, mode):
        pass
