class Registrator:
    def __init__(self, backend):
        self.backend = backend

    def register(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        raise NotImplementedError("Subclasses must implement the register method")

    def transform(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        raise NotImplementedError("Subclasses must implement the transform method")
