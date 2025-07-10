from abc import ABC, abstractmethod
from typing import Any


class N4BiasCorrector(ABC):

    @abstractmethod
    def correct(
        self,
        input_img_path: Any,
        output_img_path: Any,
    ) -> None:
        pass
