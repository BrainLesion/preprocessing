from .abstract_brain_extractor import BrainExtractor
from .bashhdbet import bash_hdbet_caller
from brainles_hd_bet import run_hd_bet


class HDBetExtractor(BrainExtractor):
    def __init__(self):
        super().__init__(backend="hdbet")

    def extract(self, input_image, output_image, log_file, mode):
        run_hd_bet(
            mri_fnames=[input_image],
            output_fnames=[output_image],
            mode="accurate",
            device=0,
            postprocess=False,
            do_tta=True,
            keep_mask=True,
            overwrite=True,
        )


class BashHDBetExtractor(BrainExtractor):
    def __init__(self):
        super().__init__(backend="bashhdbet")

    def extract(self, input_image, output_image, log_file, mode):
        bash_hdbet_caller(input_image, output_image, log_file, mode)
