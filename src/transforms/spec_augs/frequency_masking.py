import torchaudio
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)
