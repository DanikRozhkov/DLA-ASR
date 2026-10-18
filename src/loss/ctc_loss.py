import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def __init__(self, *args, zero_infinity=True, **kwargs):
        super().__init__(*args, zero_infinity=zero_infinity, **kwargs)

    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length.cpu(),
            target_lengths=text_encoded_length.cpu(),
        )

        return {"loss": loss}
