import torch
from hydra.utils import instantiate
from tqdm import tqdm

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def extract_log_probs(config, partition, device):
    """
    Run the acoustic model once (pure inference: eval(), no_grad(), no
    optimizer, no training loop) over a single dataset partition and
    return a list of {"log_probs": ndarray, "text": str} per sample
    """
    text_encoder = instantiate(config.text_encoder)
    dataset = instantiate(config.datasets[partition], text_encoder=text_encoder)
    loader = instantiate(
        config.dataloader,
        dataset=dataset,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        worker_init_fn=set_worker_seed,
    )

    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    checkpoint = torch.load(config.checkpoint_path, device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    samples = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {partition} log_probs"):
            spectrogram = batch["spectrogram"].to(device)
            spectrogram_length = batch["spectrogram_length"].to(device)
            outputs = model(
                spectrogram=spectrogram, spectrogram_length=spectrogram_length
            )
            log_probs = outputs["log_probs"].cpu().numpy()
            log_probs_length = outputs["log_probs_length"].cpu().numpy()
            for i in range(log_probs.shape[0]):
                length = int(log_probs_length[i])
                samples.append(
                    {
                        "log_probs": log_probs[i, :length].copy(),
                        "text": batch["text"][i],
                    }
                )
    return samples
