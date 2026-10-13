import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_size = len(dataset_items)
    n_feats = dataset_items[0]["spectrogram"].shape[1]

    spectrogram_lengths = torch.tensor(
        [item["spectrogram"].shape[-1] for item in dataset_items], dtype=torch.long
    )
    text_encoded_lengths = torch.tensor(
        [item["text_encoded"].shape[-1] for item in dataset_items], dtype=torch.long
    )
    max_spec_len = int(spectrogram_lengths.max())
    max_text_len = int(text_encoded_lengths.max())
    spectrogram_batch = torch.zeros(batch_size, n_feats, max_spec_len)
    text_encoded_batch = torch.zeros(batch_size, max_text_len, dtype=torch.long)

    for i, item in enumerate(dataset_items):
        time = item["spectrogram"].shape[-1]
        spectrogram_batch[i, :, :time] = item["spectrogram"].squeeze(0)
        length = item["text_encoded"].shape[-1]
        text_encoded_batch[i, :length] = item["text_encoded"].squeeze(0).long()

    return {
        "spectrogram": spectrogram_batch,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": text_encoded_batch,
        "text_encoded_length": text_encoded_lengths,
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
        "audio": [item["audio"] for item in dataset_items],
    }
