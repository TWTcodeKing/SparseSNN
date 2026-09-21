"""
GLUE benchmark dataset loaders for SpikeBERT.

Supports SST-2 (sentiment analysis) and other GLUE tasks.
Uses HuggingFace `datasets` + `transformers` for tokenization.

Install: uv pip install datasets transformers
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader


class GLUEDataset(Dataset):
    """GLUE text classification dataset.

    Tokenizes text with BERT tokenizer and returns fixed-length token IDs.

    Returns:
        token_ids: (seq_len,) long tensor
        label: scalar long tensor
    """

    def __init__(self, task='sst2', split='train', seq_len=128,
                 tokenizer_name='bert-base-uncased', data_dir=None):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len

        # Load dataset
        task_map = {'sst2': ('glue', 'sst2'), 'mrpc': ('glue', 'mrpc'),
                    'cola': ('glue', 'cola'), 'qnli': ('glue', 'qnli')}
        ds_name, ds_config = task_map.get(task, ('glue', task))

        # Map split names
        split_map = {'train': 'train', 'val': 'validation', 'test': 'test'}
        hf_split = split_map.get(split, split)

        cache_dir = data_dir if data_dir else None
        ds = load_dataset(ds_name, ds_config, split=hf_split,
                          cache_dir=cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir)

        # Determine text columns
        if task == 'sst2':
            self.texts = ds['sentence']
        elif task in ('mrpc', 'qnli'):
            self.texts = [(row['sentence1'], row['sentence2']) for row in ds]
        else:
            # Fallback: use first string column
            cols = [c for c in ds.column_names if ds.features[c].dtype == 'string']
            self.texts = ds[cols[0]] if cols else ds['sentence']

        self.labels = ds['label']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if isinstance(text, tuple):
            encoding = self.tokenizer(
                text[0], text[1],
                max_length=self.seq_len, padding='max_length',
                truncation=True, return_tensors='pt')
        else:
            encoding = self.tokenizer(
                text, max_length=self.seq_len, padding='max_length',
                truncation=True, return_tensors='pt')

        token_ids = encoding['input_ids'].squeeze(0)  # (seq_len,)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return token_ids, label


def glue_dataloaders(data_root, batch_size, task='sst2', seq_len=128,
                     num_workers=4, distributed=False, **kwargs):
    """Build GLUE train/val dataloaders.

    Args:
        data_root: cache directory for HuggingFace datasets
        batch_size: batch size
        task: GLUE task name (sst2, mrpc, cola, qnli)
        seq_len: maximum sequence length
        num_workers: dataloader workers
        distributed: use DistributedSampler

    Returns:
        (train_loader, val_loader)
    """
    train_ds = GLUEDataset(task=task, split='train', seq_len=seq_len,
                           data_dir=data_root)
    val_ds = GLUEDataset(task=task, split='val', seq_len=seq_len,
                         data_dir=data_root)

    train_sampler = None
    val_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
