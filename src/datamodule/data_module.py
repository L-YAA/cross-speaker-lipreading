import os

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AudioTransform, VideoTransform


# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out

def add(d1, d2):
    cummulative_size = int(d1.cummulative_sizes[-1]) + len(d2.list)
    d1.cummulative_sizes.append(cummulative_size)
    # cumulative_size = int(d1.cumulative_sizes[1]) + len(d2.list)
    # d1.cumulative_sizes.append(cumulative_size)
    d1.datasets.append(d2)
    return d1

class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        # todo
        # gpusbug
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        # 这行代码改变了cfg使用的gpu
        # self.cfg.gpus = torch.cuda.device_count()
        #判断gpus参数是list还是int，改变total_gpus
        # if isinstance(self.cfg.gpus, int):
        #     self.total_gpus = self.cfg.trainer.gpus * self.cfg.trainer.num_nodes
        # else:
        #     self.total_gpus = len(self.cfg.trainer.gpus) * self.cfg.trainer.num_nodes
    def _dataloader(self, ds, sampler, collate_fn):
        return torch.utils.data.DataLoader(
            ds,
            num_workers=12,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        ds_args = self.cfg.data.dataset
        train_ds = AVDataset(
            root_dir=ds_args.root_dirs[0],
            label_path=os.path.join(
                ds_args.root_dirs[0], ds_args.label_dir, ds_args.train_files[0]
            ),
            subset="train",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train")
        )
        flag = 0
        for root_dir,train_file in zip(ds_args.root_dirs[1:],ds_args.train_files[1:]):
            if flag == 0:
                train_ds += AVDataset(
                    root_dir=root_dir,
                    label_path=os.path.join(
                        root_dir, ds_args.label_dir, train_file
                    ),
                    subset="train",
                    modality=self.cfg.data.modality,
                    audio_transform=AudioTransform("train"),
                    video_transform=VideoTransform("train")
                )
                flag = 1
            else:
                av_dataset = AVDataset(
                    root_dir=root_dir,
                    label_path=os.path.join(
                        root_dir, ds_args.label_dir, train_file
                    ),
                    subset="train",
                    modality=self.cfg.data.modality,
                    audio_transform=AudioTransform("train"),
                    video_transform=VideoTransform("train")
                )
                train_ds = add(train_ds, av_dataset)

        sampler = ByFrameCountSampler(train_ds, self.cfg.data.max_frames)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)
        return self._dataloader(train_ds, sampler, collate_pad)

    def val_dataloader(self):
        ds_args = self.cfg.data.dataset
        val_ds = AVDataset(
            root_dir=ds_args.root_dirs[0],
            label_path=os.path.join(ds_args.root_dirs[0], ds_args.label_dir, ds_args.val_files[0]),
            subset="val",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("val", snr_target=self.cfg.decode.snr_target),
            video_transform=VideoTransform("val")
        )
        flag = 0
        for root_dir,val_file in zip(ds_args.root_dirs[1:],ds_args.val_files[1:]):
            if flag == 0:
                val_ds += AVDataset(
                    root_dir=root_dir,
                    label_path=os.path.join(
                        root_dir, ds_args.label_dir, val_file
                    ),
                    subset="val",
                    modality=self.cfg.data.modality,
                    audio_transform=AudioTransform("val", snr_target=self.cfg.decode.snr_target),
                    video_transform=VideoTransform("val")
                )
                flag = 1
            else:
                av_dataset = AVDataset(
                    root_dir=root_dir,
                    label_path=os.path.join(
                        root_dir, ds_args.label_dir, val_file
                    ),
                    subset="val",
                    modality=self.cfg.data.modality,
                    audio_transform=AudioTransform("val", snr_target=self.cfg.decode.snr_target),
                    video_transform=VideoTransform("val"),
                )
                val_ds = add(val_ds, av_dataset)

        # sampler = ByFrameCountSampler(
        #     val_ds, self.cfg.data.max_frames_val, shuffle=False
        # )
        # if self.total_gpus > 1:
        #     sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        dataloader = torch.utils.data.DataLoader(val_ds, batch_size=None, num_workers=12)
        # return self._dataloader(val_ds, sampler, collate_pad)
        return dataloader

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset
        test_ds = AVDataset(
            root_dir=ds_args.root_dirs[0],
            label_path=os.path.join(ds_args.root_dirs[0], ds_args.label_dir, ds_args.test_files[0]),
            subset="test",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform(
                "test", snr_target=self.cfg.decode.snr_target
            ),
            video_transform=VideoTransform("test")
        )
        flag = 0
        for root_dir,test_file in zip(ds_args.root_dirs[1:],ds_args.test_files[1:]):
            if flag == 0:
                test_ds += AVDataset(
                    root_dir=root_dir,
                    label_path=os.path.join(
                        root_dir, ds_args.label_dir, test_file
                    ),
                    subset="test",
                    modality=self.cfg.data.modality,
                    audio_transform=AudioTransform(
                        "test", snr_target=self.cfg.decode.snr_target
                    ),
                    video_transform=VideoTransform("test")
                )
                flag = 1
            else:
                av_dataset = AVDataset(
                    root_dir=root_dir,
                    label_path=os.path.join(
                        root_dir, ds_args.label_dir, test_file
                    ),
                    subset="test",
                    modality=self.cfg.data.modality,
                    audio_transform=AudioTransform(
                        "test", snr_target=self.cfg.decode.snr_target
                    ),
                    video_transform=VideoTransform("test")
                )
                test_ds = add(test_ds, av_dataset)

        dataloader = torch.utils.data.DataLoader(test_ds, batch_size=None)
        #dataloader = torch.utils.data.DataLoader(test_ds, batch_size=None, shuffle=True)
        return dataloader
