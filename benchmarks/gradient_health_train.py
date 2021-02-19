from hub import Dataset
from hub.utils import Timer
import torch
import tensorflow as tf

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)


def time_iter_pytorch(
    dataset_name="activeloop/mnist", batch_size=1, prefetch_factor=0, process=None
):

    dset = Dataset(dataset_name, cache=True, storage_cache=True, mode="r")

    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(transform=lambda x: (x['label_chexpert'])),
        batch_size=batch_size,
        # prefetch_factor=prefetch_factor,
        num_workers=16,
    )

    with Timer(
        f"{dataset_name} PyTorch prefetch {prefetch_factor:03} in batches of {batch_size:03}"
    ):
        for idx, batch in enumerate(loader):
            print(idx)
            if process is not None:
                process(idx)


def time_iter_tensorflow(
    dataset_name="activeloop/mnist", batch_size=1, prefetch_factor=0, process=None
):

    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode="r")

    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    with Timer(
        f"{dataset_name} TF prefetch {prefetch_factor:03} in batches of {batch_size:03}"
    ):
        for idx, batch in enumerate(loader):
            image = batch["image"]
            # label = batch["label"]
            if process is not None:
                process(idx, image)


time_iter_tensorflow(dataset_name='s3://snark-gradient-raw-data/output/ds3', batch_size=16, prefetch_factor=16)