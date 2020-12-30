from s3fs import S3FileSystem
from tqdm import tqdm

from hub.backend.storage import S3

chunk_size=20
path = f's3://snark-archive/snark-hub-main/imagenet/benchmark2/{str(chunk_size)}'

def get_items():
    storage = S3FileSystem()
    return storage.ls(path)


def test_old_speed():
    storage = S3(bucket='snark-archive')
    items = get_items()
    for item in tqdm(items):
        storage.get(item)


def test_new_speed():
    storage = S3FileSystem()
    items = get_items()
    for item in tqdm(items):
        storage.head(item, 5 * 1000 * 1000)
    

def main():
    test_old_speed()
    test_new_speed()

if __name__ == "__main__":
    main()