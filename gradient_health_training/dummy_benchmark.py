
import hub
import numpy as np
import tensorflow as tf
import time


def only_frontal(sample):
    viewPosition = sample["viewPosition"].compute(True)
    return True if "PA" in viewPosition or "AP" in viewPosition else False


def get_image(viewPosition, images):
    for i, vp in enumerate(viewPosition):
        if vp in [5, 12]:
            return np.concatenate((images[i], images[i], images[i]), axis=2)


def to_model_fit(sample):
    viewPosition = sample["viewPosition"]
    images = sample["image"]
    image = tf.py_function(get_image, [viewPosition, images], tf.uint16)
    labels = sample["label_chexpert"]
    return image, labels

def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        x = 0
        ct = 0 
        for i, sample in enumerate(dataset):
            # for item in sample:
            #     x += item[0].shape[0]
            print(i)
            # x += sample["image"].shape[0]
            # print(i)
            # ct += 1
            # print("now sleeping")
            # time.sleep(5)
            # print("awake now")

            # Performing a training step
    # print(x/ct)
    print("Execution time:", time.perf_counter() - start_time)


batch_size = 8
ds = hub.Dataset("s3://snark-gradient-raw-data/output_all_attributes_2/ds3")

# 1743
# ds = hub.Dataset("s3://snark-gradient-raw-data/output_all_attributes_2500_samples_300_chunk/ds3")
dsv_train = ds[0:1500]
dsv_val = ds[1500:1864]
# dsv_val = ds[0:364]
dsf_train = dsv_train.filter(only_frontal)
dsf_val = dsv_val.filter(only_frontal)
tds_train = dsf_train.to_tensorflow()
tds_train = tds_train.map(to_model_fit)
tds_train = tds_train.batch(batch_size)
tds_val = dsf_val.to_tensorflow()
# tds_val = tds_val.map(to_model_fit)
# tds_val = tds_val.batch(batch_size)

benchmark(tds_val.prefetch(1000))
# benchmark(dsf_val)

# with tuning:- 
# 20.517092499998398
# 18.888743067000178
# 17.347827763005625
# Execution time: 17.043490725991433
