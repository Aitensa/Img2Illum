"""PointAR dataset generation code
"""
import os
import fire
import multiprocessing
from datasets.pointar.preprocess.pack import pack
from datasets.pointar.preprocess.generate import generate


def gen_data():


    print("Generating test dataset")
    generate(
        'init_test',  # dataset name ["train", "test"],
        index='all',  # data item, 'all' for all items, 0 for data item 0
    )

    print("Packing test dataset")
    pack('init_test', index="all")

    print("Generating training dataset")
    generate(
        "test",  # dataset name ["train", "test"],
        index='all',  # data item, 'all' for all items, 0 for data item 0
    )

    print("Packing training dataset")
    pack("test", index="all")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    fire.Fire(gen_data)
