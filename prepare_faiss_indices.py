# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：Dr Zhu, 微信: 15000361164
@version：
@Date：
@Desc: faiss 向量索引建立
"""

import os
import sys
import time

import numpy as np
import faiss
from faiss import omp_set_num_threads, omp_get_max_threads
from faiss.contrib.ondisk import merge_ondisk

import sys

from tqdm import tqdm

sys.path.append("./")
from embed_model import EmbedModel

# os.environ["OMP_NUM_THREADS"] = "12"

# 注意使用 conda 安装 faiss-cpu
# conda install -c pytorch faiss-cpu=1.8.0

if __name__ == "__main__":

    corpus_name = sys.argv[1]  # "pubmed_0_40"
    mode = sys.argv[2]  # build & search

    # python src/faiss_ops/prepare_faiss_indices.py pubmed_0_25 build
    # python src/faiss_ops/prepare_faiss_indices.py pubmed_25_45 build
    # python src/faiss_ops/prepare_faiss_indices.py pubmed_45_65 build
    # python src/faiss_ops/prepare_faiss_indices.py pubmed_65_85 build
    # python src/faiss_ops/prepare_faiss_indices.py pubmed_85_100 build
    # python src/faiss_ops/prepare_faiss_indices.py pubmed_100_117 build

    # python src/faiss_ops/prepare_faiss_indices.py wikipedia_0_15 build
    # python src/faiss_ops/prepare_faiss_indices.py wikipedia_15_40 build
    # python src/faiss_ops/prepare_faiss_indices.py wikipedia_40_80 build
    # python src/faiss_ops/prepare_faiss_indices.py wikipedia_80_130 build
    # python src/faiss_ops/prepare_faiss_indices.py wikipedia_130_190 build
    # python src/faiss_ops/prepare_faiss_indices.py wikipedia_190_255 build
    # python src/faiss_ops/prepare_faiss_indices.py wikipedia_255_323 build

    # python src/faiss_ops/prepare_faiss_indices.py statpearls build
    # python src/faiss_ops/prepare_faiss_indices.py textbooks build

    model = EmbedModel(
        model_name_or_path="resources/bge-base-en-v1.5",
        query_instruction_for_retrieval="",
        use_fp16=True
    )

    if mode == "build":

        # index factgory
        # index_factory = "Flat"
        index_factory = "IVF4096,Flat"
        # index_factory = "Flat"
        # index_factory = "IVF4096"
        # index_factory = "IVF1024"
        # index_factory = "HNSW"
        # index_factory = "RFlat"

        # create faiss indexy
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)
        print("dim: ", dim)
        print("dtype: ", dtype)

        # corpus_name = "wikipedia_50_100"
        '''
        corpus_embeddings = np.memmap(
            f"embeddings/{corpus_name}.mp",
            mode="r",
            dtype=np.float32
        )
        '''
        corpus_embeddings = np.load(f"embeddings/{corpus_name}.npy")
        print("corpus_embeddings: ", corpus_embeddings.shape)
        corpus_embeddings = corpus_embeddings.reshape(-1, dim)
        print("corpus_embeddings: ", corpus_embeddings.shape)
        corpus_embeddings = corpus_embeddings.to(np.float32)
        print("corpus_embeddings: ", type(corpus_embeddings))

        faiss_index = faiss.index_factory(
            dim,
            index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        # co = faiss.GpuMultipleClonerOptions()
        # co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
        # # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
        # print("faiss on gpu!")

        print("training index")
        print(omp_get_max_threads())
        # omp_set_num_threads(6)

        # 注意要使用faiss-cpu
        faiss_index.train(corpus_embeddings)
        print("write " + f"embeddings/{corpus_name}.index")
        faiss.write_index(faiss_index, f"embeddings/{corpus_name}.index")

        # add data
        index = faiss.read_index(f"embeddings/{corpus_name}.index")
        print("adding vectors ")
        # index.add_with_ids(corpus_embeddings, np.arange(0, corpus_embeddings.shape[0]))
        index.add(corpus_embeddings)
        faiss.write_index(index, f"embeddings/{corpus_name}_block.index")

        # merge on disk
        index = faiss.read_index(f"embeddings/{corpus_name}.index")
        merge_ondisk(index,
                     [f"embeddings/{corpus_name}_block.index"],
                     f"embeddings/{corpus_name}_merged_index.ivfdata")
        faiss.write_index(index, f"embeddings/{corpus_name}_populated.index")

    else:


        # load index:
        index = faiss.read_index(f"embeddings/{corpus_name}_populated.index")
        index.nprobe = 16
        print("index: ", index)

        # try search
        query = "unhealthy food"
        test = model.encode(query).reshape(1, -1)
        # print(test)
        dtype = test.dtype
        dim = len(test)
        print("dim: ", dim)
        for _ in tqdm(range(10)):
            t0 = time.time()
            res = index.search(test, 16)
            t1 = time.time()
            print(res)
            print(t1 - t0)
            # print(res)
        # print(indices)

        # 注意要用conda编译的faiss-cpu
        # conda install -c pytorch faiss-cpu=1.8.0
        # python src/faiss_ops/prepare_faiss_indices.py