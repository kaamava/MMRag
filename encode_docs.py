# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：Dr Zhu, 微信: 15000361164
@version：
@Date：
@Desc: 对文档进行编码
"""

import os

from tqdm import tqdm

import numpy as np

import sys
sys.path.append("./")
from io_utils import load_json, dump_json
from embed_model import EmbedModel


def embed_docs_from_folder(embed_model,
                           from_folder,
                           to_path_embed,
                           to_path_ids,
                           from_idx=0,
                           to_idx=1000000,
                           batch_size=1024,
                           max_length=512, save_batch_size=10000):

    list_contents = []
    list_doc_ids = []
    for root, _, files in os.walk(from_folder):
        for file_ in files[from_idx: to_idx]:
            file_path = os.path.join(root, file_)
            print(file_path)

            samples = load_json(file_path)
            for samp in samples:
                id_ = samp["id"]
                contents_ = samp["contents"]

                list_contents.append(contents_)
                list_doc_ids.append(id_)

    corpus_embeddings = embed_model.encode_corpus(
        list_contents,
        batch_size=batch_size,
        max_length=max_length
    )
    dim = corpus_embeddings.shape[-1]

    print(f"saving embeddings at {to_path_embed}...")
    memmap = np.memmap(
        to_path_embed,
        shape=corpus_embeddings.shape,
        mode="w+",
        dtype=corpus_embeddings.dtype
    )

    length = corpus_embeddings.shape[0]
    # add in batch
    if length > save_batch_size:
        for i in tqdm(range(0, length, save_batch_size),
                      leave=False,
                      desc="Saving Embeddings"):
            j = min(i + save_batch_size, length)
            memmap[i: j] = corpus_embeddings[i: j]
    else:
        memmap[:] = corpus_embeddings

    # 保存id
    dump_json(
        list_doc_ids,
        to_path_ids
    )


if __name__ == "__main__":
    embed_model = EmbedModel(
        model_name_or_path="resources/bge-base-en-v1.5",
        query_instruction_for_retrieval="",
        use_fp16=False
    )
    batch_size = 256

    # encode textbooks
    from_folder = "corpus/document"
    #to_path_embed = "embeddings/document.mp"
    to_path_embed = "embeddings/document.npy"
    to_path_ids = "embeddings/doc_ids.json"
    embed_docs_from_folder(
        embed_model,
        from_folder,
        to_path_embed,
        to_path_ids,
        from_idx=0,
        to_idx=100000,
        batch_size=batch_size,
        max_length=512,
        save_batch_size=10000
    )

'''  
    # encode statpearls
    from_folder = "corpus/statpearls/chunk"
    to_path_embed = "embeddings/statpearls.mp"
    to_path_ids = "embeddings/doc_ids_statpearls.json"
    embed_docs_from_folder(
        embed_model,
        from_folder,
        to_path_embed,
        to_path_ids,
        from_idx=0,
        to_idx=100000,
        batch_size=batch_size,
        max_length=512,
        save_batch_size=10000
    )
'''

