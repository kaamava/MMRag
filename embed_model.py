# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：Dr Zhu, 微信: 15000361164
@version：
@Date：
@Desc: embedding model
"""

from typing import cast, List, Union, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, is_torch_npu_available


class EmbedModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 512,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.query_instruction_for_retrieval + queries
            else:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 512,
                      convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        return self.encode(corpus, batch_size=batch_size, max_length=max_length, convert_to_numpy=convert_to_numpy)

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 256,
               max_length: int = 512,
               convert_to_numpy: bool = True) -> np.ndarray:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        #print(f"输入文本数量: {len(sentences)}")
        #if len(sentences) == 0:
        #    raise ValueError("输入文本为空。")

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding="longest",
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            #print('all_embeddings长度为:', len(all_embeddings))

        if convert_to_numpy:
            #if not all_embeddings:
                #raise ValueError("all_embeddings 列表为空，无法进行连接操作。请检查数据处理逻辑。")
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.stack(all_embeddings)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
