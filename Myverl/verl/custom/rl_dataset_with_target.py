# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd
import copy 

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from omegaconf import DictConfig, ListConfig
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length


import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output

from verl.utils.dataset.rl_dataset import RLHFDataset

class RLHFDatasetWithTarget(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 config: DictConfig,
                 target_key='target',
                 se_target_key='se_target',  # 新增: SE target 的列名
                 max_target_length=8192,
                 filter_targets=False,
                 sample_target_ratio=1.0,
                 target_list_key='target_lst',
                 max_num_targets=5,
                 target_probs_key='target_ds_qwen_7b_probs',
                 se_target_probs_key='se_logits',
                 se_prompt_key='se_prompt',  # 新增: 保存 SE prompt 的列名
                 use_se=False,
                 # ---- summarize-then-continue (explain-style) 新增 ----
                 use_summarize=False,
                 summarize_prompts_key='summarize_prompts',
                 summarize_prompt_key='summarize_prompt',
                 max_summarize_prompts=8,           # K, 与训练时 n_prefix 对齐
                 max_summarize_length=8192,         # rollout-time 长 prompt 容量
                 # ---- 离线 SR 候选（Data/aggregate_sr_responses.py 产出）----
                 sr_response_key='sr_response',
                 max_sr_response_length=0,          # 0 = 关闭；否则 = data.max_response_length
        ):
        super().__init__(parquet_files, tokenizer, config=config)

        self.max_target_length = max_target_length
        self.filter_targets = filter_targets
        self.target_key = target_key
        self.se_target_key = se_target_key  # 新增: 保存 SE target 的列名
        self.se_prompt_key = se_prompt_key  # 新增: 保存 SE prompt 的列名
        self.sample_target_ratio = sample_target_ratio
        self.target_list_key = target_list_key
        self.target_probs_key = target_probs_key
        self.se_target_probs_key = se_target_probs_key
        self.max_num_targets = max_num_targets
        self.use_se = use_se
        # ---- summarize-then-continue (explain-style) ----
        self.use_summarize = use_summarize
        self.summarize_prompts_key = summarize_prompts_key
        self.summarize_prompt_key = summarize_prompt_key
        self.max_summarize_prompts = max_summarize_prompts
        self.max_summarize_length = max_summarize_length
        # ---- 离线 SR 候选 ----
        self.sr_response_key = sr_response_key
        self.max_sr_response_length = max_sr_response_length
        if self.filter_targets:
            self._filter_targets()
    def _filter_targets(self):
        # 将需要的变量提取到局部作用域，以便在 filter 函数中使用
        tokenizer = self.tokenizer
        target_key = self.target_key
        
        def target2len(doc) -> int:
            tgt = doc.get(target_key)
            # 如果 target 不存在或为空，返回 0 长度（即保留该样本）
            if tgt is None or not isinstance(tgt, list) or len(tgt) == 0:
                return 0
            return len(tokenizer.apply_chat_template(doc[target_key], add_generation_prompt=True))
            # 获取第一个 target 的内容
            #content = tgt[0].get('content', '')
            # 计算 token 长度 (不添加 special tokens)
            #return len(tokenizer(content, add_special_tokens=False)['input_ids'])

        # 使用 datasets 库的高效 filter 方法
        self.dataframe = self.dataframe.filter(
            lambda doc: target2len(doc) <= self.max_target_length,
            num_proc=self.num_workers,  # 利用父类中定义的 num_workers 进行多进程处理
            desc=f"Filtering targets longer than {self.max_target_length} tokens",
        )
        print(f"filter dataset len: {len(self.dataframe)}")
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # 步骤 1: 直接调用父类的 __getitem__ 方法
        # 这是最核心的改动。我们让父类去处理所有关于 prompt 的复杂逻辑，
        # 包括多模态、应用聊天模板、分词、填充和计算 position_ids。
        # 这样，无论父类如何更新，我们都能自动享受到最新的 prompt 处理能力。
        # `super()` 返回的 `row_dict` 已经包含了 'input_ids', 'attention_mask', 'position_ids' 等。
        row_dict = super().__getitem__(item)

        

        # 步骤 2: 从原始数据中获取 target 相关信息
        # 因为父类可能已经从 `row_dict` 中 pop 了一些键，
        # 所以我们从最原始的数据源 `self.dataframe[item]` 中重新获取 target 相关字段。
        original_row: dict = self.dataframe[item]
        if self.use_se:
            if self.se_prompt_key in original_row:
                se_prompt_messages = original_row.pop(self.se_prompt_key)
                #print("se_prompt_messages:", se_prompt_messages)
                if se_prompt_messages:
                    # 1. 应用聊天模板，与父类处理标准 prompt 的方式保持一致
                    messages = se_prompt_messages
                    se_full_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # 2. 分词
                    se_input_ids = self.tokenizer(se_full_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

                    # 3. 填充或截断，与标准 prompt 使用相同的最大长度
                    if se_input_ids.shape[-1] < self.max_target_length:
                        se_input_ids = pad_sequence_to_length(
                            se_input_ids,
                            max_seq_len=self.max_target_length,
                            pad_token_id=self.tokenizer.pad_token_id,
                            left_pad=True  # prompt 通常进行左填充
                        )
                    else:
                        assert self.truncation in ('right', 'error')
                        se_input_ids = se_input_ids[:, :self.max_target_length]
                    
                    
                    # 4. 计算 attention_mask 和 position_ids
                    se_attention_mask = (se_input_ids != self.tokenizer.pad_token_id).to(se_input_ids.dtype)
                    se_position_ids = compute_position_id_with_mask(se_attention_mask)
                    # 5. 添加到 row_dict 中
                    row_dict['se_input_ids'] = se_input_ids.squeeze(0)
                    row_dict['se_attention_mask'] = se_attention_mask.squeeze(0)
                    row_dict['se_position_ids'] = se_position_ids.squeeze(0)
                else:
                    # 如果 se_prompt 为空，则创建与标准 prompt 形状相同的填充张量
                    row_dict['se_input_ids'] = torch.full_like(row_dict['input_ids'], self.tokenizer.pad_token_id)
                    row_dict['se_attention_mask'] = torch.zeros_like(row_dict['attention_mask'])
                    row_dict['se_position_ids'] = torch.zeros_like(row_dict['position_ids'])
                    # 5. 添加到 row_dict 中
            
            if self.se_target_key in original_row:
                se_target_messages = original_row.pop(self.se_target_key)
                if se_target_messages:
                    # 1. 应用聊天模板，与父类处理标准 prompt 的方式保持一致
                    se_tgt = se_target_messages[0]

                    
                    # 2. 分词
                    se_tgt_input_ids = self.tokenizer(se_tgt['content'], add_special_tokens=False, return_tensors='pt')['input_ids']

                    # 3. 填充或截断，与标准 prompt 使用相同的最大长度
                    if se_tgt_input_ids.shape[-1] < self.max_target_length:
                        se_tgt_input_ids = pad_sequence_to_length(
                            se_tgt_input_ids,
                            max_seq_len=self.max_target_length,
                            pad_token_id=self.tokenizer.pad_token_id,
                            left_pad=False  # target 通常进行右填充
                        )
                    else:
                        assert self.truncation in ('right', 'error')
                        se_tgt_input_ids = se_tgt_input_ids[:, :self.max_target_length]
                    
                    # 4. 添加到 row_dict 中
                    row_dict['se_tgt_input_ids'] = se_tgt_input_ids.squeeze(0)
                else:
                    # 如果 se_target 为空，则创建与标准 prompt 形状相同的填充张量
                    row_dict['se_tgt_input_ids'] = torch.full((self.max_target_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            # 步骤 3: 处理核心的 `target` 序列 (tgt_input_ids)
            if getattr(self, 'se_target_probs_key', "se_target_probs_key") in original_row:
                se_target_probs = original_row.get(self.se_target_probs_key)
                if se_target_probs is not None:
                    se_target_probs_pt = torch.tensor(se_target_probs, dtype=torch.float32)
                    se_target_probs_pt = se_target_probs_pt.reshape(1, -1)
                
                #se_tgt_len = (row_dict['tgt_input_ids'] != self.tokenizer.pad_token_id).sum()
                # 这里的断言可能需要根据实际数据微调
                # assert se_target_probs_pt.shape[-1] == se_tgt_len + 1

                if se_target_probs_pt.shape[-1] < self.max_target_length:
                    se_target_probs_pt = pad_sequence_to_length(se_target_probs_pt,
                                                             max_seq_len=self.max_target_length,
                                                             pad_token_id=-1,
                                                             left_pad=False)
                else:
                    assert self.truncation in ('right', 'error')
                    se_target_probs_pt = se_target_probs_pt[:, :self.max_target_length]
                row_dict['se_tgt_probs'] = se_target_probs_pt.squeeze(0)
            else:
                row_dict['se_tgt_probs'] = torch.full((self.max_target_length,), -1, dtype=torch.float32)

        # 这部分逻辑与你原来的代码几乎完全相同，因为这是子类的核心功能。
        tgt = original_row.get(self.target_key)
        sample = np.random.rand() < self.sample_target_ratio

        if tgt is not None and sample is True:
            tgt = tgt[0]
            
            # 获取父类处理好的、带模板的 prompt 字符串，用于后续逻辑判断
            prompt_with_chat_template = row_dict.get("full_prompts", "")

            if prompt_with_chat_template.endswith('<think>\n') and tgt['content'].startswith('<think>\n'):
                tgt['content'] = tgt['content'][len('<think>\n'):]
            
            tgt_input_ids = self.tokenizer(tgt['content'], add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1)
            tgt_input_ids = tgt_input_ids.reshape(1, -1)
        else:
            # 如果不采样 target，则创建一个空张量
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0)

        # 对 `tgt_input_ids` 进行填充或截断，逻辑保持不变
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                                   max_seq_len=self.max_target_length,
                                                   pad_token_id=self.tokenizer.pad_token_id,
                                                   left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        row_dict['tgt_input_ids'] = tgt_input_ids.squeeze(0)

        # 步骤 4: 处理 `target_list`，逻辑保持不变
        if getattr(self, 'target_list_key', "target_list_key") in original_row:
            target_list = original_row.get(self.target_list_key)
            prompt_with_chat_template = row_dict.get("full_prompts", "")
            if target_list is None:
                tgt_input_ids_lst = [torch.zeros_like(row_dict['tgt_input_ids']).fill_(self.tokenizer.pad_token_id)] * self.max_num_targets
            else:
                tgt_input_ids_lst = [self._process_target(tgt, prompt_with_chat_template, add_eos=True) for tgt in target_list]
                if len(tgt_input_ids_lst) <= self.max_num_targets:
                    tgt_input_ids_lst.extend([torch.zeros_like(tgt_input_ids_lst[0]).fill_(self.tokenizer.pad_token_id)] * (self.max_num_targets - len(tgt_input_ids_lst)))
                else:
                    tgt_input_ids_lst = tgt_input_ids_lst[:self.max_num_targets]
            row_dict['tgt_input_ids_lst'] = torch.stack(tgt_input_ids_lst, dim=0)

        # 步骤 5: 处理 `target_probs`，逻辑保持不变
        if getattr(self, 'target_probs_key', "target_probs_key") in original_row:
            target_probs = original_row.get(self.target_probs_key)
            if target_probs is not None:
                target_probs_pt = torch.tensor(target_probs, dtype=torch.float32)
                target_probs_pt = target_probs_pt.reshape(1, -1)
                
                tgt_len = (row_dict['tgt_input_ids'] != self.tokenizer.pad_token_id).sum()
                # 这里的断言可能需要根据实际数据微调
                # assert target_probs_pt.shape[-1] == tgt_len + 1

                if target_probs_pt.shape[-1] < self.max_target_length:
                    target_probs_pt = pad_sequence_to_length(target_probs_pt,
                                                             max_seq_len=self.max_target_length,
                                                             pad_token_id=-1,
                                                             left_pad=False)
                else:
                    assert self.truncation in ('right', 'error')
                    target_probs_pt = target_probs_pt[:, :self.max_target_length]
                row_dict['target_probs'] = target_probs_pt.squeeze(0)
            else:
                row_dict['target_probs'] = torch.zeros_like(row_dict['tgt_input_ids'], dtype=torch.float32).fill_(-1)

        # ---- summarize-then-continue (explain-style) 新增 ----
        # 把离线脚本预先渲染的 K 条 summarize_prompts 转成 [K, max_summarize_length] 张量。
        # 与父类填好的 row_dict['input_ids']（原始短 prompt，max_prompt_length）并存：
        #   - input_ids: rollout 不会用，loss-time actor forward 用
        #   - summarize_input_ids[step_i]: rollout-time vllm 用
        # 注意：当 use_summarize=True 时，无论本行是否有 summarize_prompts 列，
        # 都会写入这三个张量（缺数据则用 pad 占位），保证 collate_fn 不会因 key
        # 不一致而崩。trainer 端的 assert 会对全 pad 行做诊断。
        if self.use_summarize:
            K = self.max_summarize_prompts
            sp_list = original_row.get(self.summarize_prompts_key)
            if isinstance(sp_list, np.ndarray):
                sp_list = sp_list.tolist()

            if not sp_list:
                # 数据缺失保护：返回全 pad
                summarize_input_ids = torch.full(
                    (K, self.max_summarize_length), self.tokenizer.pad_token_id, dtype=torch.long,
                )
            else:
                # 长度对齐到 K：超长截断、不足按 idx 映射插值复制
                if len(sp_list) >= K:
                    sp_list = sp_list[:K]
                elif len(sp_list) == 1:
                    sp_list = sp_list * K
                else:
                    sp_list = [sp_list[int(i / (K - 1) * (len(sp_list) - 1))] for i in range(K)]

                sum_ids_list = [self._render_summarize_ids(m) for m in sp_list]
                summarize_input_ids = torch.stack(sum_ids_list, dim=0)  # [K, L_long]

            summarize_attention_mask = (
                summarize_input_ids != self.tokenizer.pad_token_id
            ).to(torch.long)
            summarize_position_ids = torch.stack(
                [compute_position_id_with_mask(summarize_attention_mask[i])
                 for i in range(summarize_input_ids.size(0))],
                dim=0,
            )

            row_dict['summarize_input_ids'] = summarize_input_ids
            row_dict['summarize_attention_mask'] = summarize_attention_mask
            row_dict['summarize_position_ids'] = summarize_position_ids

            # ---- 单条 summarize_prompt（给 extra-step / recycle 用）----
            # 物理隔离于上面的 K 条列表：extra-step 所有 off rollout 共用这一条。
            # 该列存长度 1 数组（prepare_summarize_prompts.py 的 summarize_prompt 列）。
            # 缺列 -> 回退用列表里最短那条（[0]，prepare 已保证升序）；都没有 -> 全 pad。
            single_raw = original_row.get(self.summarize_prompt_key)
            if isinstance(single_raw, np.ndarray):
                single_raw = single_raw.tolist()
            single_msgs = None
            if single_raw:
                single_msgs = single_raw[0]
            elif sp_list:
                single_msgs = sp_list[0]
            if single_msgs is not None:
                summarize_input_id = self._render_summarize_ids(single_msgs)  # [L_long]
            else:
                summarize_input_id = torch.full(
                    (self.max_summarize_length,), self.tokenizer.pad_token_id, dtype=torch.long,
                )
            summarize_attention_mask_single = (
                summarize_input_id != self.tokenizer.pad_token_id
            ).to(torch.long)
            summarize_position_id_single = compute_position_id_with_mask(
                summarize_attention_mask_single
            )
            row_dict['summarize_input_id'] = summarize_input_id
            row_dict['summarize_attention_mask_single'] = summarize_attention_mask_single
            row_dict['summarize_position_id_single'] = summarize_position_id_single

        # ---- 离线 SR 候选（sr_response 列）----
        # Data/aggregate_sr_responses.py 预先跑好、筛好的一条「正确且长度中位数」回复。
        # 训练时 _summarize_replace_normal_step 直接用它，不再每步 generate K 条候选
        # （128 题 × 8 条 × 上万 token，最终每题只留 1 条，~97% 被丢掉）。
        #
        # 右填充，与 responses 的排布一致：off 行最终形状是 [短 prompt, response]，
        # response 段必须右填充才能和 on-policy 批的 response 宽度对齐。
        # 这里只出 token；是否真的启用由 trainer 端判断（全 pad 行 = 该题无候选）。
        if self.max_sr_response_length > 0:
            sr_text = original_row.get(self.sr_response_key)
            if isinstance(sr_text, str) and sr_text.strip():
                sr_ids = self.tokenizer(
                    sr_text, add_special_tokens=False, return_tensors='pt',
                )['input_ids'].reshape(1, -1)
                if sr_ids.shape[-1] < self.max_sr_response_length:
                    sr_ids = pad_sequence_to_length(
                        sr_ids,
                        max_seq_len=self.max_sr_response_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=False,
                    )
                else:
                    # 右截断：SR 候选是要被当成模型自己的 response 来算 loss 的，
                    # 左截断会砍掉开头、留下无头的推理，比截掉结尾更糟。
                    sr_ids = sr_ids[:, :self.max_sr_response_length]
                row_dict['sr_response_ids'] = sr_ids.squeeze(0)
            else:
                # 该题没有可用候选（aggregate 脚本 --keep-unsolved，或本来就没这列）。
                # 必须仍然写入这个 key：collate_fn 要求整批 key 一致，缺一行就崩。
                # 全 pad 是 trainer 端判「跳过这题」的判据。
                row_dict['sr_response_ids'] = torch.full(
                    (self.max_sr_response_length,), self.tokenizer.pad_token_id, dtype=torch.long,
                )

        # 父类已经处理了 'raw_prompt', 'index' 等字段，我们无需重复
        # 直接返回被我们追加了 target 相关字段的 `row_dict`
        #print(row_dict["input_ids"].shape, row_dict["attention_mask"].shape, row_dict["position_ids"].shape, row_dict['tgt_input_ids'].shape)
        row_dict.pop("full_prompts", None)
        #print(row_dict)
        return row_dict

    def _render_summarize_ids(self, messages) -> torch.Tensor:
        """Render one summarize message-list -> [max_summarize_length] ids.

        Shared by the K-list (summarize_input_ids) and the single column
        (summarize_input_id). Short prompts are left-padded (parent class's
        prompt-padding convention). Over-long prompts are RIGHT-truncated:
        we keep the head (system + instructions + [Problem]) and drop the tail
        of the draft. Left-truncation was wrong here -- it deleted the problem
        statement and produced headless prompt fragments. Right-truncation
        loses the trailing generation prompt (so the model continues the draft
        rather than re-deriving), but keeps the question visible, which is the
        lesser evil. The real fix is to bound the draft length offline in
        prepare_summarize_prompts.py.
        """
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        full = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(
            full, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        if ids.shape[-1] < self.max_summarize_length:
            ids = pad_sequence_to_length(
                ids,
                max_seq_len=self.max_summarize_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
            )
        else:
            ids = ids[:, :self.max_summarize_length]
        return ids.squeeze(0)

    def _process_target(self, tgt: str, prompt: str, add_eos=False) -> torch.Tensor:
        if prompt.endswith('<think>\n') and tgt.startswith('<think>\n'):
            tgt = tgt[len('<think>\n'):]
        tgt_input_ids = self.tokenizer(tgt, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]
        if add_eos:
            tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([self.tokenizer.eos_token_id], device=tgt_input_ids.device, dtype=tgt_input_ids.dtype).reshape(-1)])

        tgt_input_ids = tgt_input_ids.reshape(1, -1)
        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)

        return tgt_input_ids

from verl import DataProto
class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)

import torch

class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1].item() # the output index should be int

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])

def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    token_ids = prompt_token_ids[:non_pad_index[-1][0]].tolist()
    return token_ids