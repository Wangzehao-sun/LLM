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
"""PICO training dataset. Copied from verl/custom/rl_dataset_with_target.py, keeping only
the reference solution (``target``) and the summarize prompts.

Each row adds to what RLHFDataset produces:
  * ``tgt_input_ids``       [max_target_length]           right-padded reference solution
                                                          (only written into the rollout dump).
  * ``summarize_input_ids`` [K, max_summarize_length]     left-padded summarize prompts, the
                                                          teacher's prompt in joint decoding.
"""

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.torch_functional import pad_sequence_to_length


class PICORLHFDatasetWithTarget(RLHFDataset):
    def __init__(
        self,
        parquet_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        target_key="target",
        max_target_length=8192,
        summarize_prompts_key="summarize_prompts",
        max_summarize_prompts=8,
        max_summarize_length=8192,
    ):
        super().__init__(parquet_files, tokenizer, config=config)
        self.target_key = target_key
        self.max_target_length = max_target_length
        self.summarize_prompts_key = summarize_prompts_key
        self.max_summarize_prompts = max_summarize_prompts
        self.max_summarize_length = max_summarize_length

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        # The parent may have popped fields; read the target / summarize columns from the raw row.
        original_row: dict = self.dataframe[item]

        # Reference solution, right-padded (empty when the row has none).
        tgt = original_row.get(self.target_key)
        if tgt is not None:
            content = tgt[0]["content"]
            if row_dict.get("full_prompts", "").endswith("<think>\n") and content.startswith("<think>\n"):
                content = content[len("<think>\n") :]
            tgt_input_ids = self.tokenizer(content, add_special_tokens=False, return_tensors="pt")["input_ids"].reshape(1, -1)
        else:
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0)
        if tgt_input_ids.shape[-1] < self.max_target_length:
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids, max_seq_len=self.max_target_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=False)
        else:
            assert self.truncation in ("right", "error")
            tgt_input_ids = tgt_input_ids[:, : self.max_target_length]
        row_dict["tgt_input_ids"] = tgt_input_ids.squeeze(0)

        # Summarize prompts, aligned to K rows: cut when longer, repeated when there is one,
        # otherwise spread by index. A missing column gives all-pad rows so collate_fn still
        # sees the same keys in every row.
        k = self.max_summarize_prompts
        sp_list = original_row.get(self.summarize_prompts_key)
        if isinstance(sp_list, np.ndarray):
            sp_list = sp_list.tolist()
        if not sp_list:
            summarize_input_ids = torch.full((k, self.max_summarize_length), self.tokenizer.pad_token_id, dtype=torch.long)
        else:
            if len(sp_list) >= k:
                sp_list = sp_list[:k]
            elif len(sp_list) == 1:
                sp_list = sp_list * k
            else:
                sp_list = [sp_list[int(i / (k - 1) * (len(sp_list) - 1))] for i in range(k)]
            summarize_input_ids = torch.stack([self._render_summarize_ids(m) for m in sp_list], dim=0)
        row_dict["summarize_input_ids"] = summarize_input_ids

        row_dict.pop("full_prompts", None)
        return row_dict

    def _render_summarize_ids(self, messages) -> torch.Tensor:
        """Render one summarize message-list -> [max_summarize_length] ids.

        Short prompts are left-padded (the parent class's prompt-padding convention).
        Over-long prompts are RIGHT-truncated: the head (system + instructions + problem) is
        kept and the tail of the draft dropped. That also loses the trailing generation
        prompt, but keeps the question visible; the real fix is to bound the draft length
        offline in Data/prepare_summarize_prompts.py.
        """
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        full = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.shape[-1] < self.max_summarize_length:
            ids = pad_sequence_to_length(ids, max_seq_len=self.max_summarize_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True)
        else:
            ids = ids[:, : self.max_summarize_length]
        return ids.squeeze(0)
