# Extracted from miles/rollout/data_source.py (RolloutDataSource)
# and miles/utils/data.py (Dataset)
"""
GSM8K data loading and prompt formatting utilities for Tiny Miles.

Usage:
    from gsm8k_utils import load_gsm8k, GSM8KDataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data = load_gsm8k(split="train")
    loader = GSM8KDataLoader(train_data, tokenizer, batch_size=8, n_samples_per_prompt=8)
    batch = loader.get_batch()
"""

import copy
import random

from datasets import load_dataset


def load_gsm8k(split: str = "train") -> list[dict]:
    """Load GSM8K dataset from HuggingFace.

    Returns a list of dicts with keys:
        - "question": the math question
        - "answer": full solution text (with #### delimiter)
        - "label": the numeric answer extracted from the solution

    The label extraction mirrors the Miles config: --label-key label.
    GSM8K answers have the format: "...#### <number>".
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    samples = []
    for item in dataset:
        # Extract the numeric answer after "####"
        answer_text = item["answer"]
        if "####" in answer_text:
            label = answer_text.split("####")[-1].strip()
        else:
            label = answer_text.strip()

        samples.append(
            {
                "question": item["question"],
                "answer": answer_text,
                "label": label,
            }
        )
    return samples


def format_prompt(question: str, tokenizer) -> dict:
    """Format a GSM8K question as a chat prompt and tokenize it.

    Mirrors the Miles config: --input-key messages --apply-chat-template.
    Uses the model's chat template to format the prompt.

    Returns a dict with:
        - "prompt_text": the formatted prompt string
        - "token_ids": list of token IDs
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please put the answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return {
        "prompt_text": prompt_text,
        "token_ids": token_ids,
    }


class GSM8KDataLoader:
    """Data loader for GSM8K that produces batches of prompt groups.

    Mirrors miles/rollout/data_source.py: RolloutDataSource.get_samples().
    Each call to get_batch() returns `batch_size` groups, where each group
    contains `n_samples_per_prompt` copies of the same prompt (for GRPO).

    Args:
        dataset: list of dicts from load_gsm8k()
        tokenizer: HuggingFace tokenizer
        batch_size: number of unique prompts per batch
        n_samples_per_prompt: number of response samples per prompt (for GRPO)
        shuffle: whether to shuffle the dataset each epoch
        seed: random seed for shuffling
    """

    def __init__(
        self,
        dataset: list[dict],
        tokenizer,
        batch_size: int,
        n_samples_per_prompt: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_samples_per_prompt = n_samples_per_prompt
        self.shuffle = shuffle
        self.seed = seed

        self.epoch_id = 0
        self.sample_offset = 0

        # Pre-format all prompts
        self.samples = []
        for item in dataset:
            formatted = format_prompt(item["question"], tokenizer)
            self.samples.append(
                {
                    "question": item["question"],
                    "label": item["label"],
                    "prompt_text": formatted["prompt_text"],
                    "token_ids": formatted["token_ids"],
                }
            )

        # Store original order for epoch reshuffling
        self.original_samples = list(self.samples)

        if shuffle:
            self._shuffle(self.epoch_id)

    def _shuffle(self, epoch_id: int):
        """Shuffle samples deterministically based on epoch.

        Mirrors miles/rollout/data_source.py: dataset.shuffle(epoch_id).
        """
        rng = random.Random(self.seed + epoch_id)
        permutation = list(range(len(self.original_samples)))
        rng.shuffle(permutation)
        self.samples = [self.original_samples[i] for i in permutation]

    def get_batch(self) -> list[list[dict]]:
        """Get a batch of prompt groups for rollout.

        Returns:
            list of `batch_size` groups, each group is a list of
            `n_samples_per_prompt` dicts (copies of the same prompt).

        Mirrors miles/rollout/data_source.py: RolloutDataSource.get_samples().
        """
        num_samples = self.batch_size

        # Handle epoch boundary
        if self.sample_offset + num_samples <= len(self.samples):
            prompt_samples = self.samples[
                self.sample_offset : self.sample_offset + num_samples
            ]
            self.sample_offset += num_samples
        else:
            prompt_samples = self.samples[self.sample_offset :]
            remaining = num_samples - len(prompt_samples)
            self.epoch_id += 1
            if self.shuffle:
                self._shuffle(self.epoch_id)
            prompt_samples += self.samples[:remaining]
            self.sample_offset = remaining

        # Create groups: each prompt gets n_samples_per_prompt copies
        batch = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.n_samples_per_prompt):
                group.append(copy.deepcopy(prompt_sample))
            batch.append(group)

        return batch

    def __len__(self):
        return len(self.samples)
