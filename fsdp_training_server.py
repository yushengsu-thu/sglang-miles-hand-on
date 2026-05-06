f"""Tiny Miles FSDP Training Server.

A minimal training engine extracted from the Miles RL framework.
Launched via torchrun:

    torchrun --nproc-per-node=2 fsdp_training_server.py \\
        --model-path <path> --sglang-url http://localhost:30000 --port 5000

Architecture:
    - Rank 0: Flask HTTP server + FSDP training participant
    - Rank 1+: FSDP training participant (worker loop)

HTTP Endpoints (Rank 0 only):
    GET  /health          - Health check
    POST /train_step      - Receive rollout data, perform one GRPO training step
    POST /update_weights  - Sync weights to SGLang inference engine

Extracted from:
    - miles/backends/fsdp_utils/actor.py  (FSDP init, training step)
    - miles/backends/fsdp_utils/update_weight_utils.py  (weight update)
    - miles/utils/ppo_utils.py  (GRPO, policy loss, log probs)
    - miles/backends/training_utils/loss.py  (loss computation)
"""

import argparse
import logging
import os
import threading
import time

import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from flask import Flask, jsonify, request as flask_request
from torch.distributed.tensor import DTensor, Replicate
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)
_log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(_log_dir, "train_server.log")),
    ],
)

# ---------------------------------------------------------------------------
# Signal constants for rank coordination
# ---------------------------------------------------------------------------
SIGNAL_TRAIN = 1
SIGNAL_UPDATE_WEIGHTS = 2
SIGNAL_SLEEP = 3
SIGNAL_WAKE_UP = 4
SIGNAL_SHUTDOWN = -1


# ===================================================================
# FSDP Model Initialization
# Extracted from: miles/backends/fsdp_utils/actor.py
# ===================================================================


def apply_fsdp2(model, mesh=None, args=None):
    """Apply FSDP v2 to the model.

    Extracted from miles/backends/fsdp_utils/actor.py: apply_fsdp2().

    Wraps each transformer layer and the top-level model with
    fully_shard() using bf16 mixed precision.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    # Newer transformers versions return a `set` here; convert to `list` so
    # subscripting (and downstream `in` checks) behave consistently.
    layer_cls_to_wrap = list(model._no_split_modules)
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    fsdp_kwargs = {"mp_policy": mp_policy, "mesh": mesh}

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model


def fsdp2_load_full_state_dict(model, full_state, device_mesh):
    """Load full state dict into FSDP2 model with broadcast from rank 0.

    Extracted from miles/backends/fsdp_utils/actor.py:
    FSDPTrainRayActor._fsdp2_load_full_state_dict().
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        model = model.to_empty(device=torch.cuda.current_device())

    options = StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # set_model_state_dict does not broadcast buffers
    for _name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    return model


# ===================================================================
# RL Computation Utilities
# Extracted from: miles/utils/ppo_utils.py
# ===================================================================


def compute_token_log_probs(logits, tokens):
    """Compute per-token log probabilities.

    Extracted from miles/utils/ppo_utils.py:
    _calculate_log_probs_and_entropy_true_on_policy().

    Args:
        logits: [B, T, V] model output logits
        tokens: [B, T] token IDs

    Returns:
        token_log_probs: [B, T-1] log prob of tokens[t+1] given logits[t]
    """
    # Shift: logits[t] predicts tokens[t+1]
    shift_logits = logits[:, :-1, :].float()
    shift_tokens = tokens[:, 1:]

    log_probs_full = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs_full, dim=-1, index=shift_tokens.unsqueeze(-1)
    ).squeeze(-1)
    return token_log_probs


def compute_grpo_advantages(rewards, n_samples_per_prompt):
    """Compute GRPO advantages: group-normalized rewards.

    Extracted from miles/utils/ppo_utils.py: get_grpo_returns().

    In GRPO, rewards are normalized within each prompt group
    (mean-subtracted, std-divided), then broadcast to every token.

    Args:
        rewards: list of float, length = batch_size * n_samples_per_prompt
        n_samples_per_prompt: int

    Returns:
        normalized_rewards: tensor of shape [total_samples]
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    batch_size = len(rewards) // n_samples_per_prompt

    # Group by prompt and normalize within each group
    grouped = rewards_t.view(batch_size, n_samples_per_prompt)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    normalized = (grouped - mean) / (std + 1e-8)

    return normalized.view(-1)


def compute_policy_loss(log_ratio, advantages, eps_clip, eps_clip_high=None):
    """Compute clipped policy gradient loss.

    Extracted from miles/utils/ppo_utils.py: compute_policy_loss().

    Args:
        log_ratio: log(pi_new / pi_old) = new_log_probs - old_log_probs
        advantages: per-token advantages
        eps_clip: lower clip bound (e.g. 0.2 means ratio clipped to [0.8, 1+eps_clip_high])
        eps_clip_high: upper clip bound (defaults to eps_clip)

    Returns:
        (pg_loss, clipfrac) — scalar loss and fraction of clipped tokens
    """
    if eps_clip_high is None:
        eps_clip_high = eps_clip

    ratio = log_ratio.exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    pg_loss = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float().mean()
    return pg_loss, clipfrac


# ===================================================================
# Weight Update
# Extracted from: miles/backends/fsdp_utils/update_weight_utils.py
# ===================================================================


def update_weights_to_sglang(model, sglang_url, tp_size, weight_version):
    """Sync FSDP model weights to SGLang inference engine.

    Extracted from miles/backends/fsdp_utils/update_weight_utils.py:
    UpdateWeight.update_weights() and UpdateWeightFromTensor.update_bucket_weights().

    Must be called by ALL ranks simultaneously. The DTensor.redistribute() call
    is an all-gather collective — every rank must participate or the call blocks.

    Each rank serializes on its own GPU, then we Gloo-gather to rank 0.
    Rank 0 sends per-TP serialized handles to SGLang so each TP rank receives
    IPC handles pointing to the correct GPU — mirrors Miles exactly.
    """
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    BUCKET_SIZE = 256 * 1024 * 1024  # 256MB, from Miles default

    bucket = []
    bucket_size = 0

    def _send_bucket(named_tensors, flush_cache=False):
        """Each rank serializes its tensors, gather to rank 0, send to SGLang.

        Mirrors miles/backends/fsdp_utils/update_weight_utils.py:
        UpdateWeightFromTensor.update_bucket_weights().
        """
        # Group by dtype (mirrors Miles)
        tensors_by_dtype = {}
        for name, tensor in named_tensors:
            dt = tensor.dtype
            if dt not in tensors_by_dtype:
                tensors_by_dtype[dt] = []
            tensors_by_dtype[dt].append((name, tensor))

        # Each rank serializes its own tensors (on its own GPU)
        serialized_per_dtype = []
        for _dt, dt_tensors in tensors_by_dtype.items():
            flattened_bucket = FlattenedTensorBucket(named_tensors=dt_tensors)
            flattened_data = {
                "flattened_tensor": flattened_bucket.get_flattened_tensor(),
                "metadata": flattened_bucket.get_metadata(),
            }
            serialized_per_dtype.append(
                MultiprocessingSerializer.serialize(flattened_data, output_str=True)
            )

        # Gloo-gather: rank 0 collects serialized data from ALL ranks
        # gathered[r] = rank r's serialized_per_dtype list
        if rank == 0:
            gathered = [None for _ in range(world_size)]
        else:
            gathered = None
        dist.gather_object(serialized_per_dtype, gathered, dst=0)

        # Rank 0 sends to SGLang: each TP rank gets IPC handles from its own GPU
        if rank == 0:
            num_dtypes = len(gathered[0])
            for i in range(num_dtypes):
                payload = {
                    "serialized_named_tensors": [gathered[r][i] for r in range(world_size)],
                    "load_format": "flattened_bucket",
                    "flush_cache": flush_cache,
                    "weight_version": str(weight_version),
                }
                resp = requests.post(
                    f"{sglang_url}/update_weights_from_tensor", json=payload, timeout=300
                )
                resp.raise_for_status()

    # Stream parameters in size-bounded buckets (from Miles UpdateWeight.update_weights).
    # ALL ranks iterate and redistribute (all-gather); only rank 0 tracks bucket size.
    for name, param in model.state_dict().items():
        param_size = param.numel() * param.element_size()

        if bucket and bucket_size + param_size >= BUCKET_SIZE:
            _send_bucket(bucket)
            bucket = []
            bucket_size = 0

        param = param.cuda()
        if isinstance(param, DTensor):
            # All-gather: every rank must call redistribute() for the same parameter
            # at the same time, or the collective will deadlock.
            param = param.redistribute(
                placements=[Replicate()] * param.device_mesh.ndim,
                async_op=False,
            ).to_local()

        bucket.append((name, param))
        bucket_size += param_size

    if bucket:
        _send_bucket(bucket, flush_cache=False)
    # Flush cache after all buckets are sent
    if rank == 0:
        requests.get(f"{sglang_url}/flush_cache", timeout=60)


# ===================================================================
# Training Engine
# ===================================================================


class TrainingEngine:
    """FSDP training engine for GRPO.

    Mirrors the training logic from:
        - miles/backends/fsdp_utils/actor.py: FSDPTrainRayActor
        - miles/backends/training_utils/loss.py: policy_loss_function
    """

    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.weight_version = 0
        self.global_step = 0

        torch.cuda.set_device(self.rank)
        torch.manual_seed(args.seed)

        # --- Model Init (from FSDPTrainRayActor.init) ---
        logger.info(f"[Rank {self.rank}] Loading model from {args.model_path}")

        hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

        # Sequential init to avoid cache race (from Miles actor.py line 84-91)
        for i in range(self.world_size):
            if i == self.rank:
                if self.rank == 0 or hf_config.tie_word_embeddings:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                    )
                else:
                    from accelerate import init_empty_weights

                    with init_empty_weights():
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model_path,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16,
                        )
            dist.barrier()

        model.train()
        full_state = model.state_dict()

        # FSDP wrapping
        from torch.distributed.device_mesh import init_device_mesh

        device_mesh = init_device_mesh("cuda", (self.world_size,))
        model = apply_fsdp2(model, mesh=device_mesh, args=args)
        model = fsdp2_load_full_state_dict(model, full_state, device_mesh)

        self.model = model
        self.device_mesh = device_mesh

        # Gradient checkpointing
        if getattr(args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        # Optimizer (aligned with Miles test config: adam, beta2=0.98, weight_decay=0.1)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )

        logger.info(f"[Rank {self.rank}] Training engine initialized")

    def train_step(self, rollout_data):
        """Perform one GRPO training step with gradient accumulation over micro-batches.

        Mirrors:
            - miles/backends/fsdp_utils/actor.py: _train_core() + _train_step()
            - miles/backends/training_utils/loss.py: policy_loss_function()
            - miles/backends/training_utils/data.py: DataIterator (micro-batch splitting)
            - miles/utils/ppo_utils.py: compute_policy_loss(), get_grpo_returns()

        The full rollout batch is split into micro-batches for gradient accumulation.
        Advantages are computed on the full batch first (GRPO requires the full group),
        then each micro-batch does a forward+backward pass. Gradients accumulate across
        micro-batches and a single optimizer step is taken at the end.

        Loss scaling: each micro-batch loss = sum(pg_loss * mask) / total_masked_tokens,
        so accumulated gradients exactly equal the full-batch gradient.

        Args:
            rollout_data: dict with keys:
                tokens: list of list[int] — full sequences (prompt + response)
                rollout_log_probs: list of list[float] — per-response-token log probs from SGLang
                rewards: list of float — per-sample rewards (0.0 or 1.0)
                prompt_lengths: list of int
                response_lengths: list of int
                n_samples_per_prompt: int

        Returns:
            dict with training metrics
        """
        device = torch.cuda.current_device()
        tokens_list = rollout_data["tokens"]
        rollout_lp_list = rollout_data["rollout_log_probs"]
        rewards = rollout_data["rewards"]
        prompt_lengths = rollout_data["prompt_lengths"]
        response_lengths = rollout_data["response_lengths"]
        n_samples = rollout_data["n_samples_per_prompt"]
        total_samples = len(tokens_list)
        micro_batch_size = self.args.micro_batch_size

        # --- Compute GRPO advantages on the FULL batch (from ppo_utils.get_grpo_returns) ---
        # Must use the full batch so each prompt group's rewards are normalized together.
        normalized_rewards = compute_grpo_advantages(rewards, n_samples)  # [total_samples]

        # Total masked (response) tokens across the entire batch — used to scale each
        # micro-batch loss so that accumulated gradients equal the full-batch gradient.
        # Mirrors miles/backends/training_utils/loss.py: loss / global_batch_size * dp_size
        total_masked_tokens = max(sum(response_lengths), 1)

        # --- Gradient accumulation over micro-batches (from actor.py _train_core) ---
        # zero_grad once before the micro-batch loop; gradients accumulate across calls
        # to .backward() and a single optimizer step is taken after all micro-batches.
        self.optimizer.zero_grad(set_to_none=True)

        # Accumulators for logging metrics
        accum_loss = 0.0
        accum_clipfrac_num = 0.0   # weighted numerator (clipfrac * tokens)
        accum_log_ratio_num = 0.0  # weighted numerator (log_ratio * tokens)
        accum_tokens = 0

        n_micro_batches = (total_samples + micro_batch_size - 1) // micro_batch_size
        logger.info(
            f"[Rank {self.rank}] train_step: {total_samples} samples → "
            f"{n_micro_batches} micro-batches of size ≤{micro_batch_size}"
        )

        for mb_start in range(0, total_samples, micro_batch_size):
            mb_end = min(mb_start + micro_batch_size, total_samples)
            mb_size = mb_end - mb_start

            # Slice micro-batch data (mirrors DataIterator.get_next() in Miles)
            mb_tokens = tokens_list[mb_start:mb_end]
            mb_rollout_lp = rollout_lp_list[mb_start:mb_end]
            mb_prompt_lengths = prompt_lengths[mb_start:mb_end]
            mb_response_lengths = response_lengths[mb_start:mb_end]
            mb_advantages = normalized_rewards[mb_start:mb_end]  # already on CPU

            # Pad sequences within the micro-batch
            mb_max_len = max(len(t) for t in mb_tokens)

            padded_tokens = torch.zeros(mb_size, mb_max_len, dtype=torch.long, device=device)
            loss_mask = torch.zeros(mb_size, mb_max_len, dtype=torch.float32, device=device)
            old_log_probs_padded = torch.zeros(mb_size, mb_max_len - 1, dtype=torch.float32, device=device)
            advantages_padded = torch.zeros(mb_size, mb_max_len - 1, dtype=torch.float32, device=device)

            for i in range(mb_size):
                seq_len = len(mb_tokens[i])
                padded_tokens[i, :seq_len] = torch.tensor(mb_tokens[i], dtype=torch.long)

                # Loss mask: 1 for response tokens, 0 for prompt and padding
                p_len = mb_prompt_lengths[i]
                r_len = mb_response_lengths[i]
                loss_mask[i, p_len : p_len + r_len] = 1.0

                # Old log probs from SGLang aligned to shifted positions
                # [prompt_len-1, prompt_len-1+response_len)
                if len(mb_rollout_lp[i]) > 0:
                    rlp = torch.tensor(mb_rollout_lp[i], dtype=torch.float32)
                    old_log_probs_padded[i, p_len - 1 : p_len - 1 + r_len] = rlp[:r_len]

                # Broadcast normalized reward to every response token position
                advantages_padded[i, p_len - 1 : p_len - 1 + r_len] = mb_advantages[i]

            # Shifted loss mask (logits[t] predicts tokens[t+1])
            shifted_loss_mask = loss_mask[:, 1:]  # [mb_size, T-1]
            mb_masked_tokens = int(shifted_loss_mask.sum().item())

            # --- Forward pass (from actor.py _train_step) ---
            position_ids = torch.arange(mb_max_len, device=device).unsqueeze(0).expand(mb_size, -1)
            logits = self.model(
                input_ids=padded_tokens, position_ids=position_ids, attention_mask=None
            ).logits

            # --- Compute new log probs ---
            new_log_probs = compute_token_log_probs(logits, padded_tokens)  # [mb_size, T-1]

            # --- Compute policy loss ---
            log_ratio = new_log_probs - old_log_probs_padded  # [mb_size, T-1]
            pg_loss, clipfrac = compute_policy_loss(log_ratio, advantages_padded, self.args.eps_clip, self.args.eps_clip_high)

            # Scale loss so that micro-batch gradients accumulate to the full-batch gradient.
            # sum(pg_loss * mask) / total_masked_tokens  →  summed across micro-batches
            # equals  sum_all_tokens(pg_loss * mask) / total_masked_tokens  (full-batch loss).
            micro_loss = (pg_loss * shifted_loss_mask).sum() / total_masked_tokens

            # --- Backward: accumulate gradients (no optimizer step yet) ---
            micro_loss.backward()

            # Accumulate metrics (token-weighted so final values match full-batch computation)
            accum_loss += micro_loss.item()
            accum_clipfrac_num += clipfrac.item() * mb_masked_tokens
            with torch.no_grad():
                accum_log_ratio_num += (log_ratio * shifted_loss_mask).sum().item()
            accum_tokens += mb_masked_tokens

        # --- Single optimizer step after all micro-batches (from actor.py _train_core) ---
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor().item()
        else:
            grad_norm = grad_norm.item()

        self.optimizer.step()
        self.global_step += 1

        safe_tokens = max(accum_tokens, 1)
        return {
            "loss": accum_loss,
            "grad_norm": grad_norm,
            "clipfrac": accum_clipfrac_num / safe_tokens,
            "mean_log_ratio": accum_log_ratio_num / safe_tokens,
            "mean_reward": sum(rewards) / len(rewards),
            "global_step": self.global_step,
            "n_micro_batches": n_micro_batches,
        }

    def do_update_weights(self):
        """Sync weights to SGLang. Called on rank 0 only."""
        self.weight_version += 1
        logger.info(f"[Rank {self.rank}] Updating weights to SGLang (version {self.weight_version})")
        update_weights_to_sglang(
            self.model,
            self.args.sglang_url,
            tp_size=self.args.tp_size,
            weight_version=self.weight_version,
        )
        logger.info(f"[Rank {self.rank}] Weight update complete")
        return {"success": True, "weight_version": self.weight_version}

    def sleep(self):
        """Offload model and optimizer to CPU to free GPU memory for SGLang.

        Mirrors miles/backends/fsdp_utils/actor.py: sleep().
        Called after update_weights, before SGLang rollout.
        """
        logger.info(f"[Rank {self.rank}] Sleeping: offloading model and optimizer to CPU")
        self.model.cpu()
        # Move all optimizer state tensors to CPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.cuda.empty_cache()
        dist.barrier()
        logger.info(f"[Rank {self.rank}] Sleep complete")
        return {"success": True}

    def wake_up(self):
        """Reload model and optimizer from CPU back to GPU before training.

        Mirrors miles/backends/fsdp_utils/actor.py: wake_up().
        Called before train_step.
        """
        logger.info(f"[Rank {self.rank}] Waking up: loading model and optimizer to GPU")
        self.model.cuda()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        dist.barrier()
        logger.info(f"[Rank {self.rank}] Wake up complete")
        return {"success": True}


# ===================================================================
# Flask HTTP Server (Rank 0 only)
# ===================================================================

# Shared state between Flask and training engine
_engine: TrainingEngine = None
_train_data_lock = threading.Lock()
_train_data = None
_train_result = None
_train_event = threading.Event()
_result_event = threading.Event()
_update_weights_event = threading.Event()
_update_weights_result_event = threading.Event()
_update_weights_result = None
_sleep_event = threading.Event()
_sleep_result_event = threading.Event()
_sleep_result = None
_wake_up_event = threading.Event()
_wake_up_result_event = threading.Event()
_wake_up_result = None

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "rank": 0, "weight_version": _engine.weight_version})


@app.route("/train_step", methods=["POST"])
def train_step():
    global _train_data, _train_result
    data = flask_request.get_json()

    with _train_data_lock:
        _train_data = data
        _result_event.clear()
        _train_event.set()  # Signal all ranks to start training

    # Wait for training to complete
    _result_event.wait()
    return jsonify(_train_result)


@app.route("/update_weights", methods=["POST"])
def update_weights():
    global _update_weights_result
    _update_weights_result_event.clear()
    _update_weights_event.set()  # Signal all ranks

    _update_weights_result_event.wait()
    return jsonify(_update_weights_result)


@app.route("/sleep", methods=["POST"])
def sleep():
    global _sleep_result
    _sleep_result_event.clear()
    _sleep_event.set()

    _sleep_result_event.wait()
    return jsonify(_sleep_result)


@app.route("/wake_up", methods=["POST"])
def wake_up():
    global _wake_up_result
    _wake_up_result_event.clear()
    _wake_up_event.set()

    _wake_up_result_event.wait()
    return jsonify(_wake_up_result)


# ===================================================================
# Main: Rank coordination
# ===================================================================


def run_rank0_server(engine, port):
    """Run Flask server on rank 0 in a separate thread."""
    global _engine
    _engine = engine

    server_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True),
        daemon=True,
    )
    server_thread.start()
    logger.info(f"[Rank 0] Flask server started on port {port}")
    return server_thread


def rank0_loop(engine):
    """Rank 0: process signals from Flask endpoints."""
    global _train_result, _update_weights_result, _sleep_result, _wake_up_result

    while True:
        # Check for train signal
        if _train_event.is_set():
            _train_event.clear()

            signal = torch.tensor([SIGNAL_TRAIN], dtype=torch.long, device="cuda")
            dist.broadcast(signal, src=0)
            dist.broadcast_object_list([_train_data], src=0)

            _train_result = engine.train_step(_train_data)
            _result_event.set()
            continue

        # Check for weight update signal
        if _update_weights_event.is_set():
            _update_weights_event.clear()

            signal = torch.tensor([SIGNAL_UPDATE_WEIGHTS], dtype=torch.long, device="cuda")
            dist.broadcast(signal, src=0)

            _update_weights_result = engine.do_update_weights()
            _update_weights_result_event.set()
            continue

        # Check for sleep signal
        if _sleep_event.is_set():
            _sleep_event.clear()

            signal = torch.tensor([SIGNAL_SLEEP], dtype=torch.long, device="cuda")
            dist.broadcast(signal, src=0)

            _sleep_result = engine.sleep()
            _sleep_result_event.set()
            continue

        # Check for wake_up signal
        if _wake_up_event.is_set():
            _wake_up_event.clear()

            signal = torch.tensor([SIGNAL_WAKE_UP], dtype=torch.long, device="cuda")
            dist.broadcast(signal, src=0)

            _wake_up_result = engine.wake_up()
            _wake_up_result_event.set()
            continue

        time.sleep(0.01)


def worker_loop(engine):
    """Non-rank-0 workers: wait for signals from rank 0."""
    while True:
        signal = torch.tensor([0], dtype=torch.long, device="cuda")
        dist.broadcast(signal, src=0)

        sig = signal.item()
        if sig == SIGNAL_TRAIN:
            data_list = [None]
            dist.broadcast_object_list(data_list, src=0)
            engine.train_step(data_list[0])

        elif sig == SIGNAL_UPDATE_WEIGHTS:
            # All ranks must participate in the DTensor all-gather collectives.
            # update_weights_to_sglang() is rank-aware: only rank 0 sends HTTP.
            update_weights_to_sglang(
                engine.model,
                engine.args.sglang_url,
                tp_size=engine.args.tp_size,
                weight_version=engine.weight_version,
            )

        elif sig == SIGNAL_SLEEP:
            engine.sleep()

        elif sig == SIGNAL_WAKE_UP:
            engine.wake_up()

        elif sig == SIGNAL_SHUTDOWN:
            logger.info(f"[Rank {dist.get_rank()}] Shutdown signal received")
            break


def main():
    parser = argparse.ArgumentParser(description="Tiny Miles FSDP Training Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000", help="SGLang server URL")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port (rank 0)")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip epsilon (lower bound)")
    parser.add_argument("--eps-clip-high", type=float, default=0.28, help="PPO clip epsilon upper bound (asymmetric clipping)")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="L2 weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tp-size", type=int, default=2, help="SGLang tensor parallelism size")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument(
        "--micro-batch-size", type=int, default=1,
        help="Samples per micro-batch for gradient accumulation (reduces peak GPU memory)"
    )
    args = parser.parse_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    logger.info(f"[Rank {rank}] Initializing training engine...")
    engine = TrainingEngine(args)

    if rank == 0:
        run_rank0_server(engine, args.port)
        logger.info(f"[Rank 0] Training server ready at http://localhost:{args.port}")
        rank0_loop(engine)
    else:
        worker_loop(engine)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
