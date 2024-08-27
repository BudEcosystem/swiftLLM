import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        # Slice to get the last token embedding for each request
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
        
        if infer_state.do_sample:
            if infer_state.temperature is not None:
                scores = self._get_temperature_logits(logits, infer_state.temperature)
            if infer_state.top_k is not None:
                scores = self._get_topk_logits(scores, infer_state.top_k)
            if infer_state.top_p is not None:
                scores = self._get_topp_logits(scores, infer_state.top_p)

            probs = torch.nn.functional.softmax(scores, dim=1)
            output_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            output_tokens = torch.argmax(logits, dim=1)

        
        return output_tokens

    def _get_temperature_logits(self, scores: torch.FloatTensor, temperature: float) -> torch.FloatTensor:
        scores_processed = scores / temperature
        return scores_processed

    def _get_topp_logits(self, scores: torch.FloatTensor, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")
        
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, filter_value)
        return scores_processed

    def _get_topk_logits(self, scores: torch.FloatTensor, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        top_k = min(top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, filter_value)
        return scores_processed
