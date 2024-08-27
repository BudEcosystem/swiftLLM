import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_quick_gelu(
    x: torch.Tensor,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr
):

    my_token_id = tl.program_id(0).to(tl.int64)
    my_block_id = tl.program_id(1)

    offs = my_token_id*hidden_size + my_block_id*block_size + tl.arange(0, block_size)
    gel = tl.load(x + offs)
    out = gel * tl.sigmoid(1.702 * gel)

    tl.store(x + offs, out)


def quick_gelu(x: torch.Tensor):

    assert x.is_contiguous()
    num_tokens = x.shape[1]
    hidden_size = x.shape[2]
    
    block_size = 256
    assert hidden_size % block_size == 0
    _fwd_quick_gelu[(num_tokens, hidden_size // block_size)](x, hidden_size, block_size)