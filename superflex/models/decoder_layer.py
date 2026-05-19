import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from torch import Tensor
from typing import Callable, Optional, Union

class DecoderLayer(TransformerDecoderLayer):
    """MultiheadAttention with thresholding"""
    
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        swapped_attention=False):
        super(DecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        self.swapped_attention = swapped_attention

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            
            if self.swapped_attention:  
                x = x + self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
                x = x + self._sa_block(
                    self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
                )
            else: 
                x = x + self._sa_block(
                    self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
                )
                x = x + self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            x = x + self._ff_block(self.norm3(x))
        else:
            if self.swapped_attention:
                x = self.norm2(
                    x
                    + self._mha_block(
                        x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                    )
                )
                x = self.norm1(
                    x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                )
            else :
                x = self.norm1(
                    x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
                )
                x = self.norm2(
                    x
                    + self._mha_block(
                        x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                    )
                )
            x = self.norm3(x + self._ff_block(x))

        return x