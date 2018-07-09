"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from .decoder import DecoderState
from ..utils.misc import aeq
from ..modules.position_ffn import PositionwiseFeedForward
from .. import modules

from pdb import set_trace


MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      droput (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn_type="scaled-dot", keep_attn=False):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn_type = self_attn_type

        if self_attn_type == "scaled-dot":
            self.self_attn = modules.MultiHeadedAttention(
                heads, d_model, dropout=dropout)
        elif self_attn_type == "average":
            self.self_attn = modules.AverageAttention(
                d_model, dropout=dropout)

        self.context_attn = modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout, keep_attn=keep_attn)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = modules.LayerNorm(d_model)
        self.layer_norm_2 = modules.LayerNorm(d_model)

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        if self.self_attn_type == "scaled-dot":
            query, attn = self.self_attn(all_input, all_input, input_norm,
                                         mask=dec_mask)
        elif self.self_attn_type == "average":
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, all_input

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """
    def __init__(self, num_layers, d_model, heads, d_ff, attn_type,
                 copy_attn, self_attn_type, dropout, embeddings,
                 keep_attn=False):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             self_attn_type=self_attn_type, keep_attn=keep_attn)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = modules.GlobalAttention(
                d_model, attn_type=attn_type)
            self._copy = True
        self.layer_norm = modules.LayerNorm(d_model)


    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None):
        """
        See :obj:`modules.RNNDecoderBase.forward()`
        """
        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        saved_inputs = []
        saved_attns = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.transformer_layers[i](output, src_memory_bank,
                                             src_pad_mask, tgt_pad_mask,
                                             previous_input=prev_layer_input,
                                             layer_cache=cache["layer_{}".
                                                               format(i)]
                                             if cache is not None else None,
                                             step=step)
            if self.transformer_layers[i].context_attn.keep_attn:
                saved_attns.append(self.transformer_layers[i].context_attn.attn[:, :, :, :])
            saved_inputs.append(all_input)


        saved_inputs = torch.stack(saved_inputs)


        if len(saved_attns) != 0:
            saved_attns = torch.stack(saved_attns)
        if state.context_attn is not None:
            saved_attns = torch.cat((state.context_attn, saved_attns), dim=3)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()


        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        if len(saved_attns) == 0:
            state = state.update_state(tgt, saved_inputs)
        else:
            state = state.update_state(tgt, saved_inputs, saved_attns)
        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        state._init_cache(memory_bank, self.num_layers)
        return TransformerDecoderState(src)


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.context_attn = None
        self.cache = None


    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if self.context_attn is not None:
            return (self.previous_input, self.previous_layer_inputs, self.src, self.context_attn)
        else:
            return (self.previous_input, self.previous_layer_inputs, self.src)

    def detach(self):
        self.previous_input = self.previous_input.detach()
        self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs, context_attn=None):
        """ Called for every decoder forward pass. """
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        if context_attn is not None:
            state.context_attn = context_attn
        return state

    def _init_cache(self, memory_bank, num_layers):
        cache = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)
        for l in range(num_layers):
            layer_cache = {"prev_g": torch.zeros((batch_size, 1, depth))}
            cache["layer_{}".format(l)] = layer_cache
        return cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if isinstance(v, dict):
                    _recursive_map(v)
                else:
                    struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 1)
        if self.previous_input is not None:
            self.previous_input = fn(self.previous_input, 1)
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = fn(self.previous_layer_inputs, 1)
        if self.cache is not None:
            _recursive_map(self.cache)
