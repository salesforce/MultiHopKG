import torch
import pdb
import math
from typing import List, Optional
import os

import numpy as np
from torch import nn
from torch.nn import Embedding

from multihopkg.logging import setup_logger


class HunchLLM(torch.nn.Module):
    """
    Encoder with Cross Attention.
    Meant to be trained on embeddings from graph space and iteratively output new guesses at the answer.
    """

    def __init__(
        self,
        pretrained_transformer_weights_path: str,
        xattn_left_dim: int,
        llm_model_dim: int,
        llm_num_heads: int,
        llm_num_layers: int,
        llm_ff_dim: int,
        llm_max_seq_length: int,
        xattn_left_max_seq_length: int,
        dropout: float,
        embedding_padding_id: int,
        embedding_dim: int,
        embedding_vocab_size: int,
    ):
        """
        Args:
            - pretrained_language_model (torch.nn.Module): The pretrained language model to use.
            - input_dim (int): The input dimension of the projection layer on top of the pretrained language model.
        """
        super(HunchLLM, self).__init__()
        # Generally speaking it will just be a single extra layer on top of the pretrained language model

        # TODO: Do we want different times of dropout here?
        self.decoder_embedding = nn.Embedding(embedding_vocab_size, embedding_dim)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(llm_model_dim, llm_num_heads, llm_ff_dim, dropout) for _ in range(llm_num_layers)]
        )
        self.fc = nn.Linear(llm_model_dim, embedding_vocab_size)
        self.embedding_padding_id  =  embedding_padding_id
        self.pretrained_transformer = self._load_pretrained_model(pretrained_transformer_weights_path)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding_xattn_left = PositionalEncoding(xattn_left_dim, xattn_left_max_seq_length)
        self.positional_encoding_decoder = PositionalEncoding(llm_model_dim, llm_max_seq_length)
        
        ########################################
        # Load the pretrained model 
        ########################################

       #  # TODO: Recalculate this.
       #  self.pretrained_embedding = pretrained_embedding
       #  self.pretrained_embedding_dim = pretrained_embedding.config.hidden_size
       #  self.pretrained_embedding_padding_id = pretrained_embedding.config.pad_token_id
          
         
    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Args:
            - src (torch.Tensor): (batch_seq, history_length, history_dim) Would thend to be the current state history
            - tgt (toch.Tensor): (batch_seq, max_token_seq_len) Padded Answer
        Returns: 
            - attempt_at_answer (torch.Tensor): Self-descriptive name
        """


        ########################################
        # Then the rest of the code.
        ########################################
        _, tgt_mask = generate_mask(None, tgt, self.embedding_padding_id)
        batch_size = src.size(0)


        # TODO: Check this is correct.
        # TODO: If we are to use history_length dimension then we likely will have histories that have ended and others that have not. 
        src_mask = torch.ones((batch_size, 1, 1, src.size(1))).to(src.device) # Somthing like this but note quite sure.
        src_positioned = self.dropout(
            # TODO: Ensure the decoder is actually being done right instead of just not crashing on me
            # As in like the elements being summed properly
            self.positional_encoding_xattn_left(src)
        )
        tgt_embedded = self.dropout(
            self.positional_encoding_decoder(self.decoder_embedding(tgt))
        )

        enc_output = src_positioned
        dec_output = tgt_embedded
        for i, dec_layer in enumerate(self.decoder_layers):
            # self.logger.info(f"We are going through the {i}th layer of decoder ")
            dec_output = dec_layer(
                dec_output, enc_output, src_mask, tgt_mask, cross_attn=True
            )

        output = self.fc(dec_output)
        return output

    def _load_pretrained_model(self,weights_path: str, device_to_load_to: str = "cuda"):
        """
        We expect the paths to be in ckpt format
        """
        assert os.path.exists(weights_path), f"The weights path {weights_path} does not exist"
        # Create the modele itself
        checkpoint = torch.load(weights_path, map_location=device_to_load_to)
        state_dict = checkpoint['state_dict']
        self.decoder_embedding.load_state_dict({'weight' : state_dict['model.decoder_embedding.weight'],})
        decoder_state_dict = {
            k.replace("model.decoder_layers.", ""): v
            for k, v in state_dict.items()
            if "decoder_layers" in k
        }
        self.decoder_layers.load_state_dict(decoder_state_dict)

    # TODO: Load model state dict *carefully* we might not even need to load the Transformer model 


    def freeze_llm(self):
        """
        Will freeze the language model
        """
        for param in self.pretrained_language_model.parameters():
            param.requires_grad = False

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = model_dim
        self.num_heads = num_heads
        self.d_k = model_dim // num_heads

        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.logger = setup_logger(__class__.__name__)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # We want state but not back prop for positional encoding
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        max_length = x.shape[1]
        self.logger.debug(
            f"Positional Encoding is called with x of shape {x.shape} and pe of shape {self.pe.shape}"
        )
        # TODO: Feels like PE is upside down but I have to check on that.
        return x + self.pe[:, :max_length, :]



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask, cross_attn: bool):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        if cross_attn:
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout: float,
        padding_id: int,
        pretrained_embedding: Embedding,
    ):
        super(Transformer, self).__init__()
        self.logger = setup_logger(__class__.__name__)
        # Used Pretrained embeddings
        self.encoder_embedding = pretrained_embedding
        self.decoder_embedding = pretrained_embedding
        self.padding_id = padding_id

        # TODO: pretrain this one
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        tgt_vocab_size = pretrained_embedding.num_embeddings
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, tgt):
        src_mask, tgt_mask = generate_mask(src, tgt, self.padding_id)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for i, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for i, dec_layer in enumerate(self.decoder_layers):
            # self.logger.info(f"We are going through the {i}th layer of decoder ")
            dec_output = dec_layer(
                dec_output, enc_output, src_mask, tgt_mask, cross_attn=True
            )

        output = self.fc(dec_output)
        return output

def collate_token_ids_batch(batch: List[np.ndarray]) -> torch.Tensor:
    """
    Will take a list of token ids and return a tensor of shape (batch_size, seq_len)
    """
    max_seq_len = max([len(x) for x in batch])
    batch_token_ids = torch.zeros((len(batch), max_seq_len))
    for i, x in enumerate(batch):
        batch_token_ids[i, :len(x)] = torch.tensor(x)
    return batch_token_ids

def generate_mask(src: Optional[torch.Tensor], tgt: torch.Tensor, padding_id):
    # src_mask : (batch_len x seq_len) -> (batch_len x 1 x 1 x seq_len)
    if src is not None:
        src_mask = (src != padding_id).unsqueeze(1).unsqueeze(2)
    else:
        src_mask = None
    # tgt_mask : (batch_len x seq_len) -> (batch_len x 1 x seq_len x 1)
    tgt_mask = (tgt != padding_id).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (
        (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1))
        .bool()
        .to(src.device)
    )
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

class PathCrossAttentionTransformer(nn.Module):
    """
    We expect this one to be pretrained on language. 
    We will override its cross attention to forget language and understand paths.
    """
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout: float,
        padding_id: int,
        pretrained_embedding: Embedding,
    ):
        super(PathCrossAttentionTransformer, self).__init__()

        ########################################
        # Everything will be loaded from the pretrained embedding
        ########################################
        self.logger = setup_logger(__class__.__name__)
        # Used Pretrained embeddings
        self.encoder_embedding = pretrained_embedding
        self.decoder_embedding = pretrained_embedding
        self.padding_id = padding_id

        # TODO: pretrain this one
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        tgt_vocab_size = pretrained_embedding.num_embeddings
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, tgt):
        src_mask, tgt_mask = generate_mask(src, tgt, self.padding_id)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for i, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for i, dec_layer in enumerate(self.decoder_layers):
            # self.logger.info(f"We are going through the {i}th layer of decoder ")
            dec_output = dec_layer(
                dec_output, enc_output, src_mask, tgt_mask, cross_attn=True
            )

        output = self.fc(dec_output)
        return output

def collate_token_ids_batch(batch: List[np.ndarray]) -> torch.Tensor:
    """
    Will take a list of token ids and return a tensor of shape (batch_size, seq_len)
    """
    max_seq_len = max([len(x) for x in batch])
    batch_token_ids = torch.zeros((len(batch), max_seq_len))
    for i, x in enumerate(batch):
        batch_token_ids[i, :len(x)] = torch.tensor(x)
    return batch_token_ids

def generate_mask(src: Optional[torch.Tensor], tgt: torch.Tensor, padding_id):
    # src_mask : (batch_len x seq_len) -> (batch_len x 1 x 1 x seq_len)

    assert tgt.size(0)
    if src is not None:
        src_mask = (src != padding_id).unsqueeze(1).unsqueeze(2)
    else:
        src_mask = None

    # tgt_mask : (batch_len x seq_len) -> (batch_len x 1 x seq_len x 1)
    tgt_mask = (tgt != padding_id).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (
        (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1))
        .bool()
        .to(tgt.device)
    )
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask.to(torch.int32)



