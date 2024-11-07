import torch
import pdb
import math

from torch import nn
from torch.nn import Embedding

from multihopkg.logging import setup_logger


class HunchLLM(torch.nn.Module):
    """
    Will provide a hunch at the right answer prematurely to enable use of soft rewards. 
    """
    def __init__(self, pretrained_language_model: torch.nn.Module, input_dim: int):
        """
        Args:
            - pretrained_language_model (torch.nn.Module): The pretrained language model to use.
            - input_dim (int): The input dimension of the projection layer on top of the pretrained language model.
        """
        super(HunchLLM, self).__init__()
        # Generally speaking it will just be a single extra layer on top of the pretrained language model
        self.input_dim = input_dim
        self.pretrained_llm_dim = pretrained_language_model.config.hidden_size
        self.projection = torch.nn.Linear(self.input_dim, self.pretrained_llm_dim)
        self.pretrained_language_model = pretrained_language_model

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - observations (torch.Tensor): The observations at the current state. Shape: (batch_size, observation_dim)
        Returnd:
            - logits (torch.Tensor): The logits of the language model. Shape: (batch_size, vocab_size)
        """
        pdb.set_trace()
        X = self.projection(observations)
        pdb.set_trace()
        return self.pretrained_language_model(X) 

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

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe


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

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        # src_mask : (batch_len x seq_len) -> (batch_len x 1 x 1       x seq_len)
        src_mask = (src != self.padding_id).unsqueeze(1).unsqueeze(2)
        # tgt_mask : (batch_len x seq_len) -> (batch_len x 1 x seq_len x 1)
        tgt_mask = (tgt != self.padding_id).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1))
            .bool()
            .to(src.device)
        )
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
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

