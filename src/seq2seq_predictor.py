"""
seq2seq_predictor.py - 배수지 13(j_resv) 전용 Seq2Seq+Attention 모델
"""

import torch
import torch.nn as nn


class LSTMSeq2SeqAttnModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2,
                 output_size=15, embed_dim=16, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.step_embedding = nn.Embedding(output_size, embed_dim)
        self.attn_Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_Ws = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_V = nn.Linear(hidden_size, 1, bias=False)
        self.decoder = nn.LSTMCell(
            input_size=hidden_size + embed_dim, hidden_size=hidden_size
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        enc_outputs, (h_n, c_n) = self.encoder(x)
        enc_keys_projected = self.attn_Ws(enc_outputs)
        h_dec = h_n[-1]
        c_dec = c_n[-1]
        predictions = []
        step_ids = torch.arange(self.output_size, device=x.device)
        step_embs = self.step_embedding(step_ids)

        for t in range(self.output_size):
            query = self.attn_Wh(h_dec).unsqueeze(1)
            energy = torch.tanh(query + enc_keys_projected)
            score = self.attn_V(energy).squeeze(-1)
            attn_weights = torch.softmax(score, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
            step_emb = step_embs[t].unsqueeze(0).expand(batch_size, -1)
            dec_input = torch.cat([context, step_emb], dim=1)
            h_dec, c_dec = self.decoder(dec_input, (h_dec, c_dec))
            pred_t = self.fc_out(self.layer_norm(h_dec))
            predictions.append(pred_t)

        return torch.cat(predictions, dim=1)
