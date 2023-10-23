import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

# NEW
from .bert_model import BertCrossLayer


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

# NEW
class GILEHead(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.dense_d = nn.Linear(input_size, hidden_size)
        self.dense_l = nn.Linear(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, data_feats, label_feats):
        data_feats = self.activation(self.dense_d(data_feats))
        label_feats = self.activation(self.dense_l(label_feats))
        feats = (data_feats.unsqueeze(-1).expand(-1, -1, -1, label_feats.shape[0]) * \
                label_feats.T.expand(data_feats.shape[0], data_feats.shape[1], -1, -1))
        logits = self.dense(feats.permute(0,1,3,2)).squeeze(-1)
        return logits

# NEW
class Fuser(nn.Module):
    def __init__(self, config, fuser_hidden_size, meta_hidden_size, num_layer):
        super().__init__()

        self.fc = nn.Linear(fuser_hidden_size, config.hidden_size)
        self.attention_layers = nn.ModuleList([
            BertCrossLayer(config) for _ in range(num_layer)
        ])
        self.fcs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size) for _ in range(num_layer)
        ])
        self.final = nn.Linear(config.hidden_size, meta_hidden_size)
        self.num_layer = num_layer

    def forward(
        self,
        x,
        x_f,
        x_masks,
        x_f_masks,
    ):
        x, x_f = self.fc(x), self.fc(x_f)
        for i in range(self.num_layer):
            x = x + self.attention_layers[i](
                hidden_states=x,
                encoder_hidden_states=torch.cat([x_f, x], dim=1),
                encoder_attention_mask=torch.cat([x_f_masks, x_masks], dim=-1),
            )[0]

            x = x + self.fcs[i](x)
        x = self.final(F.relu(x))

        return x
