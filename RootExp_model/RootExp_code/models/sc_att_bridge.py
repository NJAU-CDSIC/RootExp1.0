"""
SC_Att_Bridge – Spatial-Channel Attention Bridge for encoder skip connections.
Applies spatial attention (shared conv) then channel attention (cross-level FC)
to all skip connections before they enter the decoder.

Adapted from H-vmunet / RootExp v8 to work with variable number of skip levels.
"""
import torch
import torch.nn as nn


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.Sigmoid(),
        )

    def forward(self, feats):
        out = []
        for t in feats:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = self.shared_conv2d(torch.cat([avg_out, max_out], dim=1))
            out.append(att)
        return out


class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list)
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.atts = nn.ModuleList([
            nn.Linear(c_list_sum, ch) if split_att == 'fc'
            else nn.Conv1d(c_list_sum, ch, 1)
            for ch in c_list
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        pooled = [self.avgpool(t) for t in feats]
        att = torch.cat(pooled, dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        out = []
        for i, att_layer in enumerate(self.atts):
            a = self.sigmoid(att_layer(att))
            if self.split_att == 'fc':
                a = a.transpose(-1, -2).unsqueeze(-1).expand_as(feats[i])
            else:
                a = a.unsqueeze(-1).expand_as(feats[i])
            out.append(a)
        return out


class SC_Att_Bridge(nn.Module):
    """Spatial + Channel attention bridge for encoder skip connections."""

    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, feats):
        residuals = list(feats)
        s_atts = self.satt(feats)
        feats = [s * t for s, t in zip(s_atts, feats)]
        post_spatial = list(feats)
        feats = [t + r for t, r in zip(feats, residuals)]
        c_atts = self.catt(feats)
        feats = [c * t for c, t in zip(c_atts, feats)]
        return [t + r for t, r in zip(feats, post_spatial)]
