import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ncps.torch import CfC

class RLTwoLayerCfcModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.diamond_size = 61
        self.FE_hidden_size = 64
        self.CfC_input_size = 48
        self.CfC_hidden_size = 128
        self.CfC_output_size = 128

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten(),

            nn.Linear(32 * 5 * 5, self.CfC_input_size),
            nn.ReLU()
        )


        self.cfc1 = CfC(
            input_size=self.CfC_input_size,
            units=self.CfC_hidden_size,
            batch_first=True,
            backbone_layers=2,
            backbone_units=64,
            backbone_dropout=0.1,
            mixed_memory=True
        )

        self.cfc2 = CfC(
            input_size=self.CfC_hidden_size,
            units=self.CfC_output_size,
            batch_first=True,
            backbone_layers=2,
            backbone_units=64,
            backbone_dropout=0.1,
            mixed_memory=True
        )

        self.norm = nn.LayerNorm(self.CfC_output_size)

        self.action_head = nn.Linear(self.CfC_output_size, num_outputs)
        self.value_head = nn.Linear(self.CfC_output_size, 1)

        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        device = obs.device

        H, W = obs.shape[-2], obs.shape[-1]

        if obs.dim() == 4:
            B, T, _, _ = obs.shape

            obs_square = obs.reshape(B * T, 1, H, W)
            
        elif obs.dim() == 3:
            B = obs.shape[0]
            T = 1
            obs_square = obs.unsqueeze(1)
            
        else:
            obs_flat = obs.reshape(-1, H, W)
            B = obs_flat.shape[0]
            T = 1
            obs_square = obs_flat.unsqueeze(1)
            
        feature_flat = self.feature_extractor(obs_square)

        feature = feature_flat.view(B, T, self.CfC_input_size)

        if state and len(state) >= 4:
            hx1 = state[0].float().to(device)
            cx1 = state[1].float().to(device)
            hx2 = state[2].float().to(device)
            cx2 = state[3].float().to(device)
            
            if hx1.shape[0] != B:
                hx1 = hx1[0:1].repeat(B, 1).contiguous()
                cx1 = cx1[0:1].repeat(B, 1).contiguous()
                hx2 = hx2[0:1].repeat(B, 1).contiguous()
                cx2 = cx2[0:1].repeat(B, 1).contiguous()
        else:
            hx1 = torch.zeros((B, self.CfC_hidden_size), device=device)
            cx1 = torch.zeros((B, self.CfC_hidden_size), device=device)
            hx2 = torch.zeros((B, self.CfC_output_size), device=device)
            cx2 = torch.zeros((B, self.CfC_output_size), device=device)

        state_cfc1 = (hx1.contiguous(), cx1.contiguous())
        out_seq, (hx1_next, cx1_next) = self.cfc1(feature, state_cfc1)
        
        state_cfc2 = (hx2.contiguous(), cx2.contiguous())
        out_seq, (hx2_next, cx2_next) = self.cfc2(out_seq, state_cfc2)

        out_seq = self.norm(out_seq)
        flat_out = out_seq.reshape(B * T, -1)

        logits = self.action_head(flat_out)
        values = self.value_head(flat_out)
        self._cur_value = values.view(B * T)

        return logits, [hx1_next, cx1_next, hx2_next, cx2_next]


    def get_initial_state(self):
        return [
            torch.zeros(self.CfC_hidden_size),
            torch.zeros(self.CfC_hidden_size),
            torch.zeros(self.CfC_output_size),
            torch.zeros(self.CfC_output_size)
        ]

    def value_function(self):
        return self._cur_value
