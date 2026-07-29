import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ncps.torch import CfC
from ncps.wirings import AutoNCP

class RLTwoLayerCfcModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        custom_config = model_config.get("custom_model_config", {})
        config_device = custom_config.get("device", "cuda:0")
        
        if "cuda" in config_device:
            try:
                test_tensor = torch.zeros(1, device=config_device)
                self.target_device = torch.device(config_device)
                
            except Exception:
                self.target_device = torch.device("cpu")
                
        else:
            self.target_device = torch.device("cpu")

        print(f"📡 [ВОРКЕР RAY] Модель успешно инициализирована на устройстве: {self.target_device}")

        self.visual_accuracy = 8
        self.diamond_size = 61
        self.FE_hidden_size = 64
        self.CfC_input_size = 48
        
        self.CfC_hidden_size = 256
        self.CfC_output_size = 128

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.visual_accuracy, self.visual_accuracy)),
            
            nn.Flatten(),

            nn.Linear(32 * self.visual_accuracy * self.visual_accuracy, self.CfC_input_size),
            nn.ReLU()
        ).to(self.target_device)

        self.wiring = AutoNCP(self.CfC_hidden_size, self.CfC_output_size)

        self.cfc = CfC(
            self.CfC_input_size,
            self.wiring,
            batch_first=True,
            mixed_memory=True
        ).to(self.target_device)
        
        if hasattr(self.cfc, "cell"):
            self.cfc.cell.to(self.target_device)

        self.norm = nn.LayerNorm(self.CfC_output_size).to(self.target_device)

        self.action_head = nn.Linear(self.CfC_output_size, num_outputs).to(self.target_device)
        self.value_head = nn.Linear(self.CfC_output_size, 1).to(self.target_device)

        self._cur_value = None
        self._last_output = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        device = self.target_device

        H, W = obs.shape[-2], obs.shape[-1]

        if obs.dim() == 5:
            B, T, C, _, _ = obs.shape
            obs_square = obs.view(B * T, 1, H, W).contiguous()

        elif obs.dim() == 4:
            B = obs.shape[0]
            T = 1
            C = obs.shape[1]

            obs_square = obs.view(B * T, 1, H, W).contiguous()

        else:
            obs_flat = obs.view(-1, 1, H, W)
            B = obs_flat.shape[0]
            T = 1
            obs_square = obs_flat.contiguous()

        feature_flat = self.feature_extractor(obs_square)
        feature = feature_flat.view(B, T, self.CfC_input_size).contiguous()

        if state and len(state) >= 2:
            s0 = state[0].float().to(device)
            s1 = state[1].float().to(device)

            hx = s0.unsqueeze(0) if s0.dim() == 1 else s0
            cx = s1.unsqueeze(0) if s1.dim() == 1 else s1

            if hx.shape[0] != B:
                if hx.shape[0] < B:
                    diff = B - hx.shape[0]
                    padding = torch.zeros((diff, self.CfC_hidden_size), device=device)
                    hx = torch.cat([hx, padding], dim=0)
                    cx = torch.cat([cx, padding], dim=0)
                else:
                    hx = hx[:B].contiguous()
                    cx = cx[:B].contiguous()
        else:
            hx = torch.zeros((B, self.CfC_hidden_size), device=device)
            cx = torch.zeros((B, self.CfC_hidden_size), device=device)

        out_seq, (hx_next, cx_next) = self.cfc(feature, (hx, cx))
        next_state = [hx_next.contiguous(), cx_next.contiguous()]

        if seq_lens is not None and T > 1:
            max_len = out_seq.size(1)
            col_indices = torch.arange(max_len, device=device).unsqueeze(0)
            lens_expanded = seq_lens.unsqueeze(1)
            mask = col_indices < lens_expanded
            out_seq = out_seq * mask.unsqueeze(-1).float()

        flat_out = out_seq.contiguous().view(B * T, -1)

        self._last_output = flat_out

        logits = self.action_head(flat_out)
        values = self.value_head(flat_out)

        self._cur_value = values.squeeze(-1)

        if seq_lens is not None and T > 1:
            flat_mask = mask.view(-1).float()
            self._cur_value = self._cur_value * flat_mask

        return logits, next_state

    def last_output(self):
        return self._last_output if hasattr(self, "_last_output") else None

    def is_time_major(self) -> bool:
        return False

    def get_initial_state(self):
        return [
            torch.zeros(self.CfC_hidden_size),
            torch.zeros(self.CfC_hidden_size)
        ]

    def value_function(self):
        return self._cur_value
