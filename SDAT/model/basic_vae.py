import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import einops
from einops.layers.torch import Rearrange

__all__ = ['Encoder', 'Decoder', 'PatchwiseEmbedding1D', 'PatchwiseProjection1D']

class PatchwiseEmbedding1D(nn.Module):
    """Split 7-dim action into 3 modality branches and embed each to D_embed dimensions.
    
    Input: (B, T, 7) where 7 = [pos(3), rot(3), grip(1)]
    Output: (B, 3*D_embed, T) ready for grouped Conv1d
    """
    def __init__(self, d_embed: int = 64, dropout: float = 0.0, norm_type: str = 'layer'):
        super().__init__()
        self.d_embed = d_embed
        
        # Per-branch MLPs
        self.pos_embed = nn.Sequential(
            nn.Linear(3, d_embed),
            nn.GELU(),
            nn.LayerNorm(d_embed) if norm_type == 'layer' else nn.Identity(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, d_embed),
        )
        self.rot_embed = nn.Sequential(
            nn.Linear(3, d_embed),
            nn.GELU(),
            nn.LayerNorm(d_embed) if norm_type == 'layer' else nn.Identity(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, d_embed),
        )
        # Change to Embedding for gripper classification
        self.grip_embed = nn.Embedding(2, d_embed)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        """x: (B, T, 7)"""


        x_pos = x[..., :3]    # (B, T, 3)
        x_rot = x[..., 3:6]   # (B, T, 3)
        x_grip = x[..., 6]    # (B, T)
        
        e_pos = self.pos_embed(x_pos)    # (B, T, D)

        e_rot = self.rot_embed(x_rot)    # (B, T, D)

        
        # Gripper: Discretize > 0.5 -> 1, else 0
        x_grip_int = (x_grip > 0).long()
        e_grip = self.grip_embed(x_grip_int) # (B, T, D)

        
        # Stack and transpose to (B, 3*D, T)
        e_all = torch.cat([e_pos, e_rot, e_grip], dim=-1)  # (B, T, 3*D)
        return e_all.transpose(1, 2)  # (B, 3*D, T)

class PatchwiseProjection1D(nn.Module):
    """Mirror of PatchwiseEmbedding - project 3*D_embed back to 7-dim action.
    
    Input: (B, 3*D_embed, T) from decoder
    Output: (B, T, 8) -> [pos(3), rot(3), grip_logits(2)]
    """
    def __init__(self, d_embed: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_embed = d_embed
        
        # Per-branch projection heads
        self.pos_head = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, 3),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed, 3),
        )
        self.grip_head = nn.Sequential(
            nn.Linear(d_embed, d_embed // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5) if dropout > 1e-6 else nn.Identity(),
            nn.Linear(d_embed // 2, 2), # Output logits for 2 classes
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, z):
        """z: (B, 3*D, T)"""
        z = z.transpose(1, 2)  # (B, T, 3*D)
        D = self.d_embed
        
        z_pos = z[..., :D]
        z_rot = z[..., D:2*D]
        z_grip = z[..., 2*D:]
        
        out_pos = self.pos_head(z_pos)    # (B, T, 3)
        out_rot = self.rot_head(z_rot)    # (B, T, 3)
        out_grip = self.grip_head(z_grip) # (B, T, 2) - Logits
        
        return torch.cat([out_pos, out_rot, out_grip], dim=-1)  # (B, T, 8)

def nonlinearity(x):
    return x * torch.sigmoid(x)

def Normalize(in_dims, num_groups=8):
    if num_groups > in_dims:
        num_groups = 1
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_dims, eps=1e-6, affine=True)

class Upsample1D_2x(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dims, in_dims, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Downsample1D_2x(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_dims, in_dims, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_dims, out_channels=None, dropout):
        super().__init__()
        self.in_dims = in_dims
        out_channels = in_dims if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_dims)
        self.conv1 = torch.nn.Conv1d(in_dims, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_dims != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_dims, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h

class GroupedResnetBlock(nn.Module):
    """ResnetBlock with grouped convolutions for branch isolation."""
    def __init__(self, *, in_dims, out_channels=None, dropout, groups=3):
        super().__init__()
        self.in_dims = in_dims
        out_channels = in_dims if out_channels is None else out_channels
        self.out_channels = out_channels
        self.groups = groups

        self.norm1 = Normalize(in_dims)
        self.conv1 = torch.nn.Conv1d(in_dims, out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        if self.in_dims != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_dims, out_channels, kernel_size=1, stride=1, padding=0, groups=groups)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.C = in_dims
        self.norm = Normalize(in_dims)
        self.qkv = torch.nn.Conv1d(in_dims, 3 * in_dims, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_dims) ** (-0.5)
        self.proj_out = torch.nn.Conv1d(in_dims, in_dims, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=1), qkv)
        q = q * self.w_ratio
        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return x + self.proj_out(out)

def make_attn(in_dims, using_sa=True):
    return AttnBlock(in_dims) if using_sa else nn.Identity()

class Encoder(nn.Module):
    def __init__(
        self, *, ch=32, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_dims=3, z_channels, double_z=False,
        using_sa=True, using_mid_sa=True,
        patchwise_cfg: dict | None = None
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_dims = in_dims
        self.patchwise_cfg = patchwise_cfg or {'enable': False}
        
        # Grouped conv setup
        self.use_grouped = self.patchwise_cfg.get('enable', False)
        self.grouped_depth = self.patchwise_cfg.get('grouped_depth', 2) if self.use_grouped else 0
        
        # conv_in: if grouped, input is 3*D_embed; else original in_dims
        actual_in_dims = in_dims if not self.use_grouped else (3 * self.patchwise_cfg.get('d_embed', 64))
        self.conv_in = torch.nn.Conv1d(
            actual_in_dims, self.ch, 
            kernel_size=3, stride=1, padding=1,
            groups=3 if self.use_grouped else 1
        )

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                # Use grouped conv for early layers if patchwise enabled
                use_groups_here = self.use_grouped and i_level < self.grouped_depth
                if use_groups_here:
                    # Custom grouped ResnetBlock - simplified, just group the convs
                    block.append(GroupedResnetBlock(
                        in_dims=block_in, out_channels=block_out, 
                        dropout=dropout, groups=3
                    ))
                else:
                    block.append(ResnetBlock(in_dims=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1D_2x(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_dims=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_dims=block_in, out_channels=block_in, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # If patchwise, x is already (B, 3*D, T); else (B, T, C) -> (B, C, T)
        if not self.use_grouped:
            x = einops.rearrange(x, 'b h d -> b d h')

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):

                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))

        return h

class Decoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_dims=3, z_channels,
        using_sa=True, using_mid_sa=True,
        patchwise_cfg: dict | None = None
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_dims = in_dims
        self.patchwise_cfg = patchwise_cfg or {'enable': False}

        # Mirror encoder: grouped conv for early up-blocks
        self.use_grouped = self.patchwise_cfg.get('enable', False)
        self.grouped_depth = self.patchwise_cfg.get('grouped_depth', 2) if self.use_grouped else 0

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = torch.nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_dims=block_in, out_channels=block_in, dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(
            in_dims=block_in, out_channels=block_in, dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                # Mirror encoder: use grouped conv for LATE layers (i_level < grouped_depth)
                use_groups_here = self.use_grouped and i_level < self.grouped_depth
                if use_groups_here:
                    block.append(GroupedResnetBlock(
                        in_dims=block_in, out_channels=block_out, 
                        dropout=dropout, groups=3
                    ))
                else:
                    block.append(ResnetBlock(in_dims=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1D_2x(block_in)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        # conv_out: if grouped, output 3*D_embed; else in_dims
        actual_out_dims = in_dims if not self.use_grouped else (3 * self.patchwise_cfg.get('d_embed', 64))
        self.conv_out = torch.nn.Conv1d(
            block_in, actual_out_dims, 
            kernel_size=3, stride=1, padding=1,
            groups=3 if self.use_grouped else 1
        )

    def forward(self, z):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        # If patchwise, return (B, 3*D, T); else (B, T, C)
        if not self.use_grouped:
            h = einops.rearrange(h, 'b d h -> b h d')
        return h


