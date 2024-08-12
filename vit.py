import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from timm.models.vision_transformer import Block, Attention


class MaskedAttention(nn.Module):
    def __init__(
            self,
            original_module: Attention,
    ) -> None:
        super(MaskedAttention, self).__init__()
        self.original_module = original_module
        self.num_heads = original_module.num_heads
        self.head_dim = original_module.head_dim
        self.scale = original_module.scale
        self.qkv = original_module.qkv
        self.q_norm = original_module.q_norm
        self.k_norm = original_module.k_norm
        self.attn_drop = original_module.attn_drop
        self.proj = original_module.proj
        self.proj_drop = original_module.proj_drop
        self.mask = nn.Parameter(torch.ones(1, 1, 1, 1, self.head_dim), requires_grad=True)

    # @get_local('mask_out')
    def forward(self, mask_attn, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = qkv * self.mask   # Here is the mask
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        mask_attn, mask = mask_attn
        attn = mask * mask_attn + attn * (1 - mask)
        attn = F.normalize(attn, p=1, dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TiViT(nn.Module):
    # @get_local('attn_lst')
    def __init__(self, num_classes, masked = False):
        super(TiViT, self).__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)
        
            # print(name, module)
        if masked:
            for name, module in self.vit.named_modules():
                # print(name, module) .
                if isinstance(module, Block):
                    attn = module.attn
                    # print(attn.num_heads)
                    # print(attn.head_dim)
                    masked_attn = MaskedAttention(attn)
                    # new_module.load_state_dict(module.state_dict(), strict=False)
                    module.attn = masked_attn
                    # attn_lst = [attn, masked_attn]
                    # del module

    def forward(self, x, mask_attn=None):
        # x = self.vit(x)
        if mask_attn:
            if mask_attn[0]:
                mask_attn = mask_attn[1]
                # print(len(mask_attn))
        else:
            mask_attn = (torch.zeros((197, 197), device="cuda:0"), torch.zeros((197, 197), device="cuda:0"))
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.pos_drop(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        for block in self.vit.blocks:
            x = x + block.drop_path1(block.ls1(block.attn(mask_attn, block.norm1(x))))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        x = self.vit.norm(x)
        x = x[:, 0]
        x = self.vit.fc_norm(x)
        x = self.vit.head_drop(x)
        x = self.vit.head(x)
        x = self.fc(x)

        return x

    def get_masks(self):
        return [param for name, param in self.named_parameters() if 'mask' in name]

    def get_params(self):
        return [param for name, param in self.named_parameters() if 'mask' not in name]
    
    def clip_masks(self, threshold = 1):
        with torch.no_grad():
            for mask in self.get_masks():
                mask.data.clamp_(0, threshold)
    
    def reset_masks(self):
        with torch.no_grad():
            for mask in self.get_masks():
                mask.data.fill_(1)

    # def save_mask_scores(self, filename="mask_scores.pth"):
    #     pass
        

if __name__ == '__main__':
    print(timm.list_models('*vit*'))