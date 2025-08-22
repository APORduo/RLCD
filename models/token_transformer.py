import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_,LayerScale
import math
from timm.models.vision_transformer import VisionTransformer,PatchEmbed
class TokViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None,init_values=None
                ):

        print('Using Token model')
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer,init_values=init_values)
        
        self.Prompt_Token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        trunc_normal_(self.Prompt_Token, std=0.01)
        #self.head = nn.Identity()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        Bs,nc,w,h = x.shape
        x = self.patch_embed(x)
        #breakpoint()
        #x = x + self.interpolate_pos_encoding(x, w, h)
        x = self._pos_embed(x)
        

        for i in range(len(self.blocks)-1):
            x = self.blocks[i](x)

        token_x = torch.cat((x, self.Prompt_Token.expand(x.shape[0], -1, -1)), dim=1)# concat to the last one
        token_x = self.blocks[-1](token_x)
        cls_x = token_x
       
          
        
        return cls_x , token_x

        
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.Prompt_Token.requires_grad_(True)
        self.blocks[-1].requires_grad_(True)

    
    
    
    def forward(self, x):
        
        cls_x , token_x= self.forward_features(x)
      
        
        return cls_x[:, 0], token_x[:, -1]
    
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x
    
       
