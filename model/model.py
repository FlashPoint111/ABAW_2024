import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary

from Whisper import Encoder, Decoder
from vit_clip import ViT_CLIP


class MultiModalModel(nn.Module):
    def __init__(self,
                 config=None,
                 temp=0.07,
                 n_mels=80,
                 n_ctx=240,
                 init_deit=True):
        super().__init__()

        # self.visual_encoder = VisionTransformer(img_size=224, patch_size=(4,16,16), embed_dim=768, depth=12, num_heads=12,
        #                                     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # if init_deit:
        #    checkpoint = torch.hub.load_state_dict_from_url(
        #        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #        map_location="cpu", check_hash=True)
        #    state_dict = checkpoint["model"]
        #    new_embed = state_dict['pos_embed'][:, 1:, :]
        #    #pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
        #    state_dict['img_embed'] = new_embed
        #    state_dict.pop('pos_embed')
        #    state_dict['patch_embed.proj.weight'] = torch.cat([state_dict['patch_embed.proj.weight'].unsqueeze(2)] * 4, dim=2)
        #    self.visual_encoder.load_state_dict(state_dict,strict=False)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visual_encoder = ViT_CLIP(224, 8, 16, 768, 12, 8, 0.2)
        self.visual_encoder.init_weights(pretrained='clip')
        for name, param in self.visual_encoder.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
                param.requires_grad = False

        self.audio_encoder = Encoder(n_mels=n_mels, n_ctx=n_ctx, n_state=768, n_head=8, n_layer=6)
        checkpoint_file = 'small.en.pt'
        with (open(checkpoint_file, "rb")) as fp:
            checkpoint = torch.load(fp)
        encoder_dict = {key.replace('encoder.', ''): value for key, value in checkpoint['model_state_dict'].items() if
                        'encoder' in key}
        encoder_dict['positional_embedding'] = encoder_dict['positional_embedding'][:n_ctx, :]
        self.audio_encoder.load_state_dict(encoder_dict, strict=False)

        self.decoder = Decoder(n_state=768, n_head=8, n_layer=6)
        # decoder_dict = {key.replace('decoder.', ''): value for key, value in checkpoint['model_state_dict'].items() if 'decoder' in key}
        # self.decoder.load_state_dict(decoder_dict, strict=False)

        # self.visual_encoder_m = VisionTransformer(img_size=224, patch_size=(4,16,16), embed_dim=768, depth=12, num_heads=12,
        #                                     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # self.audio_encoder_m = Encoder(n_mels=n_mels, n_ctx=n_ctx, n_state=768, n_head=8, n_layer=6)

        # self.vision_proj = nn.Linear(768, 256)
        # self.audio_proj = nn.Linear(768, 256)
        # self.vision_proj_m = nn.Linear(768, 256)
        # self.audio_proj_m = nn.Linear(768, 256)

        # self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
        #                    [self.vision_proj,self.vision_proj_m],
        #                    [self.audio_encoder,self.audio_encoder_m],
        #                    [self.audio_proj,self.audio_proj_m],
        #                   ]
        # self.copy_params()

        # self.norm = nn.LayerNorm(768, 1e-6)
        self.temp = nn.Parameter(torch.ones([]) * temp)
        # self.queue_size = 65536
        # self.momentum = 0.995
        # self.register_buffer("image_queue", torch.randn(256, self.queue_size))
        # self.register_buffer("audio_queue", torch.randn(256, self.queue_size))
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # self.audio_queue = nn.functional.normalize(self.audio_queue, dim=0)

        # self.avgpooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.output_layer1 = nn.Linear(768, 256)
        self.output_layer2 = nn.Linear(256, 64)
        self.output_layer3 = nn.Linear(64, 8)

    def forward(self, image, audio, label, alpha=0.4):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)
        audio_embeds = self.audio_encoder(audio)

        # image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]),dim=-1)
        # audio_feat = F.normalize(self.audio_proj(audio_embeds[:, 0, :]),dim=-1)

        # with torch.no_grad():
        #    self._momentum_update()
        #    image_embeds_m = self.visual_encoder_m(image)
        #    audio_embeds_m = self.audio_encoder_m(audio)

        #    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]),dim=-1)  
        #    audio_feat_m = F.normalize(self.audio_proj_m(audio_embeds_m[:, 0, :]),dim=-1)  

        #    image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
        #    audio_feat_all = torch.cat([audio_feat_m.t(), self.audio_queue.clone().detach()], dim=1)

        #    sim_i2a_m = image_feat_m @ audio_feat_all / self.temp 
        #    sim_a2i_m = audio_feat_m @ image_feat_all / self.temp

        #    sim_targets = torch.zeros(sim_i2a_m.size()).to(image.device)
        #    sim_targets.fill_diagonal_(1)   

        #    sim_i2a_targets = alpha * F.softmax(sim_i2a_m, dim=1) + (1 - alpha) * sim_targets
        #    sim_a2i_targets = alpha * F.softmax(sim_a2i_m, dim=1) + (1 - alpha) * sim_targets 

        # sim_i2t = image_feat @ audio_feat_all / self.temp
        # sim_t2i = audio_feat @ image_feat_all / self.temp

        # loss_i2a = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2a_targets,dim=1).mean()
        # loss_a2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_a2i_targets,dim=1).mean()

        # loss_ita = (loss_i2a+loss_a2i)/2   #ITC loss function

        # self._dequeue_and_enqueue(image_feat_m, audio_feat_m)

        fused_feat = self.decoder(image_embeds, audio_embeds)
        fc1 = F.relu(self.output_layer1(fused_feat))
        fc2 = F.relu(self.output_layer2(fc1))
        output = torch.transpose(self.output_layer3(fc2), 1, 2)
        # output = F.softmax(fc3, dim=2)

        loss = F.cross_entropy(output, label)
        # torch.square(output - label).mean()

        return loss

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, audio_feat):
        # gather keys before updating queue
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)

        batch_size = image_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
        self.audio_queue[:, ptr:ptr + batch_size] = audio_feat.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    test = MultiModalModel()
    image = torch.zeros((2, 3, 32, 224, 224))
    audio = torch.zeros((2, 80, 480))
    label = torch.zeros((2, 32), dtype=torch.int64)
    summary(test, image, audio, label)
