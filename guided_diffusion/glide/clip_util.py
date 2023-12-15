import clip

import torch
import torchvision

class clip_model_wrap():
    def __init__(self, model_name="ViT-B/32", device='cuda'):
        # text embed from clip model (ViT-B/32)
        self.clip_image_size = 224
        self.device = device
        model, preprocess = clip.load(model_name, device=device)
        self.clip_model = model.eval()
        self.resize = torchvision.transforms.Resize(224, interpolation=3)  # 3 means PIL.Image.BICUBIC
        self.centercrop = torchvision.transforms.CenterCrop(224)

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize01 = torchvision.transforms.Normalize(mean, std)
        mean11 = [2 * x - 1 for x in mean] # for range [-1,1]
        std11 = [2 * x for x in mean]
        self.normalize11 = torchvision.transforms.Normalize(mean11, std11)

        self.cosine_sim = torch.nn.CosineSimilarity()

    def clip_loss_img_txtembd(self, img, txt_embd, img_minmax=(-1,1)):
        # img in minmax range: [-1,1]
        img_embd = self.get_img_embd(img, img_minmax)
        assert img_embd.shape == txt_embd.shape, f'{img_embd.shape} =/= {txt_embd.shape}'
        loss = 1 - self.cosine_sim(img_embd, txt_embd)
        return loss

    @torch.no_grad()
    def get_img_embd(self, img, minmax=(-1,1)):
        if minmax == (-1,1):
            normalize = self.normalize11
        elif minmax == (0,1):
            normalize = self.normalize01
        else:
            print('Warning: unknown normalization for clip: ', minmax)
            normalize = self.normalize11
        img_clip_input = self.centercrop(self.resize(img))
        img_clip_input = normalize(img_clip_input)
        img_embd = self.clip_model.encode_image(img_clip_input)
        return img_embd

    @torch.no_grad()
    def get_txt_embd(self, text_list):
        # text list like ['dog', 'cat', 'face']
        text_input = clip.tokenize(text_list).to(self.device) # a face with long hair
        txt_embd = self.clip_model.encode_text(text_input)
        return txt_embd
