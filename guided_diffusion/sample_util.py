import torch
import clip
import random

def process1(kw):
    img2 = kw['img2'][5]
    f1 = kw['clip_feat'][5]
    f2 = kw['clip_feat2'][5]
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    text = clip.tokenize(["an image of a face with a blonde hair and with glasses"]).to('cpu') # of a woman with a blonde hair and
    text = clip.tokenize(["an image of a face with blue eyes"]).to('cpu') # of a woman with a blonde hair and
    f1 = model.encode_text(text)
    img2_new = []
    f1_new = []
    f2_new = []
    for i in range(8):
        fac = i/7
        img2_new.append(img2)
        f1_new.append(f1*fac + f2)
        f2_new.append(f2)

    kw2 = {}
    kw2['img2'] = torch.stack(img2_new, 0)
    kw2['clip_feat'] = torch.stack(f1_new, 0)
    kw2['clip_feat2'] = torch.stack(f2_new, 0)
    return kw2

def process2(kw):
    img2 = kw['img2'][5]
    #f1 = kw['clip_feat'][5]
    f2 = kw['clip_feat2'][5]
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    text = clip.tokenize(["an image of a face with glasses"]).to('cpu')
    f1 = model.encode_text(text)
    text2 = clip.tokenize(["an image of a face"]).to('cpu')
    f1_2 = model.encode_text(text2)
    kw['clip_feat'] = (f1)
    #kw['clip_feat2'] = torch.zeros_like(kw['clip_feat2'])
    return kw

def add_delta(kw):
    # img2 = kw['img2'][5]
    # f1 = kw['clip_feat'][5]
    # f2 = kw['clip_feat2'][5]
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    text = clip.tokenize(["a face with glasses"]).to('cpu')
    y0 = model.encode_text(text)
    text2 = clip.tokenize(["face"]).to('cpu')
    x0 = model.encode_text(text2)
    clip_new = []
    clip_feat = x1_batch = kw['clip_feat'].clone()
    kw['clip_feat_orig'] = clip_feat
    delta0 = y0-x0
    first_approx = True
    for i in range(8):
        alpha=1
        x1 = clip_feat[i]
        if first_approx:
            #print('compute_first_approx')
            x0_norm = torch.linalg.norm(x0)
            x1_norm = torch.linalg.norm(x1)
            beta = x1_norm/x0_norm
            alpha=beta
        # torch.linalg.norm(A)

        #print(x1.shape, x0.shape)
        fi = torch.nn.functional.cosine_similarity(y0[0].float(), x0[0].float(), dim=0, eps=1e-08)
        print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        fi = torch.nn.functional.cosine_similarity(clip_feat[i]+alpha*delta0[0], x1, dim=0, eps=1e-08)
        print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        clip_new.append(clip_feat[i]+alpha*delta0)
    kw['clip_feat'] = torch.stack(clip_new, 0)
    return kw

def add_delta_closest(kw):
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    text = clip.tokenize(["face with glasses"]).to('cpu')
    y0 = model.encode_text(text)[0].float()
    text2 = clip.tokenize(["face"]).to('cpu')
    x0 = model.encode_text(text2)[0].float()
    clip_new = []
    clip_feat = x1_batch = kw['clip_feat'].clone().float()
    delta0 = y0-x0
    for i in range(8):
        x1 = clip_feat[i]
        x0_norm = torch.linalg.norm(x0)
        x1_norm = torch.linalg.norm(x1)
        y0_norm = torch.linalg.norm(y0)
        delta0_norm = torch.linalg.norm(delta0)
        print(x1.shape, x0.shape, y0.shape, y0_norm, delta0.shape, delta0_norm)
        b = (y0_norm **2) * torch.dot(delta0, x1)/torch.dot(delta0, y0) - torch.dot(y0, x1)
        a = torch.dot(y0, delta0/delta0_norm)- (y0_norm **2) * delta0_norm/torch.dot(delta0, y0)
        alpha_gal = b/a
        alpha = alpha_gal/delta0_norm
        # torch.linalg.norm(A)


        fi = torch.nn.functional.cosine_similarity(x1, x0, dim=0, eps=1e-08)
        print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        clip_new.append(clip_feat[i]+alpha*delta0)
    kw['clip_feat'] = torch.stack(clip_new, 0)
    return kw

def add_delta_same_angle(kw):
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    text = clip.tokenize(["face with glasses"]).to('cpu')
    y0 = model.encode_text(text)[0].float()
    text2 = clip.tokenize(["face"]).to('cpu')
    x0 = model.encode_text(text2)[0].float()
    clip_new = []
    clip_feat = x1_batch = kw['clip_feat'].clone().float()
    delta0 = y0-x0
    for i in range(8):
        x1 = clip_feat[i]
        x0_norm = torch.linalg.norm(x0)
        x1_norm = torch.linalg.norm(x1)
        y0_norm = torch.linalg.norm(y0)
        delta0_norm = torch.linalg.norm(delta0)
        A = x1_norm**2
        Agal = x1_norm
        B = delta0_norm**2
        P = torch.dot(delta0, x1)
        #print('APB:', A,B,P)
        cosfi = torch.nn.CosineSimilarity(dim=0)(x0, y0)
        cosfi2 = cosfi**2
        a = A*B*cosfi2 -P**2
        b = 2*A*P*cosfi2-2*A*P
        c = Agal**4 *cosfi2 - Agal**4
        #print('abc',a, b, c)
        print(x1.shape, x0.shape, y0.shape, y0_norm, delta0.shape, delta0_norm)
        sq = b**2 - 4*a*c
        alpha = (-b + torch.sqrt(sq)) / 2*a
        # torch.linalg.norm(A)


        fi = torch.nn.functional.cosine_similarity(y0, x0, dim=0, eps=1e-08)
        print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        fi = torch.nn.functional.cosine_similarity(x1, clip_feat[i]+delta0, dim=0, eps=1e-08)
        print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        clip_new.append(clip_feat[i]+delta0)
    kw['clip_feat'] = torch.stack(clip_new, 0)
    return kw


def add_delta_imgs(kw):
    # img2 = kw['img2'][5]
    # f1 = kw['clip_feat'][5]
    # f2 = kw['clip_feat2'][5]
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    #text = clip.tokenize(["long hair"]).to('cpu')
    #y0 = model.encode_text(text)
    text2 = clip.tokenize(["a face"]).to('cpu')
    x0 = model.encode_text(text2).float()
    clip_new = []
    clip_feat = x1_batch = kw['clip_feat'].clone().float()


    first_approx = False
    opt_length = False
    for i in range(7):
        alpha=0.6
        x1 = clip_feat[i]
        if first_approx:
            print('compute_first_approx')
            x0_norm = torch.linalg.norm(x0)
            x1_norm = torch.linalg.norm(x1)
            beta = x1_norm/x0_norm
            alpha=beta
        if opt_length:
            y0 = clip_feat[i + 1]
            delta0 = y0 - x0[0]
            y0_norm = torch.linalg.norm(y0)
            delta0_norm = torch.linalg.norm(delta0)
            print(x1.shape, x0.shape, y0.shape, y0_norm, delta0.shape, delta0_norm)
            b = (y0_norm ** 2) * torch.dot(delta0, x1) / torch.dot(delta0, y0) - torch.dot(y0, x1)
            a = torch.dot(y0, delta0 / delta0_norm) - (y0_norm ** 2) * delta0_norm / torch.dot(delta0, y0)
            alpha_gal = b / a
            alpha = alpha_gal / delta0_norm

        # torch.linalg.norm(A)
        y0 = clip_feat[i+1]
        delta0 = y0 - x0
        print(x1.shape, x0.shape)
        fi = torch.nn.functional.cosine_similarity(x1, x0[0], dim=0, eps=1e-08)
        print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        clip_new.append(x1+alpha*delta0)

    clip_new.append(clip_feat[7].unsqueeze(0))
    kw['clip_feat'] = torch.stack(clip_new, 0)
    return kw


def add_delta_imgimg(kw):
    # img2 = kw['img2'][5]
    # f1 = kw['clip_feat'][5]
    # f2 = kw['clip_feat2'][5]
    #model, preprocess = clip.load("ViT-B/32", device='cpu')
    #text = clip.tokenize(["long hair"]).to('cpu')
    #y0 = model.encode_text(text)
    #text2 = clip.tokenize(["a face"]).to('cpu')
    #x0 = model.encode_text(text2).float()
    clip_new = []
    clip_feat = x1_batch = kw['clip_feat'].clone().float()
    clip_new.append(clip_feat[0].unsqueeze(0))

    first_approx = False
    opt_length = False
    for i in range(1,8):
        alpha=0.5
        x1 = clip_feat[i]
        x0 = clip_feat[i].unsqueeze(0)
        if first_approx:
            print('compute_first_approx')
            x0_norm = torch.linalg.norm(x0)
            x1_norm = torch.linalg.norm(x1)
            beta = x1_norm/x0_norm
            alpha=beta
        if opt_length:
            y0 = clip_feat[i + 1]
            delta0 = y0 - x0[0]
            y0_norm = torch.linalg.norm(y0)
            delta0_norm = torch.linalg.norm(delta0)
            print(x1.shape, x0.shape, y0.shape, y0_norm, delta0.shape, delta0_norm)
            b = (y0_norm ** 2) * torch.dot(delta0, x1) / torch.dot(delta0, y0) - torch.dot(y0, x1)
            a = torch.dot(y0, delta0 / delta0_norm) - (y0_norm ** 2) * delta0_norm / torch.dot(delta0, y0)
            alpha_gal = b / a
            alpha = alpha_gal / delta0_norm

        # torch.linalg.norm(A)
        y0 = clip_feat[i-1]
        delta0 = y0 - x0
        print(x1.shape, x0.shape)
        fi = torch.nn.functional.cosine_similarity(x1, x0[0], dim=0, eps=1e-08)
        print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        clip_new.append(x1+alpha*delta0)

    #clip_new.append(clip_feat[7].unsqueeze(0))
    kw['clip_feat'] = torch.stack(clip_new, 0)
    return kw


def add_delta_aug(kw):
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    text = clip.tokenize(["a face with glasses"]).to('cpu')
    y0 = model.encode_text(text)
    text2 = clip.tokenize(["face"]).to('cpu')
    x0 = model.encode_text(text2)
    clip_new = []
    clip_feat = x1_batch = kw['clip_feat_orig'].clone().cpu()
    delta0 = y0-x0
    first_approx = True
    for i in range(8):
        alpha=1
        x1 = clip_feat[i]
        if first_approx:
            #print('compute_first_approx')
            x0_norm = torch.linalg.norm(x0)
            x1_norm = torch.linalg.norm(x1)
            beta = x1_norm/x0_norm
            alpha=beta + random.uniform(-0.1,0.1)
        # torch.linalg.norm(A)

        #print(x1.shape, x0.shape)
        #fi = torch.nn.functional.cosine_similarity(x1, x0[0], dim=0, eps=1e-08)
        #print(f'cos(fi) {i}: {fi}, angle: {torch.arccos(fi)}, alpha: {alpha}')
        clip_new.append(clip_feat[i]+alpha*delta0)
    kw['clip_feat'] = torch.stack(clip_new, 0).cuda()
    return kw
