import functools
import torch
import torch.nn as nn
import os
import hashlib
from collections import namedtuple
import requests
from tqdm import tqdm

from torchvision.utils import make_grid
from torchvision import models
import torch.nn.functional as F

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class Labelator(AbstractEncoder):
    """Net2Net Interface for Class-Conditional Model"""
    def __init__(self, n_classes, quantize_interface=True):
        super().__init__()
        self.n_classes = n_classes
        self.quantize_interface = quantize_interface

    def encode(self, c):
        c = c[:,None]
        if self.quantize_interface:
            return c, None, [None, None, c.long()]
        return c


class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=2, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, 
                 nll_mode="sum"):

        super().__init__()
        assert nll_mode in ["sum", "avg"]

        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.nll_mode = nll_mode
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        print(f"{kl_weight = }")
        print(f"{nll_mode = }")

    
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # l1 loss for first channel
        l1_loss = torch.abs(inputs[:, 0, ...].contiguous() - reconstructions[:, 0, ...].contiguous())

        # sparse xent for channel 1 to 3
        xent_loss = F.cross_entropy(reconstructions[:, 1:, ...].contiguous(),
                                    inputs[:, 1, ...].contiguous().long())
        
        # ------------------ reconstruction LOSS ------------------
        rec_loss = l1_loss + xent_loss

        if self.perceptual_weight > 0:
            # --- prep reconstruction for perceptual loss ---
            # get first channel of input as img
            rec_img = reconstructions[:, 0, ...].contiguous()
            # collapse reconstructions channel 1 to 3 to 1 channel
            rec_label = torch.argmax(reconstructions[:, 1:, ...].contiguous(), dim=1)
            # label need to be normalized
            rec_label = rec_label.float() / 2.0
            # stack them together
            rec_pair = torch.stack([rec_img, rec_label], dim=1)

            # --- prep input for perceptual loss ---
            # get first channel of input as img
            input_img = inputs[:, 0, ...].contiguous()
            # get second channel of input as label
            input_label = inputs[:, 1, ...].contiguous()
            # normalize
            input_label = input_label.float() / 2.0
            # stack them together
            input_pair = torch.stack([input_img, input_label], dim=1)

            p_loss = self.perceptual_loss(input_pair.contiguous(), rec_pair.contiguous())

            # --- get total loss ---
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            # no perceptual loss
            p_loss = torch.zeros_like(rec_loss)
            
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
      
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss

        if self.nll_mode == "sum":
            # original loss function
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0] 
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0] 
        else:
            # modified loss function for readability
            weighted_nll_loss = torch.mean(weighted_nll_loss)
            nll_loss = torch.mean(nll_loss)
        
        # ------------------ KL LOSS ------------------
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        weighted_kl_loss = self.kl_weight * kl_loss

        # now the GAN part
        if optimizer_idx == 0:
            loss = weighted_nll_loss + weighted_kl_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/kl_loss".format(split): kl_loss.detach().mean(), 
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/l1_loss".format(split): l1_loss.detach().mean(),
                    "{}/xent_loss".format(split): xent_loss.detach().mean(),
                    "{}/w_nll_loss".format(split): weighted_nll_loss.detach().mean(),
                    "{}/w_kl_loss".format(split): weighted_kl_loss.detach().mean(),
                    "{}/percept_loss".format(split): (self.perceptual_weight * p_loss).detach().mean(),
                    
                   }
            return loss, log

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def im3d_to_grid(self, img, axis):
        # print("before grid", img.size())
        if axis == 1:
            nrow = 16
            img = img.permute(1,0,2,3)
            img = make_grid(img, padding=0, nrow=nrow)
            # img, label, img
            img = torch.cat((img, img[:1]), dim=0)

            return img
        elif axis == 2:
            nrow = 16
            
            img = img.permute(2,0,1,3)
            img = make_grid(img, padding=0, nrow=nrow)
            # print(f"{img2.size() = }")

            # img, label, img
            img = torch.cat((img, img[:1]), dim=0) 
            
            return img
        elif axis==3:
            # top view
            nrow = 12
            img = img.permute(3,0,1,2)
            img = make_grid(img, padding=0, nrow=nrow)
            
            img = torch.cat((img, img[:1]), dim=0)
            
            return img
        elif axis==12:
            nrow = 16
            img1 = img.permute(1,0,2,3)
            img1 = make_grid(img1, padding=0, nrow=nrow)
            
            img2 = img.permute(2,0,1,3)
            img2 = make_grid(img2, padding=0, nrow=nrow)
            
            img = torch.cat((img1, img2[:1]), dim=0)
            return img
        
    def forward_per_slice(self, input, target):
        val_total = 0
        imshape = input.size()
        x,y,z = imshape[2], imshape[3], imshape[4]
        # print(x,y,z, "x,y,z")

        val_slice_x = 0
        for i in range(x):
            # print("slice x", i)
            input_slice = input[:,:,i,:,:]
            target_slice = target[:,:,i,:,:]
            input_slice = torch.cat((input_slice, input_slice[:,:1]), dim=1)
            target_slice = torch.cat((target_slice, target_slice[:,:1]), dim=1)

            val = self.calculate_perceptual_loss(input_slice, target_slice)            
            # add them up
            val_slice_x += val

        val_slice_y = 0
        for j in range(y):
            # print("sliceY", j)
            input_slice = input[:,:,:,j,:]
            target_slice = target[:,:,:,j,:]
            input_slice = torch.cat((input_slice, input_slice[:,:1]), dim=1)
            target_slice = torch.cat((target_slice, target_slice[:,:1]), dim=1)

            val = self.calculate_perceptual_loss(input_slice, target_slice)            
            # add them up
            val_slice_y += val
        
        val_slice_z = 0
        for k in range(z):
            # print("slice z", k)
            input_slice = input[:,:,:,:,k]
            target_slice = target[:,:,:,:,k]
            input_slice = torch.cat((input_slice, input_slice[:,:1]), dim=1)
            target_slice = torch.cat((target_slice, target_slice[:,:1]), dim=1)

            val = self.calculate_perceptual_loss(input_slice, target_slice)            
            # add them up
            val_slice_z += val

        val_total = val_slice_x/x + val_slice_y/y + val_slice_z/z
        val_total = val_total / 3
        return val_total

    
    
    def forward(self, input, target):
        # we gonna make a grid first from 3d to 2d
        # B, C, H, W, D
        val_total = 0
        for axis in [1,2,3]:

            input_grid  = [self.im3d_to_grid(item, axis=axis) for item in input ]
            target_grid = [self.im3d_to_grid(item, axis=axis) for item in target]
            
            input_grid = torch.stack(input_grid, dim=0)
            target_grid = torch.stack(target_grid, dim=0)
            val = self.calculate_perceptual_loss(input_grid, target_grid)
            
            # add them up
            val_total += val
        val_total = val_total / 3.
        return val_total

    def calculate_perceptual_loss(self, input, target):
        # in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        in0_input, in1_input = input, target

        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
            # print(f"{diffs[kk].shape = }")

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            # print(f"{res[l].shape = }")
            val += res[l]
        return val



class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)



URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path
