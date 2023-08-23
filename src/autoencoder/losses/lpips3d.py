import functools
import torch
import torch.nn as nn
from collections import namedtuple
from losses import resnet

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

class LPIPS3DWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=2, disc_factor=0., disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        print("Using LPIPS3D")
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS3D().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        print(f"{kl_weight = }")

        # self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
        #                                          n_layers=disc_num_layers,
        #                                          use_actnorm=use_actnorm
        #                                          ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        print(f"{disc_factor = }")
        print(f"{disc_weight = }")
        print(f"{disc_conditional = }")

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            # print("Using last layer is not NONE")
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # print("Using last layer is NONE")
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        # print(f"calculating d_weight")
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight

        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            # print(f"{rec_loss.detach().mean() = }, {p_loss.detach().mean() = }")
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # print(nll_loss.size(), rec_loss.size(), self.logvar.size())
        # print(nll_loss, rec_loss, torch.exp(self.logvar))
        # print(self.logvar, torch.exp(self.logvar))

        weighted_nll_loss = nll_loss
        if weights is not None:
            # print(f"{weights = }")
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0] # ori
        # weighted_nll_loss = torch.mean(weighted_nll_loss) # modified
        # print(f"{weighted_nll_loss = }")
        
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0] # original
        # nll_loss = torch.mean(nll_loss) # modified
        
        # print(f"{nll_loss = }")
        

        kl_loss = posteriors.kl()
        # print(kl_loss.size(), kl_loss.shape[0])
        # print(f"{kl_loss = } {kl_loss.size()}")
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        
        # print(f"{kl_loss = } {kl_loss.size()}")
        weighted_kl_loss = self.kl_weight * kl_loss

        # now the GAN part
        if optimizer_idx == 0:
            # print(f"{self.disc_factor = }")
            # if self.disc_factor > 0.0:
            #     # generator update
            #     if cond is None:
            #         assert not self.disc_conditional
            #         logits_fake = self.discriminator(reconstructions.contiguous())
            #     else:
            #         assert self.disc_conditional
            #         logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            #     g_loss = -torch.mean(logits_fake)

            #     print(f"{g_loss = }")

            #     # because disc_factor > 0
            #     try:
            #         print("trying to calculate adaptive weight")
            #         d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            #     except RuntimeError:
            #         print("failed to calculate adaptive weight")
            #         assert not self.training
            #         d_weight = torch.tensor(0.0)

            #     disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # else:
            #     d_weight = torch.tensor(0.0)
            #     g_loss = torch.tensor(0.0) # don't bother
            #     disc_factor = torch.tensor(0.0)

            # print(d_weight)
            # print(disc_factor)
            # loss = weighted_nll_loss + weighted_kl_loss + d_weight * disc_factor * g_loss
            # print(f"{d_weight = } {disc_factor = }")
             
            loss = weighted_nll_loss + weighted_kl_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/kl_loss".format(split): kl_loss.detach().mean(), 
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/w_nll_loss".format(split): weighted_nll_loss.detach().mean(),
                    "{}/w_kl_loss".format(split): weighted_kl_loss.detach().mean(),
                    # "{}/d_weight".format(split): d_weight.detach(),
                    # "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    # "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/percept_loss".format(split): (self.perceptual_weight * p_loss).detach().mean(),
                    
                   }
            return loss, log

        # if optimizer_idx == 1:
        #     # second pass for discriminator update
        #     if cond is None:
        #         logits_real = self.discriminator(inputs.contiguous().detach())
        #         logits_fake = self.discriminator(reconstructions.contiguous().detach())
        #     else:
        #         logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
        #         logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

        #     disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        #     d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        #     log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
        #            "{}/logits_real".format(split): logits_real.detach().mean(),
        #            "{}/logits_fake".format(split): logits_fake.detach().mean()
        #            }
        #     return d_loss, log


class LPIPS3D(nn.Module):
    # Learned perceptual metric
    def __init__(self):
        super().__init__()
        # self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = resnetLpips(requires_grad=False)
    
    def forward(self, input, target):
        val_total = 0
        for l in range(input.shape[1]):
            
            input_img = input[:, l:l+1]
            target_img = target[:, l:l+1]
            val = self.calculate_perceptual_loss(input_img, target_img)
            
            # add them up
            val_total += val
            # print(f"{l =}{val = }")
        # val_total = val_total / input.shape[1]
        return val_total
    
    def calculate_perceptual_loss(self, input, target):
        in0_input, in1_input = input, target

        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        loss_val = 0
        for kk in range(len(outs0)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
            # Note: can put res in array so we can spit out all losses per layer
            res = spatial_average(diffs[kk], keepdim=True)
            loss_val += res

        return loss_val


class resnetLpips(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(resnetLpips, self).__init__()
        model_depth = 50
        resnet_shortcut = 'B'
        # resnet_10_23dataset.pth: --model resnet --model_depth 10 --resnet_shortcut B
        # resnet_18_23dataset.pth: --model resnet --model_depth 18 --resnet_shortcut A
        # resnet_34_23dataset.pth: --model resnet --model_depth 34 --resnet_shortcut A
        # resnet_50_23dataset.pth: --model resnet --model_depth 50 --resnet_shortcut B
        network = resnet.get_resnet_model(model_depth, resnet_shortcut)

        CKPT_PATH = f"resnet/pretrain/resnet_{model_depth}_23dataset.pth"
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device("cpu"))
        print("loaded from ", CKPT_PATH)
        print(checkpoint.keys())
        network.load_state_dict(checkpoint['state_dict'], strict=False)
        # load weights of resnet here
        
        # network.load_state_dict(checkpoint['state_dict'], strict=False)

        layers = [network.conv1, network.bn1, network.relu, network.maxpool, network.layer1]
        self.slice1 = nn.Sequential(*layers)
        self.slice2 = network.layer2
        self.slice3 = network.layer3
        self.slice4 = network.layer4
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h1 = self.slice1(X)
        # h_relu1_2 = h
        h2 = self.slice2(h1)
        # h_relu2_2 = h
        h3 = self.slice3(h2)
        # h_relu3_3 = h
        h4 = self.slice4(h3)
        # h_relu4_3 = h
        resnet_outputs = namedtuple("ResnetOutputs", ['h1', 'h2', 'h3', 'h4'])
        out = resnet_outputs(h1, h2, h3, h4)
        
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    # y = x.mean([2,3,4],keepdim=keepdim)
    # return y.sum([1],keepdim=keepdim)
    # we perform the avg directly on channel dim also
    return x.mean([1,2,3,4],keepdim=keepdim)

