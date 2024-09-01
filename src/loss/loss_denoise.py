# from pytorch_ssim import SSIM  # You may need to install this package
from torch import nn
import torch
import torch.nn.functional as F
from math import exp

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        window = self.create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0):
        super(MixedLoss, self).__init__()
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        ssim_out = 1 - self.ssim_loss(input, target)
        mse_out = self.mse_loss(input, target)
        tv_out = self.tv_loss(input)
        loss = self.alpha * ssim_out + (1 - self.alpha) * mse_out + self.beta * tv_out
        return loss


# class TVLoss(nn.Module):
#     def __init__(self, TVLoss_weight=1):
#         super(TVLoss, self).__init__()
#         self.TVLoss_weight = TVLoss_weight

#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self.atrix.size()[2] - 1
#         count_w = self.atrix.size()[3] - 1
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :count_h, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :count_w]), 2).sum()
#         return self.TVLoss_weight * 2 * (h_tv/count_h/w_x + w_tv/count_w/h_x) / batch_size
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = x.size()[2] - 1
        count_w = x.size()[3] - 1
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :count_h, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :count_w]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv/count_h/w_x + w_tv/count_w/h_x) / batch_size

    
if __name__ == '__main__':
    loss = MixedLoss()
    input = torch.randn(1, 1, 256, 256)
    target = torch.randn(1, 1, 256, 256)
    print(loss(input, target))
