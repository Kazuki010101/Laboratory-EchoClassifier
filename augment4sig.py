import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms

class Signal2Tensor(object):
    def __call__(self, signal):
        # Convert the signal to a PyTorch tensor
        tensor = torch.from_numpy(signal.copy()).float()
        
        return tensor
    
class Signal4DeepConvLSTM(object):
    def __call__(self, signal):
        # Convert the signal to a PyTorch tensor
        tensor = signal.permute(1,0)
        
        return tensor

class Resample(object):
    def __init__(self, num_sample=500):
        self.num_sample = num_sample
    def __call__(self, x):
        from scipy import signal
        y = np.zeros((x.shape[0], self.num_sample))
        for i in range(x.shape[0]):
            y[i] = signal.resample(x[i], self.num_sample)
        return y

class jitter(object):
    def __call__(self, x, sigma=0.03):
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

class flip(object):
    def __call__(self, x):
        if len(x.shape) == 2:
            x = x[np.newaxis,:,:]
        flip = np.flip(x, axis=2)[0]
        return flip
    

class RandomCrop1D(nn.Module):
    def __init__(self, size, padding=0, padding_mode='constant'):
        super().__init__()
        self.size = size
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, signal):
        if self.padding > 0:
            signal = nn.functional.pad(signal, (self.padding, self.padding), mode=self.padding_mode)
        
        signal_length = signal.size(-1)
        start = torch.randint(0, signal_length - self.size + 1, (1,)).item()
        return signal[..., start:start+self.size]

class RandomResizedCrop1D(nn.Module):
    def __init__(self, size, scale=(0.08, 1.0)):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, signal):
        signal_length = self.size
        target_length = int(signal_length * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        start = torch.randint(0, signal_length - target_length + 1, (1,)).item()
        cropped = signal[..., start:start+target_length]
        return nn.functional.interpolate(cropped.unsqueeze(0), size=self.size, mode='linear', align_corners=False).squeeze(0)

class RandomHorizontalFlip1D(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, signal):
        if torch.rand(1).item() < self.p:
            return torch.flip(signal, [-1])
        return signal

class GaussianNoise1D(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, signal):
        return signal + torch.randn_like(signal) * self.std

def new_data_aug_generator(args=None):
    signal_length = args.input_size
    remove_random_resized_crop = args.src
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    primary_tfl = []
    if remove_random_resized_crop:
        primary_tfl = [
            RandomCrop1D(signal_length, padding=4),
            RandomHorizontalFlip1D()
        ]
    else:
        primary_tfl = [
            RandomResizedCrop1D(signal_length, scale=(0.08, 1.0)),
            RandomHorizontalFlip1D()
        ]

    secondary_tfl = [
        transforms.RandomChoice([
            GaussianNoise1D(std=0.1),
            transforms.Lambda(lambda x: torch.abs(x)),  # Rectification (similar to Solarization)
            transforms.Lambda(lambda x: x.mean(dim=0, keepdim=True).repeat(3, 1))  # Grayscale equivalent
        ])
    ]

    if args.color_jitter is not None and args.color_jitter != 0:
        secondary_tfl.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * args.color_jitter))

    final_tfl = [
        transforms.Lambda(lambda x: (x - torch.tensor(mean).view(3, 1)) / torch.tensor(std).view(3, 1))
    ]

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

class Mixup1D:
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam
        self.mixup_enabled = True

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (xl, xh), lam = cutmix_bbox_and_lam_1d(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, xl:xh] = x_orig[j][:, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (xl, xh), lam = cutmix_bbox_and_lam_1d(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, xl:xh] = x_orig[j][:, xl:xh]
                    x[j][:, xl:xh] = x_orig[i][:, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (xl, xh), lam = cutmix_bbox_and_lam_1d(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, xl:xh] = x.flip(0)[:, :, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target

def cutmix_bbox_and_lam_1d(signal_shape, lam, ratio_minmax=None, correct_lam=True):
    _, _, L = signal_shape  # Unpack the 3D shape
    cut_rat = np.sqrt(1. - lam) if ratio_minmax is None else np.random.uniform(ratio_minmax[0], ratio_minmax[1])
    cut_w = int(L * cut_rat)

    # uniform
    cx = np.random.randint(L)
    
    xl = np.clip(cx - cut_w // 2, 0, L)
    xh = np.clip(cx + cut_w // 2, 0, L)

    if correct_lam:
        lam = 1. - (xh - xl) / L
    return (xl, xh), lam

def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)

def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)