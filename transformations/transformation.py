import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################ Modules to change the appearance of the input images ############################################################
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, v):
        return x


class CommonStyle(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.n_channels = n_channels

    def denormalize(self, x):
        ## input image is in the range -1, to 1
        #  return torch.clamp(0.5*x + 0.5, 0, 1)
        return 0.5 * x + 0.5

    def normalize(self, x):
        ## input image is in the range 0 to 1
        return 2 * x - 1

    def forward(self, x, v):
        raise NotImplementedError


#### blurring transform
class GaussianBlur(nn.Module):
    def __init__(self, nchannels=1, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padder = nn.ReflectionPad2d(kernel_size // 2)

        ## register base gaussian window as buffer
        self.register_buffer('base_gauss', self.get_gaussian_kernel2d(kernel_size).repeat(nchannels, 1, 1, 1))

    def gaussian_window(self, window_size):
        def gauss_fcn(x):
            return -(x - window_size // 2) ** 2 / 2.0

        gauss = torch.stack(
            [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
        return gauss

    def get_gaussian_kernel(self, ksize):
        window_1d = self.gaussian_window(ksize)
        return window_1d

    def get_gaussian_kernel2d(self, ksize):
        kernel_x = self.get_gaussian_kernel(ksize)
        kernel_y = self.get_gaussian_kernel(ksize)
        kernel_2d = torch.matmul(
            kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
        return kernel_2d

    def forward(self, x, v):
        gauss_kernel = self.base_gauss ** (1 / (v ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()

        x = self.padder(x)
        return F.conv2d(x, gauss_kernel)


class Brightness(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)

    def forward(self, x, v):
        x = self.denormalize(x)
        x = x + v
        x = self.normalize(x)

        return x


class LogSig(CommonStyle):
    def __init__(self):
        super().__init__()

    def forward(self, x, g):
        maximums = torch.max(torch.abs(x), keepdim=True)
        c1 = 1 / (torch.exp(0.5 * g) + 1)
        c2 = 1 / (torch.exp(-0.5 * g) + 1)
        sig_image_g = 1 / (torch.exp(g * (-1 * x / maximums / 2)) + 1)
        sig_image = (sig_image_g - c1) / (c2 - c1) * maximums * 2 - maximums
        log_image_tolog = 1 / ((x / (2 * maximums) + 0.5) * (c1 - c2) + c2) - 1 + 1e-7
        log_image = torch.log(log_image_tolog) * (2 * maximums / g)
        condition = (g > 0).type(torch.float32)
        mod_image = sig_image * condition + log_image * (1.0 - condition)
        return mod_image


class Contrast(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)

    def forward(self, x, v):
        x = self.denormalize(x)
        x = x * v
        x = self.normalize(x)

        return x


class Gamma(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)

    def forward(self, x, v):
        x = self.denormalize(x)

        x = x ** v

        x = self.normalize(x)

        return x


############################################################ Modules to change the spatial deformation of the input images ############################################################
class RandomSpatial(nn.Module):
    def __init__(self):
        super().__init__()

        ## buffers
        self.register_buffer('unit_affine', torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_custom_parameters()

    def register_custom_parameters(self, ):
        raise NotImplementedError

    def generate_affine(self, batch_size, v):
        raise NotImplementedError

    def forward(self, x, v):
        affine = self.generate_affine(x.size()[0], v)

        grid = F.affine_grid(affine, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, padding_mode='border', align_corners=True)

        return x

    @torch.no_grad()
    def test(self, x, v=None, affine=None):
        if affine is None:
            affine = self.generate_affine(x.size()[0], v)

        grid = F.affine_grid(affine, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, padding_mode='border', align_corners=True)

        return x, affine

    def get_homographic_mat(self, A):
        H = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
        H[..., -1, -1] += 1.0

        return H

    def invert_affine(self, affine):
        # affine shape should be batch x 2 x 3
        assert affine.dim() == 3
        assert affine.size()[1:] == torch.Size([2, 3])

        homo_affine = self.get_homographic_mat(affine)
        inv_homo_affine = torch.inverse(homo_affine)
        inv_affine = inv_homo_affine[:, :2, :3]

        return inv_affine



class RandomResizeCrop(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.register_buffer('scale_matrix_x', torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('scale_matrix_y', torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('translation_matrix_x',
                             torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('translation_matrix_y',
                             torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32).reshape(-1, 2, 3))

    def get_random(self, batch_size, device):
        return 2 * torch.rand(batch_size, 1, 1, device=device) - 1.0

    def generate_affine(self, batch_size, v):
        delta_x = 0.5 * v * self.get_random(batch_size, v.device)
        delta_y = 0.5 * v * self.get_random(batch_size, v.device)

        affine = (1 - torch.abs(v)) * self.scale_matrix_x + (1 - torch.abs(v)) * self.scale_matrix_y + \
                 delta_x * self.translation_matrix_x + \
                 delta_y * self.translation_matrix_y

        # affine = affine.repeat(batch_size, 1, 1)
        return affine

class RandomResizeCropV2(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.register_buffer('scale_matrix_x', torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('scale_matrix_y', torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('translation_matrix_x',
                             torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('translation_matrix_y',
                             torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32).reshape(-1, 2, 3))

    def get_random(self, batch_size, device):
        return 2 * torch.rand(batch_size, 1, 1, device=device) - 1.0

    def generate_affine(self, batch_size, v_list):
        # print(batch_size)
        # print(v_list)
        v1, v2 = v_list
        # print(v1)
        # v1 = torch.Tensor([v1]).to(self.translation_matrix_x.device)
        # v2 = torch.Tensor([v2]).to(self.translation_matrix_y.device)
        delta_x = 0.5 * v1 * self.get_random(batch_size, self.translation_matrix_x.device)
        delta_y = 0.5 * v2 * self.get_random(batch_size, self.translation_matrix_y.device)

        affine = (1 - abs(v1)) * self.scale_matrix_x + (1 - abs(v2)) * self.scale_matrix_y + \
                 delta_x * self.translation_matrix_x + \
                 delta_y * self.translation_matrix_y

        # affine = affine.repeat(batch_size, 1, 1)
        return affine

class RandomHorizontalFlip(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self, ):
        self.register_buffer('horizontal_flip',
                             torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))

    def generate_affine(self, batch_size, v):
        affine = self.unit_affine.repeat(batch_size, 1, 1)
        # randomly flip some of the images in the batch
        mask = (torch.rand(batch_size, device=self.unit_affine.device) > 0.5)
        affine[mask] = affine[mask] * self.horizontal_flip

        return affine


class RandomVerticalFlip(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self, ):
        self.register_buffer('vertical_flip', torch.tensor([1, 0, 0, 0, -1, 0], dtype=torch.float32).reshape(-1, 2, 3))

    def generate_affine(self, batch_size, v):
        affine = self.unit_affine.repeat(batch_size, 1, 1)
        # randomly flip some of the images in the batch
        mask = (torch.rand(batch_size, device=self.unit_affine.device) > 0.5)
        affine[mask] = affine[mask] * self.vertical_flip

        return affine


class RandomRotate(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.register_buffer('rotation', torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))

    def generate_affine(self, batch_size, v):
        affine = self.unit_affine.repeat(batch_size, 1, 1)
        mask = (torch.rand(batch_size, device=self.unit_affine.device) > 0.5)
        affine[mask] = self.rotation.repeat(mask.sum(), 1, 1)

        return affine

