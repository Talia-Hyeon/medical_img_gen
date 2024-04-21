import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode="nearest"):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        # Create sampling grid

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        self.mode = mode

    def get_mode(self, x):
        mode_list = ["nearest", "nearest", "linear", "bilinear", "bicubic", "trilinear"]
        return mode_list[len(x.size())]

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        flow = flow.type_as(src)
        flow = F.interpolate(flow, src.size()[2:], mode=self.get_mode(src))
        new_locs = self.grid.to(src.device) + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if torch.__version__ >= "1.3.0":
            return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)
        else:
            return F.grid_sample(src, new_locs, mode=self.mode)


class SpatialTransformer2(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode="bilinear"):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super().__init__()

        # Create sampling grid

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        self.mode = mode

    def get_mode(self, x):
        mode_list = ["nearest", "nearest", "linear", "bilinear", "bicubic", "trilinear"]
        return mode_list[len(x.size())]

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        flow = flow.type_as(src)
        flow = F.interpolate(flow, src.size()[2:], mode=self.get_mode(src))
        new_locs = self.grid.to(src.device) + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if torch.__version__ >= "1.3.0":
            return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)
        else:
            return F.grid_sample(src, new_locs, mode=self.mode)


class Random_Flip(nn.Module):
    def __init__(self, flip_axis=[], _prob=0.5) -> None:
        super().__init__()

        self.prob = _prob
        self.flip_axis = flip_axis
        if len(self.flip_axis) == 0:
            self.flip_axis = [1]

        self.last_applied = flip_axis
        self.reuse = False

    def __repr__(self):
        return f"\t Random Flip: p {self.prob} | axis: {self.flip_axis}"

    def forward(self, x):
        with torch.no_grad():
            length = len(x.size()[2:])

            if len(self.flip_axis) < length:
                self.flip_axis += [1] * (length - len(self.flip_axis))

            tmp = []
            for i in range(length):
                if self.reuse:
                    if self.last_applied[i] == 1:
                        x = torch.flip(x, [i + 2])
                else:
                    if self.flip_axis[i] == 1:
                        if torch.rand(1) < self.prob:
                            x = torch.flip(x, [i + 2])
                            tmp.append(1)
                        else:
                            tmp.append(0)
                    else:
                        tmp.append(0)
            if self.reuse:
                pass  # print(is_valid)
            else:
                self.last_applied = tmp
        return x


class Random_AF(nn.Module):
    def __init__(self, img_size, theta=0, tr=0, _prob=0.5) -> None:
        super().__init__()

        self.prob = _prob
        self.img_size = img_size
        self.dim = len(img_size)

        self.theta = theta
        self.tr = tr

        self.last_applied = None
        self.reuse = False

    def __repr__(self):
        return f"\t Random_AF: p {self.prob} | size: {self.img_size} | tr {self.tr}, theta {self.theta}"

    def forward(self, x):
        with torch.no_grad():
            sh = x.size()
            n, c = sh[:2]

            if self.reuse:
                matrix = self.last_applied[-n:].type_as(x).to(x.device)
                grids = F.affine_grid(matrix, sh, align_corners=False).to(x.device)
                x = F.grid_sample(x, grids, padding_mode="zeros", align_corners=False)

            elif torch.rand(1) < self.prob:
                matrix = random_af_matrix(sh, self.theta, self.tr, self.dim)
                grids = F.affine_grid(matrix, sh, align_corners=False).to(x.device)
                self.last_applied = matrix

                x = F.grid_sample(x, grids, padding_mode="zeros", align_corners=False)
            else:
                self.last_applied = torch.eye(self.dim + 1)[:-1, :]
                self.last_applied = torch.stack([self.last_applied] * n, dim=0)
        return x


class Random_EL(nn.Module):
    def __init__(self, img_size, grid_size=0, mag=0, _prob=0.5) -> None:
        super().__init__()

        self.prob = _prob
        self.img_size = img_size
        self.dim = len(img_size)

        self.grid_size = grid_size
        self.grid_size = [grid_size] * self.dim
        self.mag = mag

        self.last_applied = None
        self.reuse = False

        self.stn = SpatialTransformer(img_size)

    def __repr__(self):
        return f"\t Random_EL: p {self.prob} | size: {self.img_size} | grid {self.grid_size}, mag {self.mag}"

    def forward(self, x):
        with torch.no_grad():
            sh = x.size()
            n = sh[0]

            if self.reuse:
                flow = self.last_applied[-n:].type_as(x).to(x.device)
                x = self.stn(x, flow)

            elif torch.rand(1) < self.prob:
                flow = 2 * (torch.rand([n, self.dim, *self.grid_size]) - 0.5) * self.mag
                self.last_applied = flow
                flow = flow.type_as(x).to(x.device)

                x = self.stn(x, flow)
            else:
                self.last_applied = torch.zeros([n, self.dim, *self.grid_size])

        return x


class Random_Ny(nn.Module):
    def __init__(self, img_size, rate=0, _prob=0.5) -> None:
        super().__init__()

        self.prob = _prob
        self.img_size = img_size
        self.dim = len(img_size)

        self.rate = 0.2 if rate is None else rate

        self.last_applied = None
        # self.reuse=False

    def __repr__(self):
        return f"\t Random_Ny: p {self.prob} | size: {self.img_size} | rate {self.rate}"

    def forward(self, x):
        with torch.no_grad():
            sh = x.size()
            n = sh[0]

            # if self.reuse:
            #     ny = self.last_applied[-n:].type_as(x).to(x.device)
            #     x = x * (1-self.rate) + ny * self.rate

            if torch.rand(1) < self.prob:
                ny = torch.rand_like(x)
                self.last_applied = ny
                ny = ny.type_as(x).to(x.device)

                x = x * (1 - self.rate) + ny * self.rate
            else:
                self.last_applied = torch.zeros_like(x)

        return x


class Random_BC(nn.Module):
    def __init__(self, img_size, brightness, contrast_alpha, _prob=0.5) -> None:
        super().__init__()

        self.prob = _prob
        self.img_size = img_size
        self.dim = len(img_size)

        self.contrast_alpha = contrast_alpha
        self.brightness = brightness

        self.last_applied = None
        self.reuse = False

    def __repr__(self):
        return f"\t Random_BC: p {self.prob} | size: {self.img_size} | contrast_alpha {self.contrast_alpha}, brightness {self.brightness}"

    def forward(self, x):
        with torch.no_grad():
            sh = x.size()
            n = sh[0]

            if self.reuse:
                return x
            #     ny = self.last_applied[-n:].type_as(x).to(x.device)
            #     x = x * (1-self.rate) + ny * self.rate

            if torch.rand(1) < self.prob:
                _mean = torch.zeros(1).type_as(x)
                _std = torch.ones(1).type_as(x)
                # tmp_alpha = torch.clamp(
                #     1.5 * torch.normal(_mean) + 1,
                #     self.contrast_alpha[0],
                #     self.contrast_alpha[1],
                # )
                # tmp_b = torch.clamp(
                #     1.5 * torch.normal(_mean), -self.brightness / 2, self.brightness
                # )
                tmp_alpha = torch.pow(torch.tensor(2), self.contrast_alpha[0] + torch.rand(x.size(0)) * (
                            self.contrast_alpha[1] - self.contrast_alpha[0]))
                tmp_b = 2 * (torch.rand(x.size(0)) - 0.5) * self.brightness
                x = tmp_alpha.reshape(n, *[1 for _ in range(len(sh) - 1)]).type_as(x) * x + tmp_b.reshape(n,
                                                                                                          *[1 for _ in
                                                                                                            range(
                                                                                                                len(sh) - 1)]).type_as(
                    x)
                x = torch.clamp(x, 0, None)
                # x = torch.pow(x, tmp_alpha)

                self.last_applied = 0
            else:
                self.last_applied = torch.zeros_like(x)

        return x


class Random_ResizeCrop(nn.Module):
    def __init__(self, crop_size, resize_range=[0.5, 2.0], prob=0.5) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.resize_range = resize_range
        self.prob = prob

        self.last_applied = None
        self.reuse = False

    def __repr__(self):
        return f"\t Random_ResizeCrop: p {self.prob} | crop_size: {self.crop_size} | resize_range {self.resize_range}"

    def forward(self, x, y=None):
        _scale = self.resize_range[0] + np.random.rand() * (self.resize_range[1] - self.resize_range[0])

        # _x = F.interpolate(x, scale_factor=_scale)
        _img_size = x.size()[2:]
        # crop_pos = [np.random.rand() * (_img_size[i] - self.crop_size[i]) for i in range(len(_img_size))]
        crop_size = [1] * len(_img_size)

        sh = x.size()
        n, c = sh[:2]

        if self.reuse:
            matrix = self.last_applied[-n:].type_as(x).to(x.device)
            grids = F.affine_grid(matrix, sh, align_corners=False).to(x.device)
            x = F.grid_sample(x, grids, padding_mode="zeros", align_corners=False)

        elif torch.rand(1) < self.prob:
            matrix = random_sym_matrix(sh, self.resize_range[1], 1, len(_img_size))
            grids = F.affine_grid(matrix, sh, align_corners=False).to(x.device)
            self.last_applied = matrix

            x = F.grid_sample(x, grids, padding_mode="zeros", align_corners=False)
        else:
            self.last_applied = torch.eye(len(_img_size) + 1)[:-1, :]
            self.last_applied = torch.stack([self.last_applied] * n, dim=0)
        return x


class Compose:
    def __init__(
            self,
            img_size,
            types="FAEBNR",
            prob=0.5,
            device="cpu",
            rot=20,
            tr=10,
            grid_size=8,
            crop_size=[100, 100],
            resize_range=[0.5, 2.0],
            mag=5,
            brightness=0.4,
            contrast_alpha=[0.5, 1.5],
            noise_rate=0.2,
    ) -> None:
        # super().__init__()

        self.ops = []
        for _code in types:
            if _code == "F":
                AF = Random_Flip([0, 1]).to(device)
            if _code == "A":
                AF = Random_AF(img_size, theta=rot, tr=tr).to(device)
            if _code == "R":
                AF = Random_ResizeCrop(crop_size, resize_range).to(device)
            if _code == "E":
                AF = Random_EL(img_size, grid_size=grid_size, mag=mag).to(device)
            if _code == "B":
                AF = Random_BC(
                    img_size, brightness=brightness, contrast_alpha=contrast_alpha
                ).to(device)
            if _code == "N":
                AF = Random_Ny(img_size, rate=noise_rate).to(device)

            self.ops.append(AF)

    def set_resue(self, reuse):
        for i in range(len(self.ops)):
            self.ops[i].reuse = reuse

    def do_ops(self, x):
        for i in range(len(self.ops)):
            x = self.ops[i](x)
        return x

    def __call__(self, *args, is_reuse=False):
        results = [0] * len(args)
        self.set_resue(is_reuse)

        results[0] = self.do_ops(args[0])
        self.set_resue(True)
        for i in range(1, len(args)):
            results[i] = self.do_ops(args[i])
        self.set_resue(False)
        if len(results) == 1:
            results = results[0]

        return results

    def __repr__(self) -> str:
        text = ""
        for op in self.ops:
            # print(op)
            text += f"{op}\n"
        return text


def random_af_matrix(sh, theta, tr, dim):
    N = sh[0]

    if dim == 2:
        matrix = torch.zeros([N, 2, 3])
        t1 = 2 * (torch.rand(N) - 0.5) * theta
        t1 = torch.deg2rad(t1)
        trs = 2 * (torch.rand(N, 2) - 0.5) * tr
        c = torch.cos(t1)
        s = torch.sin(t1)

        matrix[:, 0, 0] = c
        matrix[:, 0, 1] = -s
        matrix[:, 0, 2] = trs[:, 0] / sh[2]  # + sh[2]//2
        matrix[:, 1, 0] = s
        matrix[:, 1, 1] = c
        matrix[:, 1, 2] = trs[:, 1] / sh[3]  # - sh[3]//2

    elif dim == 3:
        matrix = torch.zeros([N, 3, 4])
        t1 = 2 * (torch.rand(N, 3) - 0.5) * theta
        t1 = torch.deg2rad(t1)
        trs = 2 * (torch.rand(N, 3) - 0.5) * tr
        c = torch.cos(t1)
        s = torch.sin(t1)

        matrix[:, 0, 0] = c[:, 0] * c[:, 1]
        matrix[:, 0, 1] = c[:, 0] * s[:, 1] * s[:, 2] - s[:, 0] * s[:, 2]
        matrix[:, 0, 2] = c[:, 0] * s[:, 1] * c[:, 2] + s[:, 0] * s[:, 2]
        matrix[:, 0, 3] = trs[:, 0] / sh[2]  # + sh[2]//2

        matrix[:, 1, 0] = s[:, 0] * c[:, 1]
        matrix[:, 1, 1] = s[:, 0] * s[:, 1] * s[:, 2] + c[:, 0] * c[:, 2]
        matrix[:, 1, 2] = s[:, 0] * s[:, 1] * c[:, 2] - c[:, 0] * s[:, 2]
        matrix[:, 1, 3] = trs[:, 1] / sh[3]  # - sh[3]//2

        matrix[:, 2, 0] = -s[:, 1]
        matrix[:, 2, 1] = s[:, 0] * c[:, 1]
        matrix[:, 2, 2] = c[:, 0] * c[:, 1]
        matrix[:, 2, 3] = trs[:, 2] / sh[4]  # + sh[4]//2

    return matrix


def random_sym_matrix(sh, scale, tr, dim):
    N = sh[0]

    if dim == 2:
        matrix = torch.zeros([N, 2, 3])
        trs = 2 * (torch.rand(N, 2) - 0.5) * tr
        scales = torch.zeros([N]) + scale
        scales = torch.pow(scales, 2 * (torch.rand(N) - 0.5))

        matrix[:, 0, 0] = scales
        matrix[:, 0, 1] = 0
        matrix[:, 0, 2] = trs[:, 0]  # / sh[2]  # + sh[2]//2
        matrix[:, 1, 0] = 0
        matrix[:, 1, 1] = scales
        matrix[:, 1, 2] = trs[:, 1]  # / sh[3]  # - sh[3]//2

    elif dim == 3:
        matrix = torch.zeros([N, 3, 4])
        trs = 2 * (torch.rand(N, 3) - 0.5) * tr
        scales = torch.zeros([N]) + scale
        scales = torch.pow(scales, 2 * (torch.rand(N) - 0.5))

        matrix[:, 0, 0] = scales
        matrix[:, 0, 3] = trs[:, 0]  # / sh[2]  # + sh[2]//2

        matrix[:, 1, 1] = scales
        matrix[:, 1, 3] = trs[:, 1]  # / sh[3]  # - sh[3]//2

        matrix[:, 2, 2] = scales
        matrix[:, 2, 3] = trs[:, 2]  # / sh[4]  # + sh[4]//2

    return matrix


def random_grid(sh, grid_size, mag, dim):
    grid = 2 * (torch.rand([sh[0], *grid_size]) - 0.5) * mag


class Affine_TF(nn.Module):
    """
    Affine transform using torch API(torch 1.3.1 or higher)\n
    Based on 'affine_grid' and 'grid_sample'\n
    -- args(default)\n
        - img_size:     (required)img_size for grid_sample\n
        - flip_lr:      (True)enable/disable random flip left/right\n
        - flip_ud:      (True)enable/disable random flip up/down\n
        - theta:        (0)set random degree range for rotation\n
        - tr_x:         (0)set random pixel range for translation on x-axis\n
        - tr_y:         (0)set random pixel range for translation on y-axis\n
    --return\n
        - Affine_transformation class as torch.nn.Module\n
    -- example\n
        $ x = torch.zeros([1,1,400,400]) # shape: NCHW\n
        $ TF = Affine_TF([400,400], True, True, 50, 0, 0)\n
        $ new_x = TF(x)\n
    """

    def __init__(self, img_size, flip_lr=True, flip_ud=True, theta=0, tr_x=0, tr_y=0):
        super().__init__()
        self.theta = theta * 2
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.tr_x = tr_x * 2
        self.tr_y = tr_y * 2

    def m_affine(self, N):
        m = []
        for i in range(N):
            i = torch.eye(3)
            if self.theta != 0:
                if np.random.rand() > 0.5:
                    t = np.pi / 180.0 * ((np.random.rand() - 0.5) * self.theta)

                    i[:2, :2] = torch.tensor(
                        [[np.cos(t), -1.0 * np.sin(t)], [np.sin(t), np.cos(t)]]
                    )
            if self.tr_x != 0 and self.tr_y != 0:
                if np.random.rand() > 0.5:
                    tr_x1 = (torch.rand(1) - 0.5) * self.tr_x
                    tr_y1 = (torch.rand(1) - 0.5) * self.tr_y
                    i += torch.tensor([0, 0, tr_x1, 0, 0, tr_y1, 0, 0, 0]).view(3, 3)
            m.append(i[:2, :])
        m = torch.stack(m, 0)
        return m

    def forward(self, x, with_flow=False):
        with torch.no_grad():
            N, C, H, W = x.size()
            affine_matrix = self.m_affine(N).type_as(x).to(x.device)

            if self.flip_lr:
                for nn in range(N):
                    if torch.rand(1) > 0.5:
                        x[nn] = torch.flip(x[nn], [2])
            if self.flip_ud:
                for nn in range(N):
                    if torch.rand(1) > 0.5:
                        x[nn] = torch.flip(x[nn], [1])

            if torch.__version__ >= "1.3.0":
                grids = F.affine_grid(
                    affine_matrix, [N, C, H, W], align_corners=False
                ).to(x.device)
                x = F.grid_sample(x, grids, padding_mode="zeros", align_corners=False)
            else:
                grids = F.affine_grid(affine_matrix, [N, C, H, W]).to(x.device)
                x = F.grid_sample(x, grids, padding_mode="zeros")
        if with_flow:
            return x, grids
        else:
            return x


class Elastic_TF(nn.Module):
    """
    Affine transform using torch API(torch 1.3.1 or higher)\n
    Based on 'affine_grid' and 'grid_sample'\n
    -- args(default)\n
        - img_size:     (required)img_size for grid_sample\n
        - grid_size:    ([6]*len(img.size)])grid_size for random deformation field\n
        - flip_ax:      (None)enable/disable random flip axis input in depth,height,width(zyx) order\n
        - theta:        ([0]*len(img.size))set random range to deform in depth,height,width(zyx) order\n
    --return\n
        - Elastic_TF class as torch.nn.Module\n
    -- example\n
        $ x = torch.zeros([1,1,400,400]) # shape: NCHW\n
        $ size_x = x.size()[2:]\n
        $ TF = Elastic_TF(size_x, [3]*len(size_x), [1]*len(size_x), [10]*len(size_x))\n
        $ new_x = TF(x)\n
    """

    def __init__(
            self,
            img_size,
            grid_size=6,
            theta=5,
            flip_ax=None,
    ):
        super().__init__()
        self.img_size = img_size

        self.grid_size = grid_size
        if type(self.grid_size) == int:
            self.grid_size = [self.grid_size] * len(self.img_size)

        self.theta = theta
        if type(self.theta) == int or type(self.theta) == float:
            self.theta = [self.theta] * len(self.img_size)

        self.flip_ax = flip_ax

        self.STN = SpatialTransformer(img_size)

        print("Elastic Tf setup")
        print("img_size\t: ", self.img_size)
        print("grid_size\t: ", self.grid_size)
        print("theta\t: ", self.theta)
        print("flip_ax\t: ", self.flip_ax)

    def get_random_grid(self):
        zyx = []

        for i in range(len(self.theta)):
            tmp = (torch.rand(self.grid_size) - 0.5) * 2 * self.theta[i]
            zyx.append(tmp)

        zyx = torch.stack(zyx, dim=0)
        return zyx

    def forward(self, x, with_flow=True):
        with torch.no_grad():
            size_x = x.size()
            N = size_x[0]
            C = size_x[1]
            sh = size_x[2:]

            random_grid = []
            for i in range(N):
                _zyx = self.get_random_grid()
                random_grid.append(_zyx)
                if self.flip_ax != None:
                    for f in self.flip_ax:
                        if np.random.rand() > 0.5:
                            x[i] = torch.flip(x[i], dims=[f + 1])
            aug_flow = torch.stack(random_grid, dim=0)
            x = self.STN(x, aug_flow)
        if with_flow:
            return x, aug_flow
        else:
            return x


class Elastic_A2B2C(nn.Module):
    def __init__(self, img_size, grid_size=6, mag=5):
        super().__init__()
        self.img_size = img_size

        self.grid_size = grid_size
        if type(self.grid_size) == int:
            self.grid_size = [self.grid_size] * len(self.img_size)

        self.mag = mag
        if type(self.mag) == int or type(self.mag) == float:
            self.mag = [self.mag] * len(self.img_size)

        self.STN = SpatialTransformer(img_size)

        self._mag = torch.zeros([1, len(img_size), *[1] * len(img_size)])
        self._mag[0, :, 0, 0, 0] = self.mag

    def get_random_grid(self, n):
        return (
                (torch.rand([n, len(self.img_size), *self.grid_size]) - 0.5) * 2 * self._mag
        )

    def forward(self, x, with_flow=True):
        with torch.no_grad():
            size_x = x.size()
            N = size_x[0]
            random_grid = self.get_random_grid(N)
            x = self.STN(x, random_grid)
        if with_flow:
            return x, random_grid
        else:
            return x


def _check(w=400, h=400, c0=0, c1=1, blocksize=4):
    return np.kron(
        [[1, 0] * blocksize, [0, 1] * blocksize] * blocksize,
        np.ones((h // blocksize // 2, w // blocksize // 2)),
    )


# def build_test():
#     import sys, os
#
#     cwd = os.path.dirname(os.path.abspath(__file__))
#     sys.path.append(cwd + "/..")
#     from common import build_checkerline
#     import cv2
#     import matplotlib.pyplot as plt
#
#     img_size = (100, 100)
#     # img = np.random.rand(100,100)*0.5
#     img = build_checkerline(100, 100, 12)
#     cv2.circle(img, (10, 20), 5, (1))
#     plt.imshow(img)
#     plt.show()
#     # tf = tmp([100,100], 5, 0)
#     STN = SpatialTransformer(img_size)
#     zyx = []
#     # for i in range(len(self.theta)):
#     # tmp = (torch.rand(self.grid_size) - 0.5) * 2 * self.theta[i]
#     # zyx.append(tmp)
#     # grid =
#     # zyx = torch.stack(zyx, dim=0)


def ax(*args):
    for a in args:
        print(a)


if __name__ == "__main__":
    # build_test()
    # ax('train', 'f', 1, 3)

    # theta = 30
    # tr = 10
    # dim = 3
    # m = random_af_matrix(2,theta, tr, dim)
    # print(m.size())
    # print(m[0])
    from einops import rearrange
    import matplotlib.pyplot as plt

    w = 100
    # raf = Random_EL([w,w], 6, 5)
    raf = Compose([w, w], "R")
    a = torch.zeros([4, 1, w, w])
    a[:, :, 20:60, 20:60] = 1
    a1 = rearrange(a, "b c h w -> (b c h) w")
    # print(b.size())
    b, c = raf(a, a)
    # print(raf.last_applied[0])
    b = rearrange(b, "b c h w -> (b c h) w")

    # c = raf(a, is_reuse=True)
    # print(raf.last_applied[0])
    c = rearrange(c, "b c h w -> (b c h) w")
    d = raf(a)
    # print(raf.last_applied[0])
    d = rearrange(d, "b c h w -> (b c h) w")

    a = a.numpy(force=True)
    b = b.numpy(force=True)
    c = c.numpy(force=True)
    d = d.numpy(force=True)

    plt.subplot(141)
    plt.imshow(a1)
    plt.subplot(142)
    plt.imshow(b)
    plt.subplot(143)
    plt.imshow(c)
    plt.subplot(144)
    plt.imshow(d)
    plt.show()
