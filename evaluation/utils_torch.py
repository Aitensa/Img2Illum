from __future__ import annotations

import math
import PIL.Image

import torch


def spherical_to_cartesian(points,device):
    theta = points[:, 0].to(device)
    phi = points[:, 1].to(device)
    r = points[:, 2].to(device)

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    return torch.stack((x, y, z), axis=-1)


def equirectangular_uv_to_cartesian(uv,device):
    u, v = uv[:, :, 0].to(device), uv[:, :, 1].to(device)

    u = (u // torch.max(u) - 0.5) * 2  # [-1, 1]
    phi = u * torch.deg2rad(torch.tensor([180.0])).to(device)
    phi = phi.view((-1))

    v = v // torch.max(v)  # [0, 1]
    v = v * torch.deg2rad(torch.tensor([180.0])).to(device)
    theta = v.view((-1)) 

    r = torch.ones_like(theta, dtype=torch.float32).to(device)

    coord_spherical = torch.stack((theta, phi, r), axis=-1)
    coord_cartesian = spherical_to_cartesian(coord_spherical,device)

    return coord_cartesian


def euler_rotation_xyz(vertices, device, angels=(0., 0., 0.)):
    # Euler XYZ rotation matrix
    rx, ry, rz = angels
    rx = torch.tensor([rx]).to(device)
    ry = torch.tensor([ry]).to(device)
    rz = torch.tensor([rz]).to(device)
    R_z = torch.tensor([
        [torch.cos(rz), -torch.sin(rz), 0],
        [torch.sin(rz), torch.cos(rz), 0],
        [0, 0, 1]
    ]).to(device)

    R_y = torch.tensor([
        [torch.cos(ry), 0, torch.sin(ry)],
        [0, 1, 0],
        [-torch.sin(ry), 0, torch.cos(ry)]
    ]).to(device)

    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(rx), -torch.sin(rx)],
        [0, torch.sin(rx), torch.cos(rx)]
    ]).to(device)

    R = R_z @ R_y @ R_x
    vertices_rotated = torch.mm(vertices, R.T)

    return vertices_rotated


class Canvas:
    def __init__(self, width, height, device, channels=3):
        self.data = torch.zeros((height, width, channels), dtype=torch.float32).to(device)

    def clear(self):
        self.data *= 0

    def to_pil_image(self,device):
        data_rgb = (self.data.to(device).astype(torch.float32) + 1) / 2 * 255
        data_rgb = data_rgb.astype(torch.uint8)
        return PIL.Image.fromtensor(data_rgb, mode='RGB')


def canvas_equirectangular_panorama(height, device,channels=3):
    c = Canvas(height * 2, height, device, channels=channels)

    u = torch.arange(height * 2, dtype=torch.int).to(device)
    v = torch.arange(height, dtype=torch.int).to(device)
    uv = torch.stack(torch.meshgrid(u, v), axis=-1)

    uv_xyz = equirectangular_uv_to_cartesian(uv,device)
    uv_xyz = euler_rotation_xyz(
        uv_xyz,device,
        (torch.deg2rad(torch.tensor([-90.0])).item(), torch.deg2rad(torch.tensor([0.])).item(), 0.))
    uv_xyz = euler_rotation_xyz(
        uv_xyz,device,
        (torch.deg2rad(torch.tensor([0.])).item(), torch.deg2rad(torch.tensor([90.0])).item(), 0.))

    c = Canvas(height * 2, height, device)
    c.data = uv_xyz.view((height, height * 2, 3))

    return c


class SphericalHarmonics:
    degrees: int
    cef: torch.tensor
    channel_order: str

    def __init__(self, device, degrees=2, channels=3, channel_order='first'):
        if degrees > 2:
            raise NotImplementedError('Only support degree <= 2')

        self.degrees = degrees
        self.cef = torch.zeros(((degrees + 1) ** 2, channels),dtype=torch.float32,requires_grad=True).to(device)
        self.channel_order = channel_order

    @property
    def coefficients(self):
        if self.channel_order == 'first':
            return self.cef.permute(1, 0)
        elif self.channel_order == 'last':
            return self.cef
        else:
            raise ValueError('channel_order must be "first" or "last"')

    @coefficients.setter
    def coefficients(self, value):
        self.cef = value

    @staticmethod
    def from_tensor(sh_coefficients, device, channel_order='first'):
        if len(sh_coefficients.shape) == 1:
            if channel_order == 'first':
                sh_coefficients = sh_coefficients.view((3, -1))
            else:
                sh_coefficients = sh_coefficients.view((-1, 3))

        s_0, s_1 = sh_coefficients.shape

        n_channels = s_0 if channel_order == 'first' else s_1
        n_components = s_1 if channel_order == 'first' else s_0

        degrees =int(math.sqrt(n_components)) - 1
        sh = SphericalHarmonics(device=device, degrees=degrees)
        sh.coefficients = sh_coefficients
        sh.channel_order = channel_order

        return sh

    @staticmethod
    def from_sphere_points(points, device, degrees=2):
        sh = SphericalHarmonics(device,degrees=degrees)
        sh.project_sphere_points(points)

        return sh

    def get_batched_basis_at(self, device, normal_matrix):
        matrix_length = normal_matrix.shape[0]
        x, y, z = normal_matrix[:, 0].copy().to(device), normal_matrix[:, 1].to(device), normal_matrix[:, 2].to(device)
        sh_basis = torch.zeros((matrix_length, 9)).to(device)

        # degree 0
        if self.degrees >= 0:
            sh_basis[:, 0] = 0.282095

        # degree 1
        if self.degrees >= 1:
            sh_basis[:, 1] = 0.488603 * y
            sh_basis[:, 2] = 0.488603 * z
            sh_basis[:, 3] = 0.488603 * x

        # degree 2
        if self.degrees >= 2:
            sh_basis[:, 4] = 1.092548 * x * y
            sh_basis[:, 5] = 1.092548 * y * z
            sh_basis[:, 6] = 0.315392 * (3 * z * z - 1)
            sh_basis[:, 7] = 1.092548 * x * z
            sh_basis[:, 8] = 0.546274 * (x * x - y * y)

        return sh_basis

    def project_sphere_points(self, device, sphere_points):
        matrix_sh_basis = self.get_batched_basis_at(
            sphere_points.positions)

        c = torch.einsum(
            'ij,ik->ijk',
            matrix_sh_basis,
            sphere_points.features).sum(axis=0)

        norm = (4 * torch.tensor([math.pi]).to(device)) / sphere_points.positions.shape[0]
        self.coefficients = c * norm

    def reconstruct(self,device, canvas_norm: torch.Tensor):
        s = canvas_norm.shape

        canvas_norm = canvas_norm.view((-1, 3))

        # x = canvas_norm[:, 0, torch.newaxis]
        # y = canvas_norm[:, 1, torch.newaxis]
        # z = canvas_norm[:, 2, torch.newaxis]
        
        # x = canvas_norm[:,0].unsqueeze(2)
        #y = canvas_norm[:,1].unsqueeze(2)
        #z = canvas_norm[:,2].unsqueeze(2)
        t = canvas_norm.unsqueeze(2).requires_grad_().to(device)
        x = t[:,0,:]
        y = t[:,1,:]
        z = t[:,2,:]
        canvas = torch.zeros_like(canvas_norm, dtype=torch.float32).to(device)

        if self.degrees >= 0:
            canvas += self.coefficients[0, :] * 0.886227

        if self.degrees >= 1:
            canvas += self.coefficients[1, :] * 2.0 * 0.511664 * y
            canvas += self.coefficients[2, :] * 2.0 * 0.511664 * z
            canvas += self.coefficients[3, :] * 2.0 * 0.511664 * x

        if self.degrees >= 2:
            canvas += self.coefficients[4, :] * 2.0 * 0.429043 * x * y
            canvas += self.coefficients[5, :] * 2.0 * 0.429043 * y * z
            canvas += self.coefficients[6, :] * 0.743125 * z * z - 0.247708
            canvas += self.coefficients[7, :] * 2.0 * 0.429043 * x * z
            canvas += self.coefficients[8, :] * 0.429043 * (x * x - y * y)

        canvas = canvas.view(s)

        return canvas
    def reconstruct_back(self,device, canvas_norm: torch.Tensor):
        s = canvas_norm.shape

        canvas_norm = canvas_norm.view((-1, 3))

        # x = canvas_norm[:, 0, torch.newaxis]
        # y = canvas_norm[:, 1, torch.newaxis]
        # z = canvas_norm[:, 2, torch.newaxis]
        
        # x = canvas_norm[:,0].unsqueeze(2)
        #y = canvas_norm[:,1].unsqueeze(2)
        #z = canvas_norm[:,2].unsqueeze(2)
        t = canvas_norm.unsqueeze(2).requires_grad_().to(device)
        x = t[:,0,:]
        y = t[:,1,:]
        z = t[:,2,:]
        canvas = torch.zeros_like(canvas_norm, dtype=torch.float32).to(device)

        if self.degrees >= 0:
            canvas += self.coefficients[0, :] * 0.282095

        if self.degrees >= 1:
            canvas += self.coefficients[1, :] * 0.488603  * y
            canvas += self.coefficients[2, :] * 0.488603  * z
            canvas += self.coefficients[3, :] * 0.488603  * x

        if self.degrees >= 2:
            canvas += self.coefficients[4, :] * 1.092548 * x * y
            canvas += self.coefficients[5, :] * 1.092548 * y * z
            canvas += self.coefficients[6, :] * 0.946176 * z * z - 0.315392
            canvas += self.coefficients[7, :] * 1.092548* x * z
            canvas += self.coefficients[8, :] * 0.546274 * (x * x - y * y)

        canvas = canvas.reshape(s)

        return canvas
    
    def reconstruct_to_canvas(self, device, canvas=None):
        if canvas is None:
            canvas = canvas_equirectangular_panorama(
                height=128, device=device)

        canvas_res = canvas_equirectangular_panorama(
            height=canvas.data.shape[0],device=device)
        canvas_res.data = self.reconstruct(device,canvas.data)

        return canvas_res

    def vis_as_pil_image(self, device, canvas=None):
        if canvas is None:
            canvas = canvas_equirectangular_panorama(
                height=128,device=device)

        arr = self.reconstruct(device,canvas.data)
        img = PIL.Image.fromtensor((arr * 255).astype(torch.uint8))

        return img





