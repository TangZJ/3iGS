# Lifted from https://github.com/apchenstu/TensoRF

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
from .sh import eval_sh_bases
from pdb import set_trace as st
from .refl_utils import generate_ide_fn

def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

    
class MLPRender_Fea_shade(torch.nn.Module):
    def __init__(self,device, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea_shade, self).__init__()

        #self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        #self.feape = feape
        self.inChanel = inChanel
        self.in_mlpC = inChanel * 2  + 3 + 2 * viewpe *3 + 14


        self.roughness = torch.nn.Linear(inChanel, 1)
        self.rougness_act = torch.nn.Softplus()

        self.env = torch.nn.Sequential(torch.nn.Linear(inChanel,featureC), torch.nn.ReLU(inplace=True), torch.nn.Linear(featureC, featureC), torch.nn.ReLU(inplace=True),torch.nn.Linear(featureC, 3), torch.nn.Sigmoid())

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)

        self.reg = torch.nn.Tanh()
        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True))

        self.layer3 = torch.nn.Sequential(torch.nn.Linear(featureC,3, bias=False))


        self.dir_enc_fun = generate_ide_fn(viewpe)

    def forward(self, pts, kd, viewdirs, brdf, local_light):

        brdf = self.reg(brdf)
        local_light = self.reg(local_light)
        kd = kd.unsqueeze(1)
        env_light = self.env(brdf) * kd

        roughness = self.roughness(brdf)
        roughness = self.rougness_act(roughness - 1.0)

        indata = [brdf , local_light , viewdirs]

        if self.viewpe > 0:
            indata += [self.dir_enc_fun.integrated_dir_enc_fn(viewdirs, roughness)]
            #indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgba = self.mlp(mlp_in)

        spec = self.layer3(rgba)
        spec = torch.tanh(spec) * (1 - kd)
        rgb = spec + env_light
        return rgb, None

    
class MLPRender_Fea(torch.nn.Module):
    def __init__(self, device, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        #self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        #self.feape = feape
        self.inChanel = inChanel
        self.in_mlpC = inChanel * 2  + 3 + 2 * viewpe *3 + 14

        #self.roughness = torch.nn.Linear(inChanel, 1)
        self.rougness_act = torch.nn.Softplus()

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        #layer2_1 = torch.nn.Linear(featureC, featureC)



        self.reg = torch.nn.Tanh()
        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True))

        self.layer3 = torch.nn.Sequential(torch.nn.Linear(featureC,3, bias=False))
        #self.fresnel = torch.nn.Linear(featureC, 1)

        #self.fresnel_act = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
        #torch.nn.init.constant_(self.mlp[-2].bias, 0)
        #torch.nn.init.constant_(self.layer3.bias, 0)
        #torch.nn.init.constant_(self.layer3_fresnel.bias, 0)
        #torch.nn.init.constant_(self.fresnel.bias, 1.0)

        self.dir_enc_fun = generate_ide_fn(viewpe)

    def forward(self, pts, roughness, viewdirs, brdf, local_light):

        brdf = self.reg(brdf)
        local_light = self.reg(local_light)
        indata = [brdf , local_light , viewdirs]

        #roughness = self.roughness(brdf)
        #spec_tint = torch.sigmoid(tr)
        roughness = self.rougness_act(roughness - 1.0)
        if self.viewpe > 0:
            indata += [self.dir_enc_fun.integrated_dir_enc_fn(viewdirs, roughness)]
            #indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        spec_color = self.mlp(mlp_in)

        spec_color = self.layer3(spec_color)
        spec_color = torch.tanh(spec_color)

        return spec_color, None

class ref_mlp(torch.nn.Module):

    def __init__(self, brdf_fea, viewpe):
        super(ref_mlp, self).__init__()
        self.brdf_fea = brdf_fea

        self.mlp = torch.nn.Sequential(
             torch.nn.Linear(self.brdf_fea + 2 * viewpe * 3, self.brdf_fea),
             torch.nn.ReLU(),
             torch.nn.Linear(self.brdf_fea, 3),
             torch.nn.Tanh())
        
    def forward(self, brdf, viewdirs):

        norm_pred = self.mlp(brdf)
        norm_pred = norm_pred/norm_pred.norm(dim=1, keepdim=True)

        ref_vec = 2.0 * torch.sum(
            norm_pred * viewdirs, dim=-1, keepdims=True) * norm_pred - viewdirs 
        ref_vec = ref_vec/ref_vec.norm(dim=1, keepdim=True)
        return ref_vec, norm_pred



class TensorBase(torch.nn.Module):
    def __init__(self,n_voxels, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = 16  * 3
        self.aabb = aabb
        self.n_voxels = n_voxels
        self.alphaMask = alphaMask
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.init_svd_volume(gridSize[0], device)

        self.global_light = torch.nn.Parameter()
        self.renderModule = MLPRender_Fea(device, self.app_dim, view_pe, fea_pe, featureC).to(device)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)


    def init_svd_volume(self, res, device):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def forward(self, xyz):

        #local_light = self.compute_appfeature(xyz_samples)

        local_lighting = self.compute_appfeature(xyz)
        
        return local_lighting

    def forward_shader(self, pts, roughness, viewdirs, brdf, local_light):

        spec, norm_pred = self.renderModule(pts, roughness, viewdirs, brdf, local_light)
        return spec, norm_pred


class TensorVMSplit(TensorBase):
    def __init__(self, n_voxels, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(n_voxels, aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        #self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.01, device)
        self.basis_mat =  torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=True).to(device)



    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
        
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [#{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                    {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                    {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]], \
                                                align_corners=True,mode='bilinear').view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]], \
                                            align_corners=True, mode='bilinear').view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        #self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')


    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]

            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]

            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )

        """
        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb
        """
        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
        reso_cur = N_to_reso(self.n_voxels, self.aabb)
        self.upsample_volume_grid(reso_cur)