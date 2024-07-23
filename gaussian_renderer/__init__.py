#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from pdb import set_trace as st
from .image import linear_to_srgb

from utils.graphics_utils import normal_from_depth_image



def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, iteration=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    #override_color = pc.get_base_colour
    pipe.convert_SHs_python = True
    use_spec_mlp = True
    fresnel  = None

    #iteration = 1
    if iteration == None:
        iteration = 30000
    if override_color is None:
        if pipe.convert_SHs_python:
            if iteration < 3000 : 
                base = pc.get_base_colour
                three = torch.tensor(3.0, dtype=torch.float32).to("cuda")
                diffuse_linear = torch.sigmoid(base - torch.log(three))
                colors_precomp = diffuse_linear

            else:
                brdf = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                local_light = pc.get_local_lighting(pc.get_xyz).reshape(pc.get_features.shape).transpose(1,2)
                #st()
                out_light = brdf * local_light 
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                #tintrough = torch.sigmoid(pc.get_tint)
                #tint = torch.sigmoid(tintrough[:,:3])
                #roughness = torch.nn.functional.softplus(tintrough[:,-1] - 1.0)
                
                roughness = pc.get_roughness

                if use_spec_mlp:
                    spec, fresnel = pc.get_spec_mlp(pc.get_xyz, roughness, dir_pp_normalized, brdf.reshape((pc.get_features.shape[0], -1)), local_light.reshape((pc.get_features.shape[0], -1)))

                else:
                    _, norm_pred = pc.get_spec_mlp(pc.get_xyz,dir_pp_normalized, brdf.reshape((pc.get_features.shape[0], -1)), local_light.reshape((pc.get_features.shape[0], -1)))
                    
                    ref_vec = 2.0 * torch.sum(
                        norm_pred * dir_pp_normalized, dim=-1, keepdims=True) * norm_pred - dir_pp_normalized 
                    ref_vec = ref_vec/ref_vec.norm(dim=1, keepdim=True)

                    sh2rgb = eval_sh(pc.active_sh_degree, out_light, ref_vec)
                    spec = torch.relu(sh2rgb)

                base = pc.get_base_colour
                three = torch.tensor(3.0, dtype=torch.float32).to("cuda")
                diffuse_linear = torch.sigmoid(base - torch.log(three))
                #fresnel = fresnel * 0.3
                rgb = spec + diffuse_linear
                #rgb =  spec * (1 + fresnel) +  diffuse_linear * (1 - fresnel)

                #rgb = torch.clip(rgb, 0.0, 1.0)
                rgb = torch.clip(rgb, 0.0, 1.0)
                #rgb = torch.clip(linear_to_srgb(rgb), 0.0, 1.0)
                colors_precomp = rgb 
                #colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            brdf = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            local_light = pc.get_local_lighting(pc.get_xyz).reshape(pc.get_features.shape).transpose(1,2)
            shs = brdf * local_light



    else:
        colors_precomp = override_color.squeeze(1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    #opacity = opacity 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    """
    diffuse = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = diffuse_linear,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)[0]
    
    """

    # extra alphas
    out_extra = {"alpha":None}
    
    """
    norm_alpha = False
    if norm_alpha:

        raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
        alpha = torch.ones_like(means3D)        
        out_extra["alpha"] =  rasterizer_alpha(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = alpha,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        
    norm_pred = False
    if norm_pred:
        
        norm_pred = 0.5 * norm_pred + 0.5 # change to 0,1 range
        out_extra["pred_norm"] = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = norm_pred,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        out_extra["pred_norm"] =  (out_extra["pred_norm"] - 0.5 ) * 2.0 # change back to -1,1

        p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        depth = depth.repeat(1,3)

        out_extra["depth"] = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = depth,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)[0]

        out_extra["normal_ref"] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extra['depth'][0], bg_color=bg_color, alpha=out_extra['alpha'][0])

    """
    out = {"render": rendered_image,
            "diffuse_image": None,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
    
    out.update(out_extra)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return out
