# Holography Sandbox — core package
from .waveoptics  import gerchberg_saxton, reconstruct, propagate_asm, propagate_fresnel
from .degradation import (
    degrade_resolution, quantise_phase, degrade_color,
    limit_viewing_angle, add_speckle, depth_planes_to_z_list,
)
from .metrics import mse, psnr, ssim, lpips_proxy, all_metrics
from .scenes  import (
    point_sources, gaussian_spots, resolution_chart,
    checkerboard, depth_gradient, letters, multi_depth_scene,
)
