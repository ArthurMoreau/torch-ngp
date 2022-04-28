# Experiment design
Tested with the default settings on the Lego test dataset. Here the speed refers to the `iterations per second` on a RTX 2080.
| Dataset | Blender | Blender_color | JardinMines1| JardinMines7 |
| - | - | - | - | - |
| NeRF 8| - | - | - | A | x |
| torch-ngp(cuda_ray + fflp) 8 | - | - | ssim: 0.326, psnr: 13.5, mse:0.0471, lpips:0.548| ssim:0.458, psnr: 15.3, mse:0.0310, lpips: 0.470|
| torch-ngp(cuda_ray + fflp) 4 | ssim: 0.971, psnr:31.0, mse: 0.000895, lpips:0.0325 | - | - | ssim: 0.356, psnr: 14.4, mse: 0.0376, lpips:0.513 |
| NerF-W 8| x | - | - | o |
| torch-ngp-la(cuda_ray + fflp) 4 | x | - | x | - |
| torch-ngp-la(cuda_ray + fflp) 8 | ssim: 0.969, psnr:30.1, mse: 0.00114, lpips:0.0357 | ssim:0.937, psnr: 25.2, mse:0.00331, lpips:0.0564 | x | - |
