# Experiment design
Tested with the default settings on the Lego test dataset. Here the speed refers to the `iterations per second` on a RTX 2080.
| Dataset | Blender | Blender_color | JardinMines1| JardinMines7 |
| - | - | - | - | - |
| NeRF 8| - | - | - | A | x |
| torch-ngp(cuda_ray + fflp) 4 | - | - | - | - |
| torch-ngp(cuda_ray + fflp) 8 | - | - | - | - |
| NerF-W 8| x | - | - | o |
| torch-ngp-la(cuda_ray + fflp) 4 | x | - | x | - |
| torch-ngp-la(cuda_ray + fflp) 8 | x | - | x | - |
