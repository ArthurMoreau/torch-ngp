import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    ### training options
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    # (only valid when not using --cuda_ray)
    parser.add_argument('--num_steps', type=int, default=512)
    parser.add_argument('--upsample_steps', type=int, default=0)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (default is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    parser.add_argument('--ffw', action='store_true', help="use in the wild model")
    opt = parser.parse_args()
    print(opt)
    
    seed_everything(opt.seed)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
        model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        )
    elif opt.ffw:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFWNetwork
        train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        model = NeRFWNetwork(
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            N_vocab=len(train_loader.dataset)
        )
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork
    
    print(model)

    criterion = torch.nn.MSELoss()

    ### test mode
    if opt.test:

        trainer = Trainer('ngp', opt, model, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint='latest')

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=not opt.preload)

            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.
            
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:

        # optimizer = lambda model: torch.optim.Adam([
        #     {'name': 'encoding', 'params': list(model.encoder.parameters()), 'lr': 2e-2},
        #     {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6, 'lr': 1e-3},
        # ], betas=(0.9, 0.99), eps=1e-15)

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters()), 'lr': 2e-2},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()) + list(model.embedding_a.parameters()), 'weight_decay': 1e-6, 'lr': 1e-3},
        ], betas=(0.9, 0.99), eps=1e-15)

        # need different milestones for GUI/CMD mode.
        scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500, 2000] if opt.gui else [100, 200], gamma=0.33)

        trainer = Trainer('ngp', opt, model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, metrics=[PSNRMeter()], use_checkpoint='latest', eval_interval=10)

        # need different dataset type for GUI/CMD mode.

        if opt.gui:
            train_dataset = NeRFDataset(opt.path, type='all', mode=opt.mode, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=not opt.preload)
            trainer.train_loader = train_loader # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=not opt.preload)
            valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale, preload=opt.preload, fp16=opt.fp16)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, pin_memory=not opt.preload)

            trainer.train(train_loader, valid_loader, 10)

            # also test
            print(opt.path)
            test_path = os.path.join(opt.path, 'dense_test')
            print(test_path)
            test_dataset = NeRFDataset(test_path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=not opt.preload)
            
            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.evaluate(test_loader) # colmap doesn't have gt, so just test.
                # trainer.test(test_loader) # colmap doesn't have gt, so just test.
            
            trainer.save_mesh(resolution=256, threshold=10)