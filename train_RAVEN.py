# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
import pickle as pkl
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import re
import json
from datetime import datetime

from models import DiT_models, DiT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image, make_grid

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if True: #dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])




import torch
import einops 
from tqdm import trange, tqdm
from torch.utils.data import Dataset, DataLoader
def _sample_panels(train_row_img, cmb_per_class=3333):
    X = []
    y = []
    row_ids = []
    n_classes = train_row_img.shape[0]
    n_samples = train_row_img.shape[1]
    for iclass in trange(n_classes):
        tuple_loader = DataLoader(range(n_samples), batch_size=3, shuffle=True, drop_last=True)
        X_class = []
        row_ids_cls = []
        while True:
            try:
                batch = next(iter(tuple_loader))
                rows = train_row_img[iclass][batch]
                mtg = torch.cat(tuple(rows), dim=1)
                X_class.append(mtg)
                row_ids_cls.append(batch)
                if len(X_class) == cmb_per_class:
                    break
            except StopIteration:
                tuple_loader = DataLoader(range(n_samples), batch_size=3, shuffle=True, drop_last=True)

        y_class = torch.tensor([iclass]*len(X_class), dtype=torch.int)
        X_class = torch.stack(X_class)
        row_ids_cls = torch.stack(row_ids_cls)
        y.append(y_class)
        X.append(X_class)
        row_ids.append(row_ids_cls)
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    row_ids = torch.cat(row_ids, dim=0)
    return X, y, row_ids


class dataset_PGM_abstract(Dataset): 
    def __init__(self, cmb_per_class=3333, train_attrs=None, device="cpu", onehot=False): 
        """attr_list: [num_samples, 3, 9, 3]"""
        if train_attrs is None:
            train_attrs = torch.load('/n/home12/binxuwang/Github/DiffusionReasoning/train_inputs.pt') # [35, 10000, 3, 9, 3]
        n_classes = train_attrs.shape[0] # 35
        n_samples = train_attrs.shape[1] # 10k
        self.labels = torch.arange(0, n_classes).unsqueeze(1).expand(n_classes, n_samples)
        train_attrs = train_attrs.to(torch.int32)
        self.train_row_img = einops.rearrange(train_attrs, 'c s pnl (H W) attr -> c s attr H (pnl W)', H=3, W=3, attr=3, pnl=3)
        self.X, self.y, self.row_ids = _sample_panels(self.train_row_img, cmb_per_class)
        self.X = self.X.to(device) # [35 * cmb_per_class, 3, 9, 9]
        if onehot is True:
            O1 = torch.cat([torch.eye(7, 7, dtype=int), torch.zeros(1, 7, dtype=int)], dim=0)
            O2 = torch.cat([torch.eye(10, 10, dtype=int), torch.zeros(1, 10, dtype=int)], dim=0)
            O3 = torch.cat([torch.eye(10, 10, dtype=int), torch.zeros(1, 10, dtype=int)], dim=0)
            X_onehot = torch.cat([O1[self.X[:, 0], :], O2[self.X[:, 1], :], O3[self.X[:, 2], :], ], dim=-1)
            print(X_onehot.shape)
            self.X = einops.rearrange(X_onehot, 'b h w C -> b C h w')
            print(self.X.shape)
            self.Xmean = torch.tensor([0.5, ]).view(1, 1, 1, 1)
            self.Xstd = torch.tensor([0.5, ]).view(1, 1, 1, 1)
            self.X = (self.X.float() - self.Xmean) / self.Xstd
        else:
            self.Xmean = torch.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to(device)
            self.Xstd = torch.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to(device)
            self.X = (self.X - self.Xmean) / self.Xstd
        
    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, idx): 
        """attr: [3, 9, 3]"""
        return self.X[idx], self.y[idx]
    
    def dict(self):
        return {'row_ids': self.row_ids, 'y': self.y}



DiT_configs = {
    # "DiT_XL_2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    "DiT_XL_1": {"depth": 28, "hidden_size": 1152, "patch_size": 1, "num_heads": 16},
    "DiT_XL_3": {"depth": 28, "hidden_size": 1152, "patch_size": 3, "num_heads": 16},
    # "DiT_XL_4": {"depth": 28, "hidden_size": 1152, "patch_size": 4, "num_heads": 16},
    # "DiT_XL_8": {"depth": 28, "hidden_size": 1152, "patch_size": 8, "num_heads": 16},
    # "DiT_L_2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "DiT_L_1": {"depth": 24, "hidden_size": 1024, "patch_size": 1, "num_heads": 16},
    "DiT_L_3": {"depth": 24, "hidden_size": 1024, "patch_size": 3, "num_heads": 16},
    # "DiT_L_4": {"depth": 24, "hidden_size": 1024, "patch_size": 4, "num_heads": 16},
    # "DiT_L_8": {"depth": 24, "hidden_size": 1024, "patch_size": 8, "num_heads": 16},
    # "DiT_B_2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "DiT_B_1": {"depth": 12, "hidden_size": 768, "patch_size": 1, "num_heads": 12},
    "DiT_B_3": {"depth": 12, "hidden_size": 768, "patch_size": 3, "num_heads": 12},
    # "DiT_B_4": {"depth": 12, "hidden_size": 768, "patch_size": 4, "num_heads": 12},
    # "DiT_B_8": {"depth": 12, "hidden_size": 768, "patch_size": 8, "num_heads": 12},
    # "DiT_S_2": {"depth": 12, "hidden_size": 384, "patch_size": 2, "num_heads": 6},
    "DiT_S_1": {"depth": 12, "hidden_size": 384, "patch_size": 1, "num_heads": 6},
    "DiT_S_3": {"depth": 12, "hidden_size": 384, "patch_size": 3, "num_heads": 6},
    # "DiT_S_4": {"depth": 12, "hidden_size": 384, "patch_size": 4, "num_heads": 6},
    # "DiT_S_8": {"depth": 12, "hidden_size": 384, "patch_size": 8, "num_heads": 6},
}

def get_max_index(folder_path):
    import re
    indexes = []
    for item in os.listdir(folder_path):
        match = re.match(r"(\d+)-", item)
        if match:
            index = int(match.group(1))
            indexes.append(index)
    if indexes:
        return max(indexes)
    else:
        return 0
    
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # seed = args.global_seed * dist.get_world_size() + rank
    rank = 0
    device = 0
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    print(f"Starting rank={rank}, seed={seed}, ")#world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_index = get_max_index(args.results_dir) + 1
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.dataset}-{'cond' if args.cond else 'uncond'}-{model_string_name}_{run_id}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        samples_dir = f"{experiment_dir}/samples"  # Stores generated samples
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # log all args
        logger.info(f"Args: {args}")
        # dump args
        with open(f'{experiment_dir}/args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        logger = create_logger(None)

    # Create model:
    # assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size #  // 8
    model_cfg = DiT_configs[args.model]
    class_dropout_prob = args.class_dropout_prob # default 1.0, unconditional model
    is_conditional = args.cond
    if is_conditional:
        num_classes = args.num_classes
    else: # unconditional
        num_classes = 0
        class_dropout_prob = 1.0
    # load train_attrs
    train_attr_fn = args.train_attr_fn 
    train_attrs = torch.load(f'/n/home12/binxuwang/Github/DiffusionReasoning/{train_attr_fn}') # [35, 10000, 3, 9, 3]
    if args.dataset == 'RAVEN10_abstract':
        dataset = dataset_PGM_abstract(cmb_per_class=args.cmb_per_class, device='cpu', train_attrs=train_attrs)
        pkl.dump(dataset.dict(), open(f'{experiment_dir}/dataset_idx.pkl', 'wb'))
        print("Normalization", dataset.Xmean, dataset.Xstd)
        classes = []
        model = DiT(input_size=9,
            in_channels=3, **model_cfg,
            # patch_size=1, depth=12, hidden_size=384, num_heads=6,
            mlp_ratio=4.0,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=True,)
    elif args.dataset == 'RAVEN10_abstract_onehot':
        dataset = dataset_PGM_abstract(cmb_per_class=args.cmb_per_class, device='cpu', onehot=True, train_attrs=train_attrs)
        pkl.dump(dataset.dict(), open(f'{experiment_dir}/dataset_idx.pkl', 'wb'))
        print("Normalization", dataset.Xmean, dataset.Xstd)
        classes = []
        model = DiT(input_size=9,
            in_channels=27, **model_cfg,
            # patch_size=1, depth=12, hidden_size=384, num_heads=6,
            mlp_ratio=4.0,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=True,)
    elif args.dataset == 'MNIST':
        transform = transforms.Compose([
                                transforms.Resize((32, 32), antialias=True),
                                transforms.ToTensor(), 
                                # transforms.Pad(2, padding_mode='reflect'),
                                transforms.Normalize((0.5,), (0.5,))])
        dataset = MNIST("~/Datasets", train=True, download=False, 
                            transform=transform)
        model = DiT(input_size=32,
            in_channels=1, **model_cfg,
            # patch_size=4, depth=12, hidden_size=384, num_heads=6,
            mlp_ratio=4.0,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=True,)
    else:
        raise NotImplementedError(f'dataset {args.dataset} not implemented')
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # model = DDP(model.to(device), device_ids=[rank])
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    diffusion_eval = create_diffusion(timestep_respacing=args.eval_sampler)  # default: ddim100, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size, #int(args.global_batch_size // dist.get_world_size()),
        # shuffle=False,
        shuffle=True,
        # sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0) #.module # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            if is_conditional:
                y = y.to(device)
            else:
                y = torch.zeros_like(y).to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model) # .module

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                # avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.state_dict(), # .module
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                # dist.barrier()
            
            # save samples 
            if train_steps % args.save_samples_every == 0 or (train_steps == 1):
                if rank == 0:
                    model.eval() 
                    if is_conditional:# conditional case
                        y = torch.randint(0, args.num_classes, (args.num_eval_sample,), device=device)
                    else:# unconditional case
                        y = args.num_classes * torch.ones((args.num_eval_sample,), dtype=torch.int, device=device)
                    model_kwargs = dict(y=y)
                    # Sample images:
                    z = torch.randn(args.num_eval_sample, model.in_channels, latent_size, latent_size, device=device)
                    with torch.no_grad():
                        samples = diffusion_eval.ddim_sample_loop(
                            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                        )
                    model.train() 
                    if "RAVEN" in args.dataset:
                        # samples = einops.rearrange(samples, 'b c h w -> b h w c')
                        samples = (samples.detach().cpu() * dataset.Xstd) + dataset.Xmean
                        torch.save(samples, f"{samples_dir}/{train_steps:07d}.pt")
                    else:
                        samples = (0.5 * samples + 0.5).clamp(0, 1)
                        # samples = ema.decode(samples).sample
                        # Save and display images:
                        save_image(samples, f"{samples_dir}/{train_steps:07d}.png", nrow=8, ) #normalize=True, value_range=(-1, 1))
                # dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0:
        checkpoint = {
            "model": model.state_dict(), # .module
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        model.eval() 
        if is_conditional:# conditional case
            y = torch.randint(0, args.num_classes, (args.num_eval_sample,), device=device)
        else:# unconditional case
            y = args.num_classes * torch.ones((args.num_eval_sample,), dtype=torch.int, device=device)
        model_kwargs = dict(y=y)
        # Sample images:
        z = torch.randn(args.num_eval_sample, model.in_channels, latent_size, latent_size, device=device)
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        model.train() 
        if "RAVEN" in args.dataset:
            # samples = einops.rearrange(samples, 'b c h w -> b h w c')
            samples = (samples.detach().cpu() * dataset.Xstd) + dataset.Xmean
            torch.save(samples, f"{samples_dir}/{train_steps:07d}.pt")
        else:
            samples = (0.5 * samples + 0.5).clamp(0, 1)
            # samples = ema.decode(samples).sample
            # Save and display images:
            save_image(samples, f"{samples_dir}/{train_steps:07d}.png", nrow=8, ) #normalize=True, value_range=(-1, 1))
    # cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["RAVEN10_abstract", "RAVEN10_abstract_onehot", "MNIST"], default="RAVEN10_abstract")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results")
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--model", type=str, choices=list(DiT_configs.keys()), default="DiT_L_1")
    parser.add_argument("--image-size", type=int, choices=[32, 9], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--save-samples-every", type=int, default=1000)
    parser.add_argument("--num_eval_sample", type=int, default=64)
    parser.add_argument("--eval_sampler", type=str, default="ddim100")
    parser.add_argument("--cmb_per_class", type=int, default=3333)
    parser.add_argument("--class_dropout_prob", type=float, default=1.0)
    # add conditional flag
    parser.add_argument("--cond", action="store_true")
    parser.add_argument("--train_attr_fn", type=str, default="train_inputs.pt") # "train_inputs_new.pt"
    
    args = parser.parse_args()
    if not args.cond:
        args.num_classes = 0
        args.class_dropout_prob = 1.0
    main(args)
