import argparse
import os
import time
import datetime
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np # For dummy data
import glob
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import cv2 # For augmentations like homography

# Attempt to import SPyNetModified and augmentations
# This assumes 'spynet' is a sibling directory to 'rdn' or in PYTHONPATH
import sys
# Add the parent directory of 'rdn' to sys.path to allow sibling imports
# This might be needed if running this script directly from the 'rdn' folder
# For a proper package structure, these imports would be different.
current_dir_rdn = os.path.dirname(os.path.abspath(__file__))
parent_dir_rdn = os.path.dirname(current_dir_rdn)
if parent_dir_rdn not in sys.path:
    sys.path.append(parent_dir_rdn)

try:
    from spynet.spynet_modified import SPyNetModified, backwarp as spynet_backwarp
    # from spynet.augmentations import ... # If needed directly by dataset here
    # from spynet.dataset_flow import IMG_HEIGHT, IMG_WIDTH # If using fixed sizes
    from spynet.augmentations import IMG_HEIGHT as DEFAULT_SPYNET_IMG_HEIGHT # For consistency if needed
    from spynet.augmentations import IMG_WIDTH as DEFAULT_SPYNET_IMG_WIDTH
except ImportError as e:
    print(f"Could not import SPyNetModified or spynet_backwarp. Ensure spynet module is accessible: {e}")
    SPyNetModified = None # Placeholder
    spynet_backwarp = None
    DEFAULT_SPYNET_IMG_HEIGHT = 192 # Fallback
    DEFAULT_SPYNET_IMG_WIDTH = 320  # Fallback

from models import RDNInpainting # from rdn.models

# --- Configuration Defaults ---
# RDN Architecture (from paper/RDN-pytorch defaults)
DEFAULT_NUM_FEATURES = 64  # G0
DEFAULT_GROWTH_RATE = 64   # G
DEFAULT_NUM_BLOCKS = 16    # D
DEFAULT_NUM_LAYERS = 8     # C

# Input/Output for Inpainting RDN
DEFAULT_NUM_INPUT_CHANNELS = 32 # As per f_in: I_k^m (3), S_k (1), (K-1)*(Î_j^m(3), Š_j(1), V_j(1), f_kj^m(2)) = 4 + 4*7 = 32 for K=5
DEFAULT_NUM_OUTPUT_CHANNELS = 3  # For RGB residual

# Training parameters (from paper)
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 4e-5 # 4*10^-5
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 16 # Adjust based on GPU memory
DEFAULT_K_FRAMES = 5    # Burst size

# Image/Patch size (from paper)
DEFAULT_IMG_WIDTH = 320
DEFAULT_IMG_HEIGHT = 192


# --- Warping function (can use or adapt from SPyNet) ---
def warp_frame_with_flow(frame_tensor, flow_tensor):
    """ Warps frame_tensor according to flow_tensor using spynet_backwarp. """
    if spynet_backwarp is None:
        raise RuntimeError("spynet_backwarp function not imported. Cannot warp frame.")
    # spynet_backwarp expects flow to be in pixel displacements.
    # It also expects Batch, C, H, W for frame and Batch, 2, H, W for flow.
    # Ensure input tensors are 4D (add batch dim if necessary) and C is appropriate.
    is_single_frame = frame_tensor.dim() == 3
    is_single_flow = flow_tensor.dim() == 3

    if is_single_frame:
        frame_tensor = frame_tensor.unsqueeze(0)
    if is_single_flow:
        flow_tensor = flow_tensor.unsqueeze(0)
    
    warped_frame = spynet_backwarp(frame_tensor, flow_tensor)
    
    if is_single_frame:
        warped_frame = warped_frame.squeeze(0)
    return warped_frame

def create_validity_mask(frame_height, frame_width, flow_tensor_for_warp):
    """
    Creates a validity mask. Pixels are valid if their source coordinates for warping were within bounds.
    The flow tensor (flow_tensor_for_warp) gives displacements (dx, dy).
    Normalized grid coordinates range from -1 to 1.
    A source coordinate (x_s, y_s) is sampled for each target pixel (x_t, y_t).
    grid_x(x_t, y_t) = 2*x_t/(W-1) - 1 + 2*dx(x_t,y_t)/(W-1) 
    grid_y(x_t, y_t) = 2*y_t/(H-1) - 1 + 2*dy(x_t,y_t)/(H-1)
    Pixel is valid if -1 <= grid_x <= 1 and -1 <= grid_y <= 1.
    """
    B, _, H, W = flow_tensor_for_warp.shape if flow_tensor_for_warp.dim() == 4 else (1, *flow_tensor_for_warp.shape)
    if flow_tensor_for_warp.dim() == 3: # Add batch dim if not present
        flow_tensor_for_warp = flow_tensor_for_warp.unsqueeze(0)

    # Create a base grid of [-1, 1] coordinates
    grid_y_base, grid_x_base = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=flow_tensor_for_warp.device),
        torch.linspace(-1.0, 1.0, W, device=flow_tensor_for_warp.device),
        indexing='ij'
    )
    grid_base = torch.stack((grid_x_base, grid_y_base), dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # B, 2, H, W

    # Normalize flow to also be in [-1, 1] range relative to total size
    norm_flow_x = flow_tensor_for_warp[:, 0:1, :, :] * (2.0 / (W - 1.0))
    norm_flow_y = flow_tensor_for_warp[:, 1:2, :, :] * (2.0 / (H - 1.0))
    norm_flow = torch.cat([norm_flow_x, norm_flow_y], dim=1) # B, 2, H, W

    # Final sampling grid used by F.grid_sample
    sampling_grid = grid_base + norm_flow # B, 2, H, W
    
    # Check if sampling coordinates are within [-1, 1]
    valid_x = (sampling_grid[:, 0, :, :] >= -1.0) & (sampling_grid[:, 0, :, :] <= 1.0)
    valid_y = (sampling_grid[:, 1, :, :] >= -1.0) & (sampling_grid[:, 1, :, :] <= 1.0)
    
    validity_mask = (valid_x & valid_y).float().unsqueeze(1) # B, 1, H, W
    return validity_mask.squeeze(0) if B == 1 else validity_mask # Return C, H, W or B, C, H, W

# --- Dataset Placeholder ---
class InpaintingDataset(Dataset):
    def __init__(self, config, spynet_model_path, spynet_model_name_for_gt_flow_in_spynet_m, is_train=True):
        self.config = config
        self.is_train = is_train
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.k_frames = config.k_frames

        self.vimeo_root_dir = config.vimeo_dir # Path to vimeo_septuplet/sequences
        self.defencing_root_dir = config.defencing_dir # Path to De-fencing-master/dataset
        
        self.vimeo_sequences_paths = self._load_vimeo_sequences()
        self.fence_items = self._load_defencing_items()

        if not self.vimeo_sequences_paths:
            raise ValueError(f"No Vimeo sequences found. Check path and list file: {self.vimeo_root_dir}")
        if not self.fence_items:
            raise ValueError(f"No fence items found. Check path: {self.defencing_root_dir}")

        print(f"Loaded {len(self.vimeo_sequences_paths)} Vimeo sequence groups.")
        print(f"Loaded {len(self.fence_items)} fence items.")

        print("Loading pre-trained SPyNetModified for flow calculation...")
        if SPyNetModified is None or spynet_backwarp is None:
            raise ImportError("SPyNetModified or spynet_backwarp could not be imported. Cannot proceed.")
        
        self.spynet_m = SPyNetModified(model_name=spynet_model_name_for_gt_flow_in_spynet_m, pretrained=False)
        try:
            # Ensure spynet_m is on the main device specified by config for initial loading
            # Workers will handle model placement if they have their own GPUs.
            # For now, load to CPU then it can be moved by the main training loop or worker.
            map_location = torch.device('cpu') 
            checkpoint = torch.load(spynet_model_path, map_location=map_location)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k 
                new_state_dict[name] = v
            self.spynet_m.load_state_dict(new_state_dict)
            print(f"Successfully loaded SPyNetModified weights from {spynet_model_path}")
        except Exception as e:
            raise IOError(f"Error loading SPyNetModified weights from {spynet_model_path}: {e}")
        
        self.spynet_m.eval() # Set to evaluation mode
        # self.spynet_m_device = torch.device(config.device) # For use in __getitem__
        # self.spynet_m.to(self.spynet_m_device) 

        # Augmentation Transforms (can be more sophisticated)
        self.to_tensor = T.ToTensor()
        self.h_flip = T.RandomHorizontalFlip(p=0.5)
        # Perspective transform params for fences (example)
        self.perspective_distorter = T.RandomPerspective(distortion_scale=0.3, p=1.0) 

    def _load_vimeo_sequences(self):
        # Assumes vimeo_dir is the root of vimeo_septuplet (contains sep_trainlist.txt etc.)
        # And sequences are in vimeo_dir/sequences/
        list_file_name = "sep_trainlist.txt" if self.is_train else "sep_testlist.txt"
        vimeo_list_file = os.path.join(os.path.dirname(self.vimeo_root_dir.rstrip('/')), list_file_name)
        sequence_base_path = self.vimeo_root_dir 
        
        if not os.path.exists(vimeo_list_file):
            print(f"Warning: Vimeo list file not found: {vimeo_list_file}. Trying to scan directory {sequence_base_path} for sequences.")
            # Fallback: scan directory for sequences (e.g. 00001/0001, 00001/0002 ...)
            # This is a simplified fallback, assumes structure like vimeo_septuplet/sequences/GROUP/SEQ_NUM/
            # This part might need robust implementation if list file is missing.
            # For now, let's stick to requiring the list file as per FlowDataset in spynet
            raise FileNotFoundError(f"Vimeo list file not found: {vimeo_list_file}")

        vimeo_sequences = []
        with open(vimeo_list_file, 'r') as f:
            for line in f:
                # line is like "00001/0001"
                seq_folder_path = os.path.join(sequence_base_path, line.strip())
                # Each septuplet has 7 frames: im1.png ... im7.png
                frames_in_seq = sorted(glob.glob(os.path.join(seq_folder_path, "im*.png")))
                if len(frames_in_seq) >= self.k_frames:
                    vimeo_sequences.append(frames_in_seq) # Store list of full paths to frames
        return vimeo_sequences

    def _load_defencing_items(self):
        # Assumes structure: defencing_root_dir/Training Set/Training_Images and Training_Labels
        # Or similar for test set if is_train is False
        set_type = "Training Set" # TODO: Add Test Set if needed
        img_dir = os.path.join(self.defencing_root_dir, set_type, "Training_Images")
        mask_dir = os.path.join(self.defencing_root_dir, set_type, "Training_Labels")
        
        fence_items = []
        fence_img_names = sorted(os.listdir(img_dir))
        for name in fence_img_names:
            base_name, ext = os.path.splitext(name)
            # Try to find matching mask (could be .png, .jpg, etc.)
            # This logic is simplified from spynet.dataset_flow.py
            potential_mask_names = [base_name + ".png", base_name + ".jpg", base_name + ".jpeg", name]
            found_mask_path = None
            for m_name in potential_mask_names:
                p = os.path.join(mask_dir, m_name)
                if os.path.exists(p):
                    found_mask_path = p
                    break
            if found_mask_path:
                fence_items.append({'img': os.path.join(img_dir, name), 'mask': found_mask_path})
            else:
                print(f"Warning: Mask not found for fence image {name} in {mask_dir}")
        return fence_items

    def _augment_background_burst(self, bg_frames_pil):
        # bg_frames_pil: list of K PIL Images (clean backgrounds)
        # Apply consistent augmentation (crop, flip) across the burst.
        # Random homography can be more complex: either one for all, or evolving.
        # For now: shared random crop and flip.
        
        # 1. Shared Random Crop to target size (e.g., 320x192)
        # Get crop parameters from the first frame and apply to all
        if not bg_frames_pil: return []
        
        # Ensure all frames are at least target size before crop, can resize if necessary.
        # For now, assume they are larger or equal.
        i, j, th, tw = T.RandomCrop.get_params(bg_frames_pil[0], output_size=(self.img_height, self.img_width))
        cropped_frames = [TF.crop(frame, i, j, th, tw) for frame in bg_frames_pil]

        # 2. Shared Random Horizontal Flip
        if random.random() > 0.5:
            cropped_frames = [TF.hflip(frame) for frame in cropped_frames]
            
        # TODO: Add random homography if needed (consistent across burst or evolving)
        # TODO: Add color jitter etc. if specified by paper for background.
        return cropped_frames # List of K augmented PIL images

    def _augment_fence_for_burst(self, fence_img_pil, fence_mask_pil):
        # fence_img_pil, fence_mask_pil: PIL images for a single fence structure
        # Returns: list of K pairs: [(F'_1, M'_1_fence), ..., (F'_K, M'_K_fence)], each PIL
        
        augmented_fences = []
        
        # 1. Base Augmentations (once on the raw fence before K perspectives)
        #    (Random downsample, random window crop, color jitter, random Gaussian blur)
        #    This logic can be adapted from spynet.augmentations.augment_fence_structure
        #    For simplicity, let's apply some basic ones here:
        if random.random() < 0.5: # Color jitter
            fence_img_pil = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)(fence_img_pil)
        if random.random() < 0.3: # Gaussian blur
            kernel_s = random.choice([3, 5])
            fence_img_pil = TF.gaussian_blur(fence_img_pil, kernel_size=kernel_s, sigma=random.uniform(0.1, 1.0))

        # 2. K Random Perspective Distortions
        for _ in range(self.k_frames):
            # Apply a new perspective transform for each frame in the burst
            distorted_fence_img = self.perspective_distorter(fence_img_pil)
            # Ensure mask uses NEAREST interpolation if perspective is applied directly to it
            # For RandomPerspective, torchvision applies it to both if a list/tuple is passed.
            # However, mask should be NEAREST. A common practice is to get params and apply separately.
            # For now, let's assume RandomPerspective is smart or we apply it carefully.
            # It's safer to transform image then mask with same params but different interpolation.
            # To simplify, if RandomPerspective is applied to mask, it might blur it.
            # Let's assume we get params and apply: (this is a simplification)
            distorted_fence_mask = self.perspective_distorter(fence_mask_pil) # This might not be ideal for masks.
                                                                           # TODO: Ensure mask remains binary after perspective. 
                                                                           # TF.perspective for mask with INTERP_NEAREST
            
            # Resize to target image dimensions (IMG_HEIGHT, IMG_WIDTH)
            distorted_fence_img_resized = TF.resize(distorted_fence_img, [self.img_height, self.img_width],
                                                     interpolation=T.InterpolationMode.BILINEAR)
            distorted_fence_mask_resized = TF.resize(distorted_fence_mask, [self.img_height, self.img_width],
                                                      interpolation=T.InterpolationMode.NEAREST)
            augmented_fences.append((distorted_fence_img_resized, distorted_fence_mask_resized))
        return augmented_fences

    def __len__(self):
        return len(self.vimeo_sequences_paths) # Each sequence can produce one training item

    def __getitem__(self, idx):
        # 1. Load K clean background frames B_j from Vimeo.
        selected_vimeo_frame_paths = self.vimeo_sequences_paths[idx]
        # Randomly select K consecutive frames from the chosen septuplet (or other length sequence)
        start_frame_idx = random.randint(0, len(selected_vimeo_frame_paths) - self.k_frames)
        burst_frame_paths = selected_vimeo_frame_paths[start_frame_idx : start_frame_idx + self.k_frames]
        
        bg_frames_pil = [Image.open(p).convert('RGB') for p in burst_frame_paths]

        # Apply background augmentation (shared crop, flip for the burst)
        bg_frames_aug_pil = self._augment_background_burst(bg_frames_pil)
        # Convert to tensor: List of [3, H, W]
        B_j_tensors = [self.to_tensor(frame) for frame in bg_frames_aug_pil]

        # 2. Select keyframe B_k (e.g., middle frame).
        keyframe_idx_in_burst = self.k_frames // 2
        B_k_tensor = B_j_tensors[keyframe_idx_in_burst]

        # 3. Load fence F_raw, M_raw_fence from De-fencing dataset.
        selected_fence_item = random.choice(self.fence_items)
        F_raw_pil = Image.open(selected_fence_item['img']).convert('RGB')
        M_raw_fence_pil = Image.open(selected_fence_item['mask']).convert('L') # Grayscale mask

        # 4. Apply K random perspective distortions + fg_augmentation to get F'_j, M'_j_fence.
        # augmented_fences_pil is a list of K (F_pil, M_pil) tuples
        augmented_fences_pil = self._augment_fence_for_burst(F_raw_pil, M_raw_fence_pil)

        # 5. Create obstructed frames I_j and fence masks S_j.
        I_j_tensors = [] # List of obstructed frames [3,H,W]
        S_j_tensors = [] # List of fence masks [1,H,W], 1 for fence, 0 for bg

        for i in range(self.k_frames):
            F_prime_pil, M_prime_pil = augmented_fences_pil[i]
            B_j_i_tensor = B_j_tensors[i]

            F_prime_tensor = self.to_tensor(F_prime_pil) # [3,H,W]
            M_prime_tensor = (self.to_tensor(M_prime_pil) > 0.5).float() # [1,H,W], binarized
            
            I_j_i_tensor = M_prime_tensor * F_prime_tensor + (1 - M_prime_tensor) * B_j_i_tensor
            I_j_tensors.append(I_j_i_tensor)
            S_j_tensors.append(M_prime_tensor)

        I_k_tensor = I_j_tensors[keyframe_idx_in_burst]
        S_k_tensor = S_j_tensors[keyframe_idx_in_burst]

        # Prepare for SPyNetModified - move to device if model is on GPU
        # For now, assume CPU computation for spynet_m as it's loaded to CPU
        # Or move spynet_m to config.device in init if workers don't have separate GPUs
        # This part can be slow on CPU.
        current_spynet_device = next(self.spynet_m.parameters()).device

        # 8. Create I_k^m = I_k * (1 - S_k).
        I_k_m_tensor = I_k_tensor * (1 - S_k_tensor)

        all_features_for_f_in = [I_k_m_tensor, S_k_tensor]

        for j in range(self.k_frames):
            if j == keyframe_idx_in_burst:
                continue

            I_j_current_tensor = I_j_tensors[j]
            S_j_current_tensor = S_j_tensors[j]

            # 6. Prepare input for SPyNet^m: [I_k; S_k] and [I_j; S_j]
            # Ensure they are on the same device as spynet_m
            input_k_rgbm = torch.cat([I_k_tensor, S_k_tensor], dim=0).to(current_spynet_device)
            input_j_rgbm = torch.cat([I_j_current_tensor, S_j_current_tensor], dim=0).to(current_spynet_device)

            # 7. Compute flow: f_kj^m (output of spynet_m is [2,H,W])
            with torch.no_grad():
                # Add batch dimension for spynet_m, then remove
                f_kj_m_tensor = self.spynet_m(input_k_rgbm.unsqueeze(0), input_j_rgbm.unsqueeze(0)).squeeze(0)
            f_kj_m_tensor = f_kj_m_tensor.to(torch.device('cpu')) # Move flow back to CPU for subsequent ops if needed

            # 9. Warp non-keyframes:
            I_j_content_to_warp = I_j_current_tensor * (1 - S_j_current_tensor) # Mask out fence before warping content
            
            # Ensure inputs to warp_frame_with_flow are on the same device
            # And flow is on the same device as the frame to be warped by grid_sample
            # Let's assume I_j_content_to_warp and S_j_current_tensor are on CPU from ToTensor
            # f_kj_m_tensor is also moved to CPU above.

            # If warp_frame_with_flow uses spynet_backwarp, which uses grid_sample,
            # frame and flow need to be on same device for grid_sample.
            # The grid in spynet_backwarp is created on flow.device.
            # So, I_j_content_to_warp and S_j_current_tensor need to be on flow.device.
            target_warp_device = f_kj_m_tensor.device 
            
            Î_j_m_tensor = warp_frame_with_flow(I_j_content_to_warp.to(target_warp_device), f_kj_m_tensor) 
            Š_j_tensor = warp_frame_with_flow(S_j_current_tensor.to(target_warp_device), f_kj_m_tensor) 
            Š_j_tensor = (Š_j_tensor > 0.5).float() # Ensure warped mask is binary
            
            # V_j = ValidityMask(Š_j)
            # Create validity mask by warping a tensor of ones
            V_j_tensor = create_validity_mask(self.img_height, self.img_width, f_kj_m_tensor)
            V_j_tensor = V_j_tensor.to(torch.device('cpu')) # Move to CPU if other features are

            all_features_for_f_in.extend([
                Î_j_m_tensor.cpu(), 
                Š_j_tensor.cpu(), 
                V_j_tensor.cpu(), 
                f_kj_m_tensor.cpu()
            ])

        # 10. Concatenate: f_in
        try:
            f_in_tensor = torch.cat(all_features_for_f_in, dim=0)
        except Exception as e:
            print(f"Error concatenating features for f_in: {e}")
            for i, feat in enumerate(all_features_for_f_in):
                print(f"Feature {i} shape: {feat.shape}, device: {feat.device}")
            raise
            
        # Check if f_in_tensor has the expected number of channels
        expected_f_in_channels = 3 + 1 + (self.k_frames - 1) * (3 + 1 + 1 + 2)
        if f_in_tensor.shape[0] != expected_f_in_channels:
            print(f"WARNING: f_in_tensor has {f_in_tensor.shape[0]} channels, expected {expected_f_in_channels}")
            # This might happen if k_frames is not matching default or logic error

        return f_in_tensor, I_k_m_tensor, B_k_tensor # f_in, masked_keyframe_bg, clean_keyframe_bg

    # TODO: Add helper methods for loading data, augmentations, warping
    # def _load_vimeo_sequences(self): ...
    # def _load_defencing_items(self): ...
    # def _warp_frame(self, frame, flow): using torch.nn.functional.grid_sample


def main(config):
    print(f"Starting RDN Inpainting training on device: {config.device}")
    os.makedirs(config.outputs_dir, exist_ok=True)
    current_device = torch.device(config.device)

    # 1. Model
    print("Initializing RDNInpainting model...")
    model = RDNInpainting(
        num_input_channels=config.num_input_channels,
        num_output_channels=config.num_output_channels,
        num_features=config.num_features,
        growth_rate=config.growth_rate,
        num_blocks=config.num_blocks,
        num_layers=config.num_layers
    ).to(current_device)
    
    # TODO: Load pre-trained RDN weights if specified in config

    # 2. Optimizer and Loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon
    )
    criterion = nn.L1Loss()

    # 3. Dataset and DataLoader
    if not config.spynet_m_weights_path or not os.path.exists(config.spynet_m_weights_path):
        print(f"ERROR: Pre-trained SPyNetModified weights path not provided or invalid: {config.spynet_m_weights_path}")
        return

    print("Preparing InpaintingDataset and DataLoader...")
    # Pass the main config object to the dataset
    dataset_config_obj = argparse.Namespace(**vars(config)) # Convert args to a Namespace if main config is one
    
    train_dataset = InpaintingDataset(config=dataset_config_obj, 
                                      spynet_model_path=config.spynet_m_weights_path, 
                                      spynet_model_name_for_gt_flow_in_spynet_m=config.spynet_base_model_name, 
                                      is_train=True)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False,
        drop_last=True
    )

    # Move SPyNet-m in dataset to the main device if not handled by workers
    # This is tricky with DataLoader workers. If num_workers > 0, each worker 
    # has its own spynet_m instance. It should be on CPU by default from dataset init, 
    # and flow computation would be on CPU. 
    # If we want GPU flow computation in workers, dataset init would need device info.
    # For now, SPyNet-m is on CPU in dataset. Let's ensure its parameters are not part of RDN graph.
    if hasattr(train_dataset, 'spynet_m') and train_dataset.spynet_m is not None:
         train_dataset.spynet_m.to(torch.device('cpu')) # Ensure it stays on CPU if it was moved
         print(f"SPyNet-m in dataset explicitly set to CPU to avoid issues with workers/main device conflict.")


    # 4. Training Loop
    print(f"--- Starting training for {config.num_epochs} epochs ---")
    model.train()

    for epoch in range(config.start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, (f_in_batch, i_k_m_batch, b_k_batch) in enumerate(progress_bar):
            f_in_batch = f_in_batch.to(current_device)
            i_k_m_batch = i_k_m_batch.to(current_device)
            b_k_batch = b_k_batch.to(current_device)

            optimizer.zero_grad()
            residual_pred_batch = model(f_in_batch)
            b_k_pred_batch = i_k_m_batch + residual_pred_batch
            loss = criterion(b_k_pred_batch, b_k_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f'{running_loss / (batch_idx + 1):.4f}')

        epoch_loss = running_loss / len(train_dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(f"End of Epoch {epoch+1}. Average Loss: {epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")

        if (epoch + 1) % config.save_every_n_epochs == 0 or (epoch + 1) == config.num_epochs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_name = f"rdn_inpainting_epoch{epoch+1}_{timestamp}.pth"
            checkpoint_path = os.path.join(config.outputs_dir, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("--- Training finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RDN for Flow-Guided Inpainting")
    
    # Paths
    parser.add_argument('--vimeo_dir', type=str, default='data_raw/vimeo_septuplet/sequences', help='Path to Vimeo-90k dataset (sequences subfolder)')
    parser.add_argument('--defencing_dir', type=str, default='data_raw/De-fencing-master/dataset', help='Path to De-fencing dataset')
    parser.add_argument('--outputs_dir', type=str, default='./rdn_inpainting_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--spynet_m_weights_path', type=str, required=True, help='Path to pre-trained SPyNetModified weights (.pth)')
    parser.add_argument('--spynet_base_model_name', type=str, default='sintel-final', help='Base model name for SPyNetModified (e.g. sintel-final)')


    # RDN Architecture
    parser.add_argument('--num_features', type=int, default=DEFAULT_NUM_FEATURES, help='Number of features (G0)')
    parser.add_argument('--growth_rate', type=int, default=DEFAULT_GROWTH_RATE, help='Growth rate (G)')
    parser.add_argument('--num_blocks', type=int, default=DEFAULT_NUM_BLOCKS, help='Number of RDBs (D)')
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS, help='Number of conv layers in each RDB (C)')
    parser.add_argument('--num_input_channels', type=int, default=DEFAULT_NUM_INPUT_CHANNELS, help='Input channels for RDN (f_in)')
    parser.add_argument('--num_output_channels', type=int, default=DEFAULT_NUM_OUTPUT_CHANNELS, help='Output channels for RDN (residual)')

    # Training Params
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--beta1', type=float, default=DEFAULT_BETA1, help='Adam optimizer beta1')
    parser.add_argument('--beta2', type=float, default=DEFAULT_BETA2, help='Adam optimizer beta2')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON, help='Adam optimizer epsilon')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--k_frames', type=int, default=DEFAULT_K_FRAMES, help='Number of frames in a burst (K)')
    
    # Image/Patch size
    parser.add_argument('--img_width', type=int, default=DEFAULT_IMG_WIDTH)
    parser.add_argument('--img_height', type=int, default=DEFAULT_IMG_HEIGHT)

    # Checkpointing & Eval
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    # parser.add_argument('--eval_every_n_epochs', type=int, default=5)
    # parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training')

    # Misc
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (0 for main process)') # Default 0 for easier debugging
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()

    # --- Basic Test/Example ---
    # This won't run full training but will check if components can be initialized.
    print("--- Running Basic Setup Test for train_inpainting.py ---")
    
    # Create dummy SPyNetModified weights file for test if it doesn't exist
    # to allow InpaintingDataset to initialize without erroring on file not found.
    dummy_spynet_m_path = "dummy_spynet_m_weights.pth"
    if not os.path.exists(args.spynet_m_weights_path) and args.spynet_m_weights_path != dummy_spynet_m_path :
         print(f"WARNING: Provided SPyNet-m weights path '{args.spynet_m_weights_path}' does not exist.")
         print(f"For this basic test, attempting to use a dummy placeholder if SPyNetModified import succeeded.")
         if SPyNetModified is not None:
             if not os.path.exists(dummy_spynet_m_path):
                 print(f"Creating dummy SPyNetModified weights file at '{dummy_spynet_m_path}' for test purposes.")
                 # Save a dummy state_dict of a SPyNetModified instance
                 temp_spynet_m = SPyNetModified(model_name=args.spynet_base_model_name, pretrained=False)
                 torch.save(temp_spynet_m.state_dict(), dummy_spynet_m_path)
                 args.spynet_m_weights_path = dummy_spynet_m_path # Use dummy for test
             else:
                 args.spynet_m_weights_path = dummy_spynet_m_path
                 print(f"Using existing dummy SPyNetModified weights at '{dummy_spynet_m_path}' for test.")

    elif args.spynet_m_weights_path == dummy_spynet_m_path and not os.path.exists(dummy_spynet_m_path) and SPyNetModified is not None:
        print(f"Creating dummy SPyNetModified weights file at '{dummy_spynet_m_path}' for test purposes.")
        temp_spynet_m = SPyNetModified(model_name=args.spynet_base_model_name, pretrained=False)
        torch.save(temp_spynet_m.state_dict(), dummy_spynet_m_path)


    try:
        print(f"Attempting to initialize RDNInpainting model with test parameters...")
        test_model = RDNInpainting(
            num_input_channels=args.num_input_channels,
            num_output_channels=args.num_output_channels,
            num_features=args.num_features,
            growth_rate=args.growth_rate,
            num_blocks=args.num_blocks,
            num_layers=args.num_layers
        ).to(args.device)
        print("RDNInpainting model initialized successfully.")

        if os.path.exists(args.spynet_m_weights_path):
            print("Attempting to initialize InpaintingDataset (will use dummy data)...")
            # Pass a minimal config for dataset init test
            test_dataset_config = argparse.Namespace(
                num_input_channels=args.num_input_channels,
                num_output_channels=args.num_output_channels,
                img_height=args.img_height,
                img_width=args.img_width,
                k_frames=args.k_frames,
                device=args.device 
                # Add other paths if dataset requires them even for dummy run
            )
            test_dataset = InpaintingDataset(config=test_dataset_config, 
                                             spynet_model_path=args.spynet_m_weights_path,
                                             spynet_model_name_for_gt_flow_in_spynet_m=args.spynet_base_model_name,
                                             is_train=True)
            print(f"InpaintingDataset initialized. Length (placeholder): {len(test_dataset)}")
            
            # Test getting one item
            f_in, i_k_m, b_k = test_dataset[0]
            print(f"Dummy f_in shape: {f_in.shape}")
            print(f"Dummy i_k_m shape: {i_k_m.shape}")
            print(f"Dummy b_k shape: {b_k.shape}")
            assert f_in.shape == (args.num_input_channels, args.img_height, args.img_width)
            assert i_k_m.shape == (args.num_output_channels, args.img_height, args.img_width)
            assert b_k.shape == (args.num_output_channels, args.img_height, args.img_width)
            print("Dataset item shapes are correct.")

            print("--- Basic setup test passed. To run actual training, ensure all paths and data are correctly configured.")
            print("You will need to fill in the TODO sections in InpaintingDataset for actual data loading and processing.")
        else:
            print(f"Skipping InpaintingDataset test as SPyNetModified weights path '{args.spynet_m_weights_path}' not found or not SPyNetModified import failed.")
            if SPyNetModified is None:
                 print("SPyNetModified could not be imported. Dataset cannot be tested.")

    except ImportError as e:
        print(f"Import Error during test: {e}. Ensure all dependencies are met and modules are in PYTHONPATH.")
    except Exception as e:
        print(f"Error during basic setup test: {e}")
        import traceback
        traceback.print_exc()
    
    # To run the actual main training function:
    # main(args) # Uncomment and ensure all paths and data are correct.
    print("--- To run actual training: main(args)")
    print("Remember to replace placeholders in InpaintingDataset and provide valid data paths.") 