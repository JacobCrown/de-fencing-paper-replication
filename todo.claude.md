# Complete Guide to Replicate "Efficient Flow-Guided Multi-frame De-fencing"

## Overview

This paper presents a 3-stage pipeline for multi-frame de-fencing:

1. **Fence Segmentation** (Skip - you have this)
2. **Segmentation-aware Optical Flow Estimation**
3. **Flow-guided Multi-frame Fence Removal (Inpainting)**

Since you have segmentation, we'll focus on stages 2 and 3.

## Data Preparation

### 1. Synthetic Data Generation

**Purpose**: Create training data by overlaying fence masks on clean background scenes.

**Components needed**:

- **Background scenes**: Vimeo-90k dataset (clean video sequences)
- **Fence obstructions**: De-fencing dataset (fence images + binary masks)

**Process**:

1. **Load Vimeo-90k sequences** (7-frame sequences)
2. **Load De-fencing dataset** fence images and their ground truth masks
3. **Generate synthetic bursts** of K=5 frames by:
   - Taking a clean background sequence from Vimeo-90k
   - Selecting a fence image from De-fencing dataset
   - Applying K random perspective distortions to the fence to simulate motion
   - Overlaying the distorted fence on each background frame using: `I_i = S_i * F_i + (1 - S_i) * B_i`
   - Where S_i is the fence mask, F_i is the fence, B_i is the background

**Data Augmentation** (crucial for robustness):

- **Background augmentation**:
  - Random homography transformations
  - Center cropping to avoid black borders
  - Random cropping to 320×192 windows
  - Random horizontal flips
- **Fence augmentation**:
  - Random downsampling (different fence thickness/distance)
  - Random window cropping (focus on fence subregions)
  - Color jitter (different lighting conditions)
  - Random perspective distortions
  - Random Gaussian blur (defocus simulation)

### 2. Real Data Collection

**Purpose**: Create evaluation dataset with pseudo ground truth.

**Process** (as described in paper):

1. Capture ground truth frame by centering camera in fence cell
2. Fix camera exposure and focus on background
3. Move backwards and capture 5-frame burst with fence
4. Align keyframe to ground truth using SIFT+RANSAC homography
5. Extract 320×192 crops and filter based on SSIM/PSNR quality

## Stage 2: Segmentation-aware Optical Flow Estimation

### Architecture: Modified SPyNet (SPyNet^m)

**Purpose**: Compute optical flow maps that ignore fence obstructions and focus only on background motion.

**Key Innovation**: Standard optical flow networks fail with repetitive fence patterns. This modified version uses fence masks as additional input to focus on background-only flow.

### Implementation Details:

1. **Base Architecture**: SPyNet (lighter than PWC-Net)

   - Use publicly available SPyNet implementation
   - Initialize with pretrained weights

2. **Modification**:

   - **Input channels**: Modify first convolution layer to accept 6 channels instead of 6:
     - Original: [I_i, I_j] (2×3 RGB frames)
     - Modified: [I_i, S_i, I_j, S_j] (2×3 RGB frames + 2×1 fence masks)
   - **Architecture**: `f^m_ij = SPyNet^m([I_i; S_i], [I_j; S_j])`

3. **Training Process**:
   - **Input**: Synthetic fence-obstructed frame pairs + their fence masks
   - **Ground Truth**: Clean optical flow computed between corresponding background frames using vanilla SPyNet
   - **Loss Function**: L1 loss between predicted and ground truth flows
   ```
   L_f = (1/2N) * Σ|SPyNet(B_i, B_j)|_x - f^m_ij|_x|
   ```
   - **Training**: Fine-tune the modified SPyNet for 1000 epochs

**Training Data Generation**:

- Create pairs of synthetic obstructed frames
- Compute ground truth flows using vanilla SPyNet on corresponding clean backgrounds
- Train the modified network to predict these clean flows despite obstructions

## Stage 3: Flow-guided Multi-frame Fence Removal

### Architecture: Residual Dense Network (RDN) for Inpainting

**Purpose**: Use aligned frames to inpaint missing regions in keyframe.

### Process Flow:

1. **Input Preparation**:

   - Obstructed frame sequence: {I_i}
   - Fence masks: {S_i} (from your segmentation model)
   - Optical flows: {f^m_ki} (from Stage 2, keyframe to each other frame)

2. **Frame Masking**:

   - Mask out fence regions: `I^m_i = I_i ⊙ (1 - S_i)` (multiply by inverse of fence mask)

3. **Frame Alignment**:

   - Warp each frame to keyframe: `Ĩ^m_i = W(I^m_i, f^m_ki)`
   - Warp corresponding masks: `S̃_i = W(S_i, f^m_ki)`
   - Compute validity masks: `V_i` (binary masks marking valid warped regions)

4. **Feature Concatenation**:

   - Concatenate all information: `f_in = [{Ĩ^m_i}; {S̃_i}; {V_i}]`
   - This creates a multi-channel input with aligned frames, masks, and validity info

5. **Inpainting with RDN**:
   - Pass concatenated features through Residual Dense Network
   - **Skip Connection**: Add masked keyframe to RDN output
   - **Final Output**: `B̃_k = I^m_k + RDN(f_in)`

### Training Details:

**Loss Function**: L1 loss between output and clean background keyframe

```
L_in = (1/N) * Σ|B_k|_x - (I^m_k + RDN(f_in))|_x|
```

**Training Data**: Use synthetic bursts with known clean backgrounds as ground truth

## Implementation Specifications

### Network Architectures:

1. **SPyNet^m** (Segmentation-aware optical flow):

   - Base: SPyNet with pretrained weights
   - Modification: First conv layer 6→8 channels (RGB+mask for each frame)
   - Training: Fine-tune on synthetic data for 1000 epochs

2. **RDN** (Residual Dense Network):
   - Use standard RDN architecture for inpainting
   - Input: Multi-channel concatenated features
   - Output: Residual to be added to masked keyframe

### Training Parameters:

- **Learning rate**: 1e-4
- **Weight decay**: 4e-5
- **Optimizer**: Adam (α=1e-4, β1=0.9, β2=0.999, ε=1e-8)
- **Epochs**: 1000 for all models
- **Input size**: 320×192 pixels
- **Burst size**: K=5 frames

### Training Strategy:

- **Independent training**: Train SPyNet^m and RDN separately (not end-to-end)
- **Data**: Use synthetic bursts for training both networks
- **Validation**: Hold-out synthetic test set + real burst evaluation

## Evaluation Metrics

### Quantitative Metrics:

- **SSIM** (Structural Similarity Index)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **LPIPS** (Learned Perceptual Image Patch Similarity with VGG-16)

### Evaluation Regions:

- **Inside fence mask**: Most important (measures inpainting quality)
- **Outside fence mask**: Should remain unchanged
- **Total image**: Overall quality

### Datasets for Evaluation:

1. **Synthetic test set**: 100 bursts from Vimeo-90k + De-fencing
2. **Real bursts**: Your collected dataset with pseudo ground truth

## Key Implementation Notes

### Critical Success Factors:

1. **Proper synthetic data generation**: Quality of synthetic training data is crucial
2. **Flow network conditioning**: SPyNet^m must learn to ignore fence patterns effectively
3. **Frame alignment accuracy**: Poor flow estimation will ruin inpainting quality
4. **Mask quality**: Since you're skipping segmentation training, ensure your existing masks are accurate

### Common Pitfalls to Avoid:

1. **Flow network failing on repetitive patterns**: Standard networks struggle with fence patterns
2. **Insufficient data augmentation**: Limited fence variety in De-fencing dataset requires heavy augmentation
3. **Training instability**: Train components separately, not end-to-end
4. **Runtime optimization**: Use efficient implementations (SPyNet vs PWC-Net)

## Expected Performance

**Target metrics** (from paper):

- **Synthetic data**: ~34 dB PSNR inside fence mask, ~0.95 SSIM
- **Real data**: ~29 dB PSNR inside fence mask, ~0.87 SSIM
- **Runtime**: ~7 FPS for 320×192 5-frame bursts on GTX 1080 Ti

## Next Steps After Implementation

1. **Validate on paper's datasets**: Ensure you match their reported performance
2. **Test with your segmentation model**: Replace their U-Net with your model
3. **Apply to your fence dataset**: Use trained networks on your walking-around-fence data
4. **Fine-tune if needed**: May need domain adaptation for your specific fence types

This approach prioritizes efficiency and practicality while maintaining high quality results through the modular pipeline design.
