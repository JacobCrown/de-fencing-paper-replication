# TODO: Replicating "Efficient Flow-Guided Multi-frame De-fencing"

## A. Prerequisites & Data Setup

- [ ] **Acquire Datasets:**
  - [ ] Download Vimeo-90k dataset (specifically `test_clean.zip` from `http://data.csail.mit.edu/tofu/testset/` for backgrounds).
  - [ ] Download De-fencing dataset (for fence images and ground truth masks).
- [ ] **Setup Software Environment:**
  - [ ] Install PyTorch.
  - [ ] Obtain or implement SPyNet (e.g., from `https://github.com/sniklaus/pytorch-spynet`).
  - [ ] Obtain or implement RDN (Residual Dense Network) (e.g., from `https://github.com/yjn870/RDN-pytorch`).
- [ ] **Prepare Your Fence Segmentation Model:**
  - [ ] Ensure your existing segmentation model can output soft fence masks (0-1, 1 for fence) for individual frames. Define this as `S_i` for frame `I_i`.

## B. Step 1: Train Segmentation-Aware Optical Flow Network (SPyNet^m)

_(Corresponds to Paper's Section 3.2)_

- [ ] **Modify SPyNet Architecture (Create SPyNet^m):**
  - [ ] Start with a standard SPyNet.
  - [ ] Modify its first convolutional layer to accept concatenated input: `[Frame_RGB; Frame_Mask]`.
    - If Frame_RGB is 3 channels and Frame_Mask is 1 channel, then each of the two inputs to the first layer is 4 channels. The layer will expect 8 input channels in total.
  - [ ] Initialize the modified first layer's weights randomly (or Xavier/He).
  - [ ] Initialize the rest of SPyNet^m with pre-trained SPyNet weights.
- [ ] **Implement Synthetic Data Generation for SPyNet^m Training:**
  - [ ] **Background Pair Processing (from Vimeo-90k):**
    - [ ] Load two consecutive clean frames (B_i, B_j).
    - [ ] Implement "Background Augmentation" (Paper's Appendix A.1):
      - [ ] Random homography.
      - [ ] Center crop (post-homography).
      - [ ] Random 320x192 crop.
      - [ ] Random horizontal flip.
            (Result: B'\_i, B'\_j)
  - [ ] **Fence Structure Processing (from De-fencing dataset):**
    - [ ] Load a fence image (F_raw) and its binary mask (M_raw_fence).
    - [ ] Implement "Foreground Augmentation" (Paper's Appendix A.1 - _excluding K perspective distortions for this step_):
      - [ ] Random downsample (fence & mask).
      - [ ] Random "outer" window crop (fence & mask).
      - [ ] Color jitter (on fence image only).
      - [ ] Random Gaussian blur (on fence image only).
            (Result: F'\_aug, M'\_aug_fence)
  - [ ] **Create Obstructed Frame Pair:**
    - [ ] Overlay _the same_ F'\_aug onto _both_ B'\_i and B'\_j using M'\_aug_fence.
      - I_obstructed_i = M'\_aug_fence _ F'\_aug + (1 - M'\_aug_fence) _ B'\_i
      - I_obstructed_j = M'\_aug_fence _ F'\_aug + (1 - M'\_aug_fence) _ B'\_j
    - [ ] The input mask for SPyNet^m for both frames will be M'\_aug_fence.
  - [ ] **Generate Ground Truth Flow:**
    - [ ] Compute optical flow between the _clean augmented background frames_ B'\_i and B'\_j using a _standard, pre-trained vanilla SPyNet_. This is `flow_gt(B'_i, B'_j)`.
- [ ] **Train SPyNet^m:**
  - [ ] **Input:** ([I_obstructed_i; M'_aug_fence], [I_obstructed_j; M'_aug_fence]).
  - [ ] **Output:** Predicted flow `flow_pred`.
  - [ ] **Loss Function:** L1 loss: `sum(|flow_gt(B'_i, B'_j) - flow_pred|)`.
  - [ ] **Optimizer:** ADAM (lr=1e-4, weight decay=4e-5).
  - [ ] **Training Duration:** ~1000 epochs or until convergence.

## C. Step 2: Train Flow-Guided Inpainting Network (RDN)

_(Corresponds to Paper's Section 3.3)_

- [ ] **Prepare RDN Architecture:**
  - [ ] Use a standard RDN.
  - [ ] Implement the skip connection: RDN output is a residual, added to the masked keyframe.
  - [ ] Determine RDN input channel count based on K frames (e.g., K=5):
        `K * (num_channels_per_warped_masked_frame + num_channels_per_warped_mask + 1_for_validity_mask)`.
        Plus channels for the keyframe's own masked frame and mask. Example: `(3+1+1) for keyframe + (K-1)*(3+1+1)` if RGB.
- [ ] **Implement Input Preparation for RDN Training (within training loop):**
  - [ ] **Generate Synthetic Burst (K frames):**
    - [ ] Load K clean background frames {B_1, ..., B_K} from Vimeo-90k (apply background augmentation).
    - [ ] Select a keyframe B_k (e.g., middle frame). This is the RDN's ground truth target.
    - [ ] Load a fence image (F_raw) and its mask (M_raw_fence) from De-fencing.
    - [ ] Apply "Foreground Augmentation" (Appendix A.1), _including K random perspective distortions_ to create {F'\_1, ..., F'\_K} and {M'\_1_fence, ..., M'\_K_fence}.
    - [ ] Create K obstructed frames: I_j = M'\_j_fence _ F'\_j + (1 - M'\_j_fence) _ B_j.
  - [ ] **Get Fence Masks {S_j}:** These are your M'\_j_fence.
  - [ ] **Create Masked Frames {I_j^m}:** Zero out fence regions in I_j using S_j_binarized.
        `I_j^m = I_j * (1 - S_j_binarized)`.
  - [ ] **Get Flows {f_kj^m}:** For each non-keyframe I_j, compute flow to keyframe I_k using your _trained SPyNet^m_:
        `f_kj^m = SPyNet^m([I_k; S_k], [I_j; S_j])`.
  - [ ] **Warp Frames and Masks to Keyframe I_k:**
    - For each non-keyframe j:
      - [ ] Î_j^m = Warp(I_j^m, f_kj^m)
      - [ ] Š_j = Warp(S_j, f_kj^m) (S_j is the fence mask M'\_j_fence)
      - [ ] V_j = Validity mask from warping.
  - [ ] **Concatenate for RDN input `fin`:**
        `fin = [I_k^m, S_k, V_k (all 1s), Î_1^m, Š_1, V_1, ..., Î_K^m, Š_K, V_K]` (excluding keyframe j from warped list, ensure correct channel order and count).
- [ ] **Train RDN:**
  - [ ] **Input:** `fin`.
  - [ ] **Output:** Predicted residual `Res_pred = RDN(fin)`.
  - [ ] **Final Reconstruction (for loss):** `B_k_pred = I_k^m + Res_pred`.
  - [ ] **Ground Truth:** Clean background keyframe B_k.
  - [ ] **Loss Function:** L1 loss: `sum(|B_k - B_k_pred|)`.
  - [ ] **Optimizer:** ADAM (same parameters as SPyNet^m).
  - [ ] **Training Duration:** ~1000 epochs or until convergence.

## D. Step 3: Inference (De-fencing Your Data)

- [ ] **Input:** K-frame burst {I_1, ..., I_K} from your camera. Select keyframe I_k.
- [ ] **Fence Segmentation:** For each I_j, get fence mask S_j from your segmentation model.
- [ ] **Mask out fences:** Create I_j^m for all j.
- [ ] **Optical Flow:** For each non-keyframe I_j, compute `f_kj^m` using trained SPyNet^m.
- [ ] **Warping:** Generate Î_j^m, Š_j, V_j for all non-keyframes.
- [ ] **Prepare RDN Input `fin`:** Concatenate I_k^m, S_k, and all warped components.
- [ ] **Inpainting:**
  - [ ] `Res_pred = RDN(fin)`
  - [ ] `B_k_reconstructed = I_k^m + Res_pred`
- [ ] **Output:** `B_k_reconstructed` (de-fenced keyframe).

## E. Key Considerations & Notes

- [ ] **Clarify Mask Interpretation for `I_j^m`:** Confirm that `I_j^m` means fence regions are zeroed out (`I_j * (1 - S_j_binarized)`), and `Š_j` (warped `S_j`) are the _fence_ masks indicating where RDN should inpaint. (Figure 4 caption: "mask out (0) the occluded areas").
- [ ] **Implement Differentiable Warping:** Use `torch.nn.functional.grid_sample` or similar.
- [ ] **Maintain Consistent Resolution:** (e.g., 320x192) throughout.
- [ ] **Review Data Augmentation Details:** Carefully follow Appendix A.1 in the paper for both background and foreground augmentations.
- [ ] **Test Each Module Independently:** After training, verify SPyNet^m produces reasonable background flows on synthetic data, and RDN inpaints plausibly given correct inputs.
