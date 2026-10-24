# Method description

Team: tinyreef-praneel   Track(s): 1   Reported testing-phase submission(s): 900002

## Approach (one page at most)
TinyReefNet (0.052 M parameters) is a joint restoration + segmentation network derived from the EURS starter-kit
baseline: one shared depthwise-separable encoder (4 stages, channels 16/32/64/128, 16x downsampling) and two decoders with
U-Net-style skip connections. The restoration decoder ends in a 1x1 convolution + sigmoid and outputs an RGB image in
[0, 1]; the segmentation decoder outputs 5-class logits (0 background, 1 coral, 2 fish, 3 elasmobranch, 4 other), and the
mask is the argmax. Both come from a single forward pass at 256 x 256.
Loss: restoration = 0.8 L1 + 0.2 SSIM-like; segmentation = 0.5 weighted cross-entropy + 0.5 Dice.
Optimiser: AdamW, one-cycle learning rate 3e-4, batch size 16, random horizontal flip.

## Training data
Trained from scratch on the EURS development training scenes (degraded/ground-truth image + ground-truth mask pairs,
at most 60 degraded images per scene, resized to 256 x 256). No separate validation split: the checkpoint
`tinyreef_best.pth` is the epoch with the lowest training loss (epoch 100 of the run).
External data and pretrained models: None.

## Anything non-standard at inference
None. Input is resized to 256 x 256 (bilinear); outputs are resized back to the scoring size (longest side at most 1024 px,
aspect kept, downscale only): restored image bilinear, mask nearest-neighbour. No test-time augmentation, no ensembling, no
post-processing. The restored image is quantised with round(x * 255) (not truncated).

## Reproduction notes
`run.py` is the same code path as `infer.py` of our research repository (which produced submission 900002), reusing the
model from `src/tinyreef/`. Needs nothing beyond the base image (requirements.txt is empty). Measured on 6 images of 1024 x 768 px on an
RTX 3090: about 1.3 s in total after start-up (CPU about the same), peak GPU memory about 16 MB for the model and activations. The checkpoint is `weights/tinyreef_best.pth`
(SHA-256 6ed616a0f15a45065a2934aaa6ad525f8c2cc80c4e41943badb26964b725c65f), a dict {"model", "epoch", "cfg"}.
