# Comprehensive Technical Solutions & Guidance: Dry Vegetation CCTV Detection Project

---

## 1. Initial Architecture & Feasibility Analysis

### Direct Architecture Recommendation
At **1–2 FPS** on a **fixed CCTV view** with a **strict less than 2 GB VRAM cap** and **scarce training data**, the highest-reliability solution (greater than 70% mask precision/IoU after light adaptation) is a **two-stage hybrid pipeline** rather than a single monolithic deep model:

1. **Stage 1 — Static-Scene Spatial & Spectral Prior (Zero VRAM, CPU/OpenCV):** Because the CCTV camera is fixed, compute a **static background / persistent vegetation anchor mask** and apply **color-space & texture cues** (e.g., ExG / NGRDI-like RGB ratios, HSV saturation/value decay, and local high-frequency roughness) restricted to outdoor/ground regions.
2. **Stage 2 — Quantized Hybrid Segmentation Model (FP16/INT8 ONNX Runtime or TensorRT, ~150–450 MB VRAM):** Use a lightweight hierarchical encoder-decoder such as **SegFormer-B0** (MiT-B0 backbone + lightweight All-MLP decoder) or a **MobileNetV3 / YOLO11-seg-nano** backbone paired with a **custom lightweight decoder head** (or SAM 2 / mask-refinement head fed by fixed spatial prompts on static frames).
3. **Stage 3 — Temporal Consensus & Hysteresis Gate (1–2 FPS):** Require multi-frame persistence across consecutive 0.5–1.0 s ticks before toggling mask state, eliminating single-frame lighting flicker or passing shadows.

With a 1–2 FPS ceiling, latency per frame can comfortably be 100–300 ms, which lets you trade peak throughput for **higher resolution crops/tiles** (crucial for dry grass/twigs/foliage boundaries) while staying well under **2 GB VRAM**—provided you avoid loading heavy PyTorch training graphs, large foundation backbones in FP32, or unbatched multi-stream buffers into GPU memory at inference time.

---

### Existing Solutions, Research & Industry Patterns

#### Existing Approaches & Why They Fail Without Tuning
* **Pure Color / HSV / ExG Thresholding:** Fast and zero-VRAM, but dry vegetation (straw, dead grass, dry leaves, dormant shrubs) shares hue/luminance ranges with dry soil, sand, concrete dust, cardboard, and sunlit paving. Fails reliability targets (>70%) under variable sun/shadow or monsoon-to-dry transitions.
* **Generic Pretrained Semantic Models (ADE20K / Cityscapes):** Standard segmentation checkpoints distinguish broad `vegetation`, `tree`, `grass`, or `terrain` classes, not *"dry/senescent vs. green/live vegetation"*. Without domain adaptation, dry brush is frequently misclassified as `ground`/`dirt` or missed entirely.
* **Heavy Foundation / Video Models (SAM 2 Large, Mask2Former, ViT-Backbone U-Nets):** High zero-shot/prompted mask quality, but heavy encoders and video memory banks easily push VRAM past 2 GB if run naively at high resolution or kept resident alongside other services.

#### Research & Commercial Directions
* **Green vs. Senescent / Dry Vegetation Split:** Agricultural and ecological segmentation research (such as *SegVeg* and UAV/edge crop-weed/senescence studies) shows that separating **vegetation vs. non-vegetation** first and then classifying **senescent/dry vs. green** (using structural/contextual features rather than raw RGB alone) reduces confusion with bare soil.
* **Lightweight Vision-CNN Hybrids & Edge Transformers:** Architectures combining hierarchical/depthwise convolutions with lightweight attention or compact MLP decoders (e.g., SegFormer-B0, TinySegformer, mobile U-Net/DeepLab variants) offer strong boundary fidelity at low parameter counts.
* **Industrial & Wildfire / Perimeter Vendors:** Commercial perimeter, utility corridor, and wildfire/brush-hazard systems typically pair **fixed-camera ROI masking + change/motion or spectral-texture heuristics** with **quantized edge/server segmentation models** and **alert persistence rules** rather than relying on raw single-frame neural masks alone.

---

### Hybrid Model Design (Few-Shot + <2 GB VRAM)

| Component / Stage | Role & Design | VRAM / Compute Strategy |
| :--- | :--- | :--- |
| **Frozen Encoder** | Pretrained `MiT-B0` or `MobileNetV3-Large` | Freeze early/mid stages; fine-tune top stage only |
| **Custom Decoder** | Lightweight FPN / All-MLP + shallow conv head | Outputs binary/3-class mask (dry veg, live veg, bg) |
| **Quantization** | Post-training **FP16** or calibrated **INT8** | Keeps weights + activations tiny (<500 MB) |
| **Post-Processing** | ROI lock, morphological cleanup, 1–2 FPS voting | CPU/OpenCV; cuts false positives on static view |

#### Practical Layer-Freezing & Few-Shot Recipe
1. **Start from Pretrained Weights:** Initialize the encoder from ImageNet/ADE20K/Cityscapes weights so low-level edge, leaf-blade, and branch textures already exist.
2. **Freeze 70–85% of Backbone:** Freeze lower feature blocks (low-level edges/color). Unfreeze only the **deepest encoder block** plus your **custom decoder/segmentation head** so scarce labeled frames adapt semantic dry-foliage cues without catastrophic overfitting.
3. **Few-Shot Data Augmentation for Fixed CCTV:** Even 30–100 annotated frames/patches from your camera view can be multiplied safely using **random brightness/contrast/gamma jitter** (simulating sun/cloud/dusk), **horizontal flips** (when no strict directional road/sign constraint breaks), **cutout/coarse dropout**, and **synthetic dry-color/texture shifts** on existing vegetation masks—while avoiding heavy geometric warps that distort a fixed camera's ground plane.
4. **Quantize & Export:** Export to **ONNX** and compile to **TensorRT (FP16 or INT8 with a small calibration set of ~50–200 representative frames)** or ONNX Runtime CUDA with strict memory limits. Keep **batch size = 1** (since 1–2 FPS is low-rate streaming) and pre-allocate inference buffers.

---

### Implementation Blueprint & Verification

* **Isolate Inference VRAM:** Run segmentation in its own dedicated process/container with explicit memory caps (e.g., TensorRT workspace cap or ONNX Runtime `arena_extend_strategy="kSameAsRequested"`), and never leave PyTorch autograd/optimizer states loaded during live 1–2 FPS inference.
* **Tile High-Res CCTV Feeds:** If a 1080p/4K fixed camera has thin dry grass or distant brush, run a fast full-frame low-res pass (`512x512` or `640x640`) to locate candidate regions, then run high-res crops **only** on active candidate tiles—keeping peak VRAM low while preserving fine mask edges.
* **Verify Against Your >70% Target:** Benchmark **IoU / Dice** and **false-alarm rate per hour** across morning, midday glare, late afternoon, and overcast/windy clips from the actual CCTV feed, tuning the **temporal persistence window** (e.g., 2 out of 3 frames at 1–2 FPS) to hit your reliability bar.

---

## 2. High-Level Human-Readable Explanation & Analogy

To detect dry vegetation reliably (more than 70% accuracy) using a fixed CCTV camera without running out of memory (staying under **2 GB VRAM**) or needing thousands of training images, the best approach is a **3-step smart assembly line** rather than one giant AI model.

Think of it like a security guard with a magnifying glass: instead of scanning every inch of a huge wall every millisecond, they ignore the brick wall they already know is there, quickly glance at the bushes, and only look closely where something looks dry.

---

### Why a Single AI Model Fails Here
* **Simple Color Filters fail:** If you just tell a computer "highlight anything brown or yellow," it will accidentally highlight dry mud, wooden fences, cardboard boxes, and concrete in bright sunlight.
* **Out-of-the-box AI models fail:** Standard AI models know what a "tree" or "grass" is, but they aren't trained to tell the difference between *healthy green grass* and *dead, dry fire-hazard brush*.
* **Giant AI models crash your GPU:** Massive models (like standard SAM or large Vision Transformers) eat up 4 GB to 12 GB of VRAM instantly, blowing past your 2 GB limit.

---

### The 3-Step Hybrid Solution

#### Step 1: The "Ignore the Concrete" Filter (Zero VRAM)
* **Keywords:** *Static Scene Prior, ROI (Region of Interest) Masking, Color & Texture Heuristics.*
* **What it is:** Since your CCTV camera never moves, the roads, buildings, and walls stay in the exact same spot every day. Before the AI even touches the video, we draw a digital boundary (an **ROI mask**) telling the system to completely ignore roads, sky, and buildings, and only look at ground/plant zones. We also use lightweight math formulas on the image colors (like **ExG — Excess Green Index** and **HSV color space**) to spot areas losing greenness.
* **Why we use it:** It runs purely on your regular CPU (using **0 MB of GPU VRAM**). By throwing away 70% of the useless background right away, the AI only has to focus on the areas that actually matter.

#### Step 2: The "Transfer-Learned" Lightweight AI (Under 500 MB VRAM)
* **Keywords:** *Encoder-Decoder Architecture (SegFormer-B0 or MobileNetV3), Layer Freezing, Fine-Tuning, Quantization (FP16/INT8).*
* **What it is:** Instead of teaching a baby AI from scratch, we take a small, fast pre-trained model like **SegFormer-B0** (which already understands shapes, leaves, and edges from millions of internet photos) and customize it:
  1. **Encoder (The Eyes):** Takes the image and breaks it down into basic shapes like edges, twigs, and leaf textures.
  2. **Layer Freezing:** We "lock" (freeze) 80% of the model's brain so it doesn't forget how to recognize basic shapes when we train it on your small dataset.
  3. **Custom Decoder (The Decision Maker):** We attach a tiny, lightweight head to the end and train *only* that part using just **50 to 100 labeled photos** from your camera. This teaches it: *"This specific crusty, yellow texture is dry vegetation."*
  4. **Quantization (Shrinking the Model):** Normally, AI does math using bulky 32-bit decimal numbers (**FP32**). We compress those numbers down to 16-bit (**FP16**) or 8-bit integers (**INT8**) using **ONNX Runtime** or **TensorRT**.
* **Why we use it:**
  * **Freezing** prevents the AI from memorizing your few photos and failing when the weather changes (overfitting).
  * **Quantization** shrinks the model's memory footprint by up to **75%** with almost zero drop in accuracy—keeping total VRAM usage around **200 MB to 500 MB** (way below your 2 GB ceiling).

#### Step 3: The "Double-Check" Timer (1–2 FPS Advantage)
* **Keywords:** *Temporal Consensus, Hysteresis Gate.*
* **What it is:** Because your camera only needs to check **1 to 2 frames per second (FPS)**, we don't trust a single frame on its own. We set a rule called **Temporal Consensus**: a patch of grass must be flagged as "dry" in **at least 2 out of 3 consecutive frames** before the system officially draws the warning mask.
* **Why we use it:** If a cloud passes over the sun, a car headlight flashes, or a shadow moves across the grass for half a second, a single-frame AI might panic and draw a false mask. Waiting 1.5 seconds to confirm eliminates flickering and easily pushes your real-world reliability above **70%**.

---

### How It Fits Hardware & Constraints

#### 1. Memory Budget (Staying Under 2 GB VRAM)
* **Smart Tiling:** If you feed a massive 4K image into an AI all at once, VRAM spikes. Instead, we scan a shrunk-down version (`512x512`) first, or slice the vegetation zone into small tiles, processing them **one at a time (`Batch Size = 1`)**.
* **Clean Runtime:** During live operation, we strip out all heavy training tools (like PyTorch gradients) and run only the slimmed-down inference engine, keeping peak GPU memory safely locked well below **1 GB**.

#### 2. Why 1–2 FPS Is Actually a Superpower Here
Most CCTV projects struggle because they have to process 30 frames every second. Because dry vegetation takes days or weeks to dry out—not milliseconds—running at **1 to 2 FPS** gives your GPU a massive 500ms to 1000ms breathing room per frame. This allows you to run higher-quality edge cleanup (smoothing the mask outline neatly around bushes) without ever overloading your graphics card.

---

## 3. Commercial Licensing Audit & Safe Model Options

**SegFormer’s official NVIDIA weights (`nvidia/mit-b0` through `b5`) are under the NVIDIA Source Code License, which strictly prohibits commercial use**—and you cannot legally "bypass" a non-commercial weight license just by fine-tuning or quantizing it.

Instead, you can swap the backbone for **100% commercially permissive (Apache 2.0 / BSD-3 / MIT) models** while keeping the exact same 3-step hybrid architecture and staying well under **2 GB VRAM**.

---

### Safe, Commercially Unrestricted Backbones

| Model / Backbone | License | VRAM (FP16/INT8) | Why It Fits |
| :--- | :--- | :--- | :--- |
| **MobileNetV3-Large + LR-ASPP** | **BSD-3-Clause** (PyTorch/Torchvision) | ~100–250 MB | Built into `torchvision`, explicitly commercial-safe, fast on edge/server |
| **DINOv2-Small (`vit_small_patch14`)** | **Apache 2.0** (Meta AI) | ~250–450 MB | Unmatched texture & fine-detail understanding from tiny labeled datasets |
| **ConvNeXt-Tiny / EfficientNet-B0** | **MIT / Apache 2.0** (`timm` / PyTorch) | ~150–350 MB | Strong multi-scale feature extraction when paired with a lightweight U-Net/FPN decoder |

*(Note: Even when using permissive backbones like ImageNet-pretrained MobileNetV3 or DINOv2, avoid weights pretrained on non-commercial datasets like Cityscapes—stick to **ImageNet-1K** or **LVD-142M** weights, which carry clean commercial terms.)*

---

### How to Build It With Zero Licensing Risk

#### Option A: The Fastest & Safest Commercial Stack (`MobileNetV3-Large` + `LR-ASPP`)
* **What it is:** `torchvision.models.segmentation.lraspp_mobilenet_v3_large` (or a `timm` MobileNetV3 encoder + lightweight U-Net/FPN decoder) initialized with **ImageNet-1K** weights (BSD-3-Clause).
* **How we adapt it:**
  1. **Freeze** the first 75% of the MobileNetV3 layers (which already detect edges, grass blades, and branches).
  2. **Replace & Train** only the final **LR-ASPP (Lite Reduced Atrous Spatial Pyramid Pooling)** decoder head on your 50–100 labeled CCTV frames.
  3. **Export & Quantize** to **ONNX / TensorRT FP16** (~150 MB VRAM total).
* **Why it works:** Zero third-party restrictive licenses, native PyTorch support, and tiny memory footprint.

#### Option B: The Best Few-Shot Accuracy (`DINOv2-Small` Frozen Encoder + Custom Conv Decoder)
* **What it is:** Meta’s **DINOv2 ViT-Small** (21M parameters, **Apache 2.0 license**) paired with a tiny 3-layer convolutional decoder that you write yourself from scratch.
* **How we adapt it:**
  1. **Freeze 100% of DINOv2-Small:** DINOv2 was trained via self-supervised learning on 142M images, meaning it already separates dry straw, dead leaves, green foliage, and concrete at a pixel-patch level without needing backbone fine-tuning.
  2. **Train Only Your Tiny Decoder (~500 KB of weights):** Feed your 50–100 labeled CCTV frames through the frozen DINOv2 encoder and train only your lightweight upsampling decoder to output the dry-vegetation mask.
  3. **Quantize to FP16:** Runs in **~300–450 MB VRAM** at 1–2 FPS.
* **Why it works:** Because you freeze the entire Apache-2.0 encoder and write the tiny decoder yourself, your commercial IP is 100% clean, and DINOv2’s self-supervised features generalize across lighting changes far better than standard CNNs when data is scarce.

---

### Complete Commercial-Safe Pipeline Summary

1. **CPU Static ROI + Color Prior (0 MB VRAM):** Mask out non-ground/building zones on the fixed CCTV frame and pre-filter using **ExG (Excess Green)** and **HSV saturation** heuristics.
2. **Permissive Hybrid Model (~150–450 MB VRAM):** Run **Frozen DINOv2-Small (Apache 2.0)** or **Partially Frozen MobileNetV3-Large (BSD-3)** + your custom lightweight decoder in **FP16 ONNX/TensorRT** at `Batch Size = 1`.
3. **1–2 FPS Temporal Voting (0 MB VRAM):** Confirm dry-vegetation masks over 2–3 consecutive frames before rendering/alerting to push reliability past **70%**.

---

## 4. Deep Technical Breakdown & Code Implementations

Below is the complete practical breakdown of **Approach A** and **Approach B**—where to get the models, how layer freezing works in code, how to build and export them, and the exact manual work required to reach greater than 70% mask reliability under **2 GB VRAM** at **1–2 FPS**.

---

### Approach A: MobileNetV3-Large + LR-ASPP (Lightweight CNN + Custom Head)

#### 1. Where to Get the Model
* **Source:** Built directly into `torchvision.models.segmentation` (`lraspp_mobilenet_v3_large`) under a **BSD-3-Clause** license.
* **How to load:**
  ```python
  import torchvision.models.segmentation as seg_models

  # Load model with ImageNet-1K pretrained backbone (commercial-friendly)
  model = seg_models.lraspp_mobilenet_v3_large(
      weights=None,
      weights_backbone=seg_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
      num_classes=2,  # 0 = background/non-dry, 1 = dry vegetation
  )
  ```
* *Licensing note:* Avoid `weights="DEFAULT"` / `COCO_WITH_VOC_LABELS_V1` if you want to avoid COCO/VOC dataset terms; starting from `weights_backbone="IMAGENET1K_V1"` with `num_classes=2` gives you a clean `LRASPP` head for binary dry-vegetation segmentation.

#### 2. How to Freeze Layers & What to Train
In `LRASPP`, `model.backbone` extracts multi-scale feature maps (`low_features` and `high_features`), and `model.classifier` is the lightweight Lite R-ASPP decoder head that predicts the segmentation mask.

* **What to freeze:** Freeze the early and middle convolutional blocks of `model.backbone` so low-level edges, grass blades, and lighting textures aren't distorted by a small dataset.
* **What to train:** Unfreeze the **last stage of `model.backbone`** plus **`model.classifier`** (the decoder head).

```python
# 1. Freeze entire model first
for param in model.parameters():
    param.requires_grad = False

# 2. Unfreeze the decoder head (LRASPP classifier)
for param in model.classifier.parameters():
    param.requires_grad = True

# 3. Unfreeze only the final block(s) of MobileNetV3 backbone (e.g., last 2-3 layers)
for block in list(model.backbone.children())[-2:]:
    for param in block.parameters():
        param.requires_grad = True

# Pass ONLY trainable parameters to the optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4,
)
```

#### 3. How to Train, Quantize & Run at 1–2 FPS
1. **Loss Function:** Use **BCEWithLogitsLoss + Dice Loss** (or Soft Dice Loss). Dry vegetation often covers a smaller portion of the frame than background, so combining pixel cross-entropy/BCE with Dice prevents the model from predicting "all background" and getting high pixel accuracy with a terrible mask.
2. **Training Setup:** Resize/crop frames to a consistent resolution (e.g., `512x512` or `640x640`), train for **20–40 epochs** with a small batch size (4 or 8) and a low learning rate (`1e-3` on decoder, `1e-4` on unfrozen backbone block).
3. **Export to ONNX & FP16:** Export with `torch.onnx.export(..., opset_version=17)` using a static input shape like `(1, 3, 512, 512)`. Run in **ONNX Runtime (CUDA)** or **TensorRT (FP16)**. Because `Batch Size = 1` and the model is compact (~3.2M parameters), VRAM stays around **150–300 MB**.

---

### Approach B: DINOv2-Small Frozen Encoder + Custom Conv Decoder

#### 1. Where to Get the Model
* **Source:** Meta AI’s **DINOv2 (`dinov2_vits14`)** via PyTorch Hub (`facebookresearch/dinov2`) under the **Apache 2.0** license.
* **How to load:**
  ```python
  import torch

  # Apache 2.0 self-supervised Vision Transformer backbone (ViT-S/14)
  encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
  ```
* *Important detail:* `dinov2_vits14` uses a **patch size of 14** and outputs **384-dimensional** patch tokens (`embedding_dim = 384`). Input height and width should be multiples of `14` (for example, `518x518`, since `518 / 14 = 37` patches per side).

#### 2. How to Freeze Layers & Build the Custom Decoder
Here, you **freeze 100% of the DINOv2 backbone** and attach a **small convolutional decoder** that you write yourself.

* **Why freeze 100% of DINOv2?** DINOv2 was trained self-supervised on 142 million images, so its patch embeddings already capture strong structural and material distinctions without needing backbone fine-tuning—and freezing 100% of it keeps training fast and prevents overfitting on scarce CCTV frames.
* **How to reshape tokens into a spatial grid & decode:**
  ```python
  import torch
  import torch.nn as nn


  class DryVegConvDecoder(nn.Module):

      def __init__(self, in_channels=384, num_classes=1):
          super().__init__()
          self.net = nn.Sequential(
              nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              nn.GELU(),
              nn.Upsample(
                  scale_factor=2, mode="bilinear", align_corners=False
              ),
              nn.Conv2d(128, 64, kernel_size=3, padding=1),
              nn.BatchNorm2d(64),
              nn.GELU(),
              nn.Upsample(
                  scale_factor=2, mode="bilinear", align_corners=False
              ),
              nn.Conv2d(64, num_classes, kernel_size=1),
          )

      def forward(self, patch_tokens, grid_h, grid_w):
          # patch_tokens shape: (B, grid_h * grid_w, 384)
          x = patch_tokens.transpose(1, 2).reshape(
              -1, patch_tokens.shape[-1], grid_h, grid_w
          )
          logits = self.net(x)
          return logits


  # Freeze DINOv2 encoder completely
  for param in encoder.parameters():
      param.requires_grad = False
  encoder.eval()

  decoder = DryVegConvDecoder(in_channels=384, num_classes=1)
  optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
  ```

In your forward pass on a `518x518` image (`grid_h = 37`, `grid_w = 37`):
1. Extract spatial patch tokens via `feats = encoder.forward_features(x)["x_norm_patchtokens"]`.
2. Pass `(feats, 37, 37)` into `decoder` to get a spatial mask map, then upsample the final mask to `518x518` (or back to your ROI crop size) using bilinear interpolation before thresholding (e.g., `sigmoid(logits) > 0.5`).

#### 3. Smart Speed Trick for Training & Inference
* **Feature Caching (Huge Time Saver):** Because the `encoder` is completely frozen and deterministic for a given image, you can run `encoder` **once** over your small training set, save the extracted patch tensors to disk, and train `DryVegConvDecoder` in seconds.
* **Live Inference at 1–2 FPS:** At runtime, run `encoder` + `decoder` in `torch.no_grad()` or export the combined/split pipeline to **ONNX / TensorRT FP16** with `Batch Size = 1`. Expected VRAM stays around **300–500 MB**.

---

### The Manual Work Required to Hit >70% Reliability

Because you have limited training data and a fixed camera, **data quality and post-processing matter just as much as the model**:

1. **Curate 60–120 Diverse Frames From Your Exact Camera:**
   * Pull frames across **different times of day** (morning, harsh midday sun, late afternoon, dusk/overcast) and **different dryness stages** (lush green, drying/yellowing, bone-dry straw/brush).
   * Include **hard negative frames** (dry dirt, concrete, gravel, sand, dry wooden pallets/debris) so the model learns what *not* to mark as dry vegetation.
2. **Draw Clean Polygon / Pixel Masks:**
   * Annotate dry vegetation regions using a tool like **CVAT**, **LabelMe**, or **Roboflow**.
   * Be consistent on boundaries: trace around clumps/patches of dry grass or brush rather than drawing giant sloppy boxes around empty ground.
3. **Define a Fixed-Camera ROI Mask (Once):**
   * Using OpenCV on a reference frame, mask out areas where dry vegetation cannot exist or isn't monitored (e.g., sky, indoor areas, paved roads, vehicle lanes). This immediately cuts background false positives.
4. **Apply Safe Augmentations During Training:**
   * Use **color jitter** (brightness, contrast, saturation), **random crops** (zooming into dry patches), and **horizontal flips** (if appropriate for the scene).
   * Avoid extreme rotation or heavy perspective warping that makes grass look unnatural compared to your fixed camera angle.
5. **Add Temporal Smoothing & Tune Thresholds on Validation Video:**
   * Test on held-out video clips (not frames from the exact same 2-second burst as your training set).
   * Tune your **probability threshold** (e.g., `0.5` vs `0.65`), remove tiny noise specks with **morphological opening**, and require **2 out of 3 frames** at 1–2 FPS to confirm a mask—verifying that **IoU / mask agreement and false-alarm rate** clear your **>70%** bar.