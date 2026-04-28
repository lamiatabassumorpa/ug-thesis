# Report 05: Thesis Defense Preparation
## Simulated Questions & Model Answers
### For BSc Thesis Viva / Committee Defense

**Project:** Multi-Stream Deepfake Face Detection
**Prepared by:** Research Committee (10 Experts)
**Date:** April 18, 2026

---

## How to Use This Document

Read each question carefully. Practice answering OUT LOUD before the defense. The model answers here are guidelines — adapt them in your own words. Examiners can tell when answers are memorized vs. understood.

**General rule:** If you don't know something, say: *"That's an interesting limitation I hadn't considered. One approach to address it would be..."* — showing awareness of limitations is more impressive than pretending to know everything.

---

## Category 1: Fundamental Understanding

**Q1. In simple terms, what exactly is a deepfake and why is detection hard?**

Model Answer:
> A deepfake is an AI-generated image that realistically depicts a person or face that doesn't exist, or a real person doing or saying something they never did. Detection is difficult for three main reasons: First, modern generative models produce images that are photorealistic and visually indistinguishable from real photos. Second, there are many different generators (Stable Diffusion, SDXL, Midjourney, StyleGAN) and each produces different artifact patterns — a detector trained on one may fail on another. Third, simple post-processing like JPEG compression, resizing, or social media re-uploading can destroy subtle artifacts that detectors rely on.

---

**Q2. What is AUC-ROC and why did you use it as the primary metric instead of accuracy?**

Model Answer:
> AUC-ROC stands for Area Under the Receiver Operating Characteristic curve. It measures how well the model separates the two classes across ALL possible classification thresholds, not just at the default 0.5 threshold. Accuracy is misleading when classes are imbalanced — if 90% of images are real, a model that always says "real" gets 90% accuracy but catches zero fakes. AUC is threshold-independent: our AUC of 99.92% means if you randomly pick one fake and one real image, there's a 99.92% chance the model ranks the fake as more suspicious. This is a more fundamental measure of discriminative ability.

---

**Q3. You report 100% in-distribution AUC. Isn't that suspicious? Could there be data leakage?**

Model Answer:
> This is a very valid concern and we took it seriously. We verified there is no data leakage in three ways: First, our stratified split uses a single fixed random seed (42), shuffles each class independently, and partitions into non-overlapping slices — guaranteeing no image appears in both train and test. Second, we had to debug and fix a leakage bug in the original code where two different RNG states could produce overlapping indices. Third, even after fixing leakage, we still get near-100% — which suggests our face-filtered DiffusionDB dataset may genuinely be an "easy" binary problem, because SD v1.x face artifacts are distinct enough from FFHQ photographs. The cross-generator evaluation (98.09% on SDXL) confirms the model learned genuine discriminative features, not memorized training data.

---

## Category 2: Architecture Questions

**Q4. Why three streams? Why not just use one very deep network?**

Model Answer:
> One very deep network typically learns a single hierarchy of features optimized for one type of signal. Spatial, frequency, and semantic information are fundamentally different in nature: spatial artifacts appear in pixel space, frequency artifacts appear in the Fourier spectrum, and semantic inconsistencies appear in high-level structural representations. A single backbone would have to learn all three implicitly, which empirically leads to worse generalization. Our ablation shows this directly: spatial+semantic TOGETHER performs worse than spatial alone on cross-generator testing (-6.36% vs -3.56%). This counter-intuitive result suggests the two streams interfere when not properly mediated. Adding the frequency stream as a third "mediator" resolves this, giving the best overall generalization (-1.91%).

---

**Q5. Why did you choose EfficientNet-B0 for the spatial stream? Have you compared other backbones?**

Model Answer:
> EfficientNet-B0 was chosen based on three criteria: efficiency (5.3M parameters, suitable for our GPU), compound scaling design (optimally balances depth/width/resolution), and published effectiveness in image forensics literature. We did not perform a systematic backbone comparison (e.g., ResNet-50 vs EfficientNet-B0 for the spatial stream) — this is a limitation. A fair comparison would require training multiple configurations with identical hyperparameters. I would list this as a future work item: ablating backbone choice within each stream. However, given that the full model achieves 99.92% AUC, the current choice is at least sufficient.

---

**Q6. The frequency stream gets only 68.5% AUC alone. Why include it at all?**

Model Answer:
> This is the most interesting finding in our ablation study. 68.5% standalone AUC means the frequency stream cannot detect deepfakes well on its own. But looking at the cross-generator table: spatial+semantic gives 93.64% SDXL AUC, while full model (adding frequency) gives 98.09%. The frequency stream contributes +4.45% by being combined with the other two. This is complementarity: the frequency stream encodes information that is orthogonal to what spatial and semantic streams capture. Its low standalone performance proves this orthogonality — if it captured the same information as spatial, it would have similar standalone AUC. We cite Durall et al. (2020) "Watch Your Up-Convolution" which shows spectral artifacts require multi-scale context to generalize across generators.

---

**Q7. How does cross-stream attention work mechanically?**

Model Answer:
> We project all three stream outputs to a common 256-dimensional space first. Then we stack them as a sequence of three vectors and apply multi-head self-attention with 4 heads. In self-attention, each vector in the sequence queries all other vectors: the spatial stream vector asks "which parts of the frequency and semantic outputs should I attend to?" This produces attention weights — for example, if the semantic stream found something suspicious in the face structure, the spatial stream will attend more heavily to semantic features, reinforcing its own suspicion. The 4 heads run this process in parallel with different learned "query perspectives," each specializing in different inter-stream relationships. The output is a residual connection (original + attended), then mean-pooled across the three streams into a single 256-dim representation.

---

## Category 3: Dataset and Evaluation

**Q8. Your training uses SD v1.x fakes. SDXL uses a fundamentally different architecture. How do you know your model generalizes vs. just exploiting shared Stable Diffusion artifacts?**

Model Answer:
> This is the strongest critique of our cross-generator evaluation. SD v1.x and SDXL are both latent diffusion models from the same family, sharing similar VAE (Variational Autoencoder) decoder architecture. They may share subtle spectral signatures from the same upsampling strategy. To truly test cross-family generalization, we would need to test on StyleGAN3, ProGAN, or Midjourney — architecturally unrelated generators. We acknowledge this limitation explicitly. Our current result (98.09% on SDXL) demonstrates within-family generalization. Future work should evaluate on GAN-based generators to test cross-paradigm generalization, which would be a substantially stronger claim.

---

**Q9. FFHQ images show 76.8% accuracy (the model calls 23% of FFHQ images "fake"). How do you explain this?**

Model Answer:
> FFHQ images are extremely high-quality, professionally aligned and retouched face photographs. They have very smooth skin textures, consistent lighting, and minimal noise — properties that, paradoxically, overlap with some characteristics of AI-generated images. The model appears to exhibit a "quality bias": it learned that real images have certain imperfections, and when it sees an unusually perfect real image, it flags it as suspicious. This is a documented phenomenon in deepfake detection literature. The fix would be to diversify the real image training set to include high-quality photographs alongside more natural images, or to use domain randomization during training. We report this honestly as a limitation rather than hiding it.

---

**Q10. Your fake dataset has only 5,524 images vs 15,000 real images (3:1 ratio). How did you handle this imbalance?**

Model Answer:
> We handled this in two ways. First, during training we used a weighted loss function — `BCEWithLogitsLoss` with `pos_weight=2.72` (≈ 15000/5524). This tells the model that misclassifying a fake image as real is 2.72× more costly than misclassifying a real image as fake. Second, we used stratified splitting to ensure both the training and test sets maintain the same 3:1 ratio, so metrics are computed on a representative sample. An alternative approach would be oversampling the minority class (fake images) using WeightedRandomSampler, which our code also supports but we used weighted loss instead for its mathematical cleanliness.

---

## Category 4: Technical and Implementation

**Q11. What prevented you from testing on Midjourney and DALL-E 3 images?**

Model Answer:
> We had the directory structure for GenImage (which includes Midjourney, DALL-E, ADM, GLIDE) configured, but the actual image files were not downloaded due to the large dataset size (~16GB) and time constraints during development. The SDXL evaluation uses a smaller, publicly available HuggingFace dataset (8clabs/sdxl-faces, 154MB) that was more practical to download. Testing on GenImage generators would straightforwardly extend this work and is listed as the highest-priority future work item.

---

**Q12. Why did you use WSL2 (Windows Subsystem for Linux) instead of native Linux?**

Model Answer:
> The development machine runs Windows 11 and has an RTX 5060 Ti GPU. WSL2 allows running a full Ubuntu environment with CUDA GPU access while maintaining the Windows development workflow. The RTX 5060 Ti (Blackwell architecture) has better driver support on Windows than native Linux at this time. The tradeoff is that WSL2's `fork()` system call is slower than native Linux, which is why `num_workers=0` was used for DataLoader (multiprocessing is slow in WSL2). This is documented in the config file as a comment.

---

## Category 5: Results and Claims

**Q13. You claim to outperform CNNDetect and UnivFD. Were these trained and tested on exactly the same data under exactly the same conditions?**

Model Answer:
> Yes. All three models — CNNDetect, UnivFD, and our model — were trained on the same faces_dataset with the same 80/10/10 split, same seed (42), same augmentation settings, and evaluated on the identical test set. The baselines were trained using `scripts/compare_baselines.py` which instantiates all models and calls the same training loop with the same hyperparameters where applicable. This is important: if we had used pre-trained baseline checkpoints from different datasets, the comparison would be unfair. All comparison results are from models trained from scratch on our dataset.

---

**Q14. What is your model's inference speed? Is it practical for real-time use?**

Model Answer:
> On an RTX 5060 Ti, single-image inference takes approximately 8ms. For batch processing (batch_size=32), effective throughput is around 200-300 images per second. This is fast enough for asynchronous content moderation (processing uploaded images before publication) but not fast enough for real-time video deepfake detection at 30fps, which would require processing 30 frames per second simultaneously. For video, a more lightweight single-stream spatial model would be more practical, potentially at the cost of generalization. This is an interesting application-level trade-off we note as future work.

---

## Category 6: Broader Context

**Q15. This system can detect AI-generated faces. Couldn't a bad actor just use this knowledge to make their generator harder to detect?**

Model Answer:
> Yes — this is the fundamental "arms race" dynamic in deepfake detection, acknowledged in the ethics section of our thesis. Publishing detection methods does provide information that can help adversarial image generators improve. However, the consensus in the research community is that transparency and open detection research is net-positive: it allows defenders, journalists, and platform moderators to access state-of-the-art tools, and advances in detection also provide feedback to the research community about which artifacts are most detectable, motivating more responsible generator design. Additionally, our cross-generator evaluation approach — testing on unseen generators — is explicitly designed to be robust to this adversarial dynamic by learning fundamental rather than generator-specific artifacts.

---

## Quick Reference: Numbers to Know by Heart

| Metric | Value |
|--------|-------|
| Test AUC-ROC | 99.92% |
| Test Accuracy | 94.25% |
| Test Recall (fake detection rate) | 99.82% |
| Test EER | 1.27% |
| SDXL Cross-Generator AUC | 98.09% |
| SDXL Generalization Drop | −1.91% (best among all configurations) |
| Training dataset size | ~20,524 images total |
| Fake training images | 5,524 (SD v1.x faces, DiffusionDB) |
| Real training images | ~15,000 (FFHQ + CelebA-HQ) |
| Best epoch | 15 |
| Total model parameters | ~22.9M |
| Hardware | RTX 5060 Ti 16GB, WSL2 Ubuntu, CUDA 12.8 |
