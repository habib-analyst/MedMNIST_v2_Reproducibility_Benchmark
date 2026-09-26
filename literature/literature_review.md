# Literature review for joint MedMNIST learning

Reviewed from the primary papers, supplements, and official repositories on 2026-09-21. Each “Project implication” is an inference for this project, not a result claimed by the cited authors. No statement below is evidence that the proposed MedMNIST model has been implemented or evaluated.

## 1. A fully open AI foundation model applied to chest radiography (Ark+)

**Citation:** Ma, D., Pang, J., Gotway, M. B., and Liang, J. (2025). *A fully open AI foundation model applied to chest radiography*. Nature, 643, 488–498. https://doi.org/10.1038/s41586-025-09079-8  
**Code:** https://github.com/JLiangLab/Ark

- **Problem and insufficiency of prior work:** large proprietary chest-radiography models are difficult to reproduce, while public datasets are individually smaller and use heterogeneous expert labels. Collapsing those labels into one manually aligned vocabulary is costly and can erase dataset-specific semantics.
- **Model/data flow:** one shared representation model is exposed cyclically to differently labelled chest-radiography datasets; task-specific heads retain the native label spaces. The teacher–student mechanism reuses accumulated knowledge across tasks.
- **Knowledge sharing:** the representation is shared, but prediction heads are task specific. Cyclic training moves learned information between tasks without requiring a single universal label index.
- **Loss and schedule:** each dataset uses the loss appropriate to its own head. The paper and supplement study multi-task heads and cyclic training; teacher parameters accumulate student knowledge through exponential moving averages during the cycle.
- **Strongest evidence:** the paper evaluates transfer across multiple downstream settings and supplies public code, pretrained weights, and supplementary source data, making openness itself part of the contribution.
- **Main limitation for this project:** every pretraining dataset is a chest radiograph. MedMNIST includes histology, retinal, dermoscopic, ultrasound, abdominal CT-derived, and other images; Ark+ does not establish that one representation will transfer cleanly across that wider modality range.
- **Project implication:** use a shared representation with dataset-specific heads and controlled dataset exposure. Treat cyclic teacher–student training as a later ablation, not as an assumed requirement.

## 2. Foundation Ark: Accruing and Reusing Knowledge for Superior and Robust Performance

**Citation:** Ma, D., Pang, J., Gotway, M. B., and Liang, J. (2023). *Foundation Ark: Accruing and Reusing Knowledge for Superior and Robust Performance*. MICCAI 2023, 651–662. https://doi.org/10.1007/978-3-031-43907-0_62  
**Paper and review record:** https://conferences.miccai.org/2023/papers/282-Paper0231.html  
**Code:** https://github.com/JLiangLab/Ark

- **Problem and insufficiency of prior work:** high-performing medical models usually depend on very large labelled collections, yet many public datasets are smaller and have non-identical expert annotations.
- **Model/data flow:** Ark aggregates public chest X-ray datasets and learns from their heterogeneous supervision, then evaluates the resulting representation through fine-tuning, linear probing, and bias analysis.
- **Knowledge sharing:** a common model accrues knowledge from multiple datasets while preserving supervision from each source rather than treating datasets as interchangeable records.
- **Loss and schedule:** supervised objectives follow the available labels for the active dataset. The central training idea is sequential accrual and reuse rather than one merged label table.
- **Strongest evidence:** the paper reports broad downstream evaluation of two Ark models trained on hundreds of thousands of images and compares them with fully supervised, self-supervised, and proprietary-model baselines.
- **Main limitation for this project:** its evidence concerns chest radiographs and substantially larger pretraining collections than most MedMNIST subsets. Dataset scale and domain diversity are therefore confounded with the sharing strategy.
- **Project implication:** the initial MedMNIST joint model should preserve dataset identity and native heads, then measure whether the shared representation helps small datasets without degrading others.

## 3. Adam-v2: anatomy-based self-supervision

**Citation:** Taher, M. R. H., Gotway, M. B., and Liang, J. (2024). *Representing Part-Whole Hierarchies in Foundation Models by Learning Localizability, Composability, and Decomposability from Anatomy via Self-Supervision*. CVPR 2024, 11269–11281. https://openaccess.thecvf.com/content/CVPR2024/html/Taher_Representing_Part-Whole_Hierarchies_in_Foundation_Models_by_Learning_Localizability_Composability_CVPR_2024_paper.html

- **Problem and insufficiency of prior work:** generic self-supervised objectives can learn useful appearance features without explicitly representing anatomical part–whole structure.
- **Model/data flow:** a teacher–student encoder sees whole images and multi-scale anatomical crops. Localizability distinguishes anatomical patterns; composability predicts a whole representation from its parts; decomposability predicts part representations from the whole.
- **Knowledge sharing:** the three branches share the representation learner and impose complementary constraints across anatomical scales.
- **Loss and schedule:** the reported objective is `L = λ1 L_localizability + λ2 L_composability + λ3 L_decomposability`, with all three weights set to 1 in the paper. Composability and decomposability use MSE similarity; training proceeds coarse-to-fine over anatomical scales and updates the teacher from the student.
- **Strongest evidence:** evaluation covers zero-shot structure analysis and few-shot/full transfer on ten downstream tasks across chest-radiograph and fundus domains, with at least five runs per downstream method/task in the reported fine-tuning protocol.
- **Main limitation for this project:** part–whole anatomy is meaningful for some MedMNIST datasets but much weaker for isolated blood cells or small tissue patches. Applying the objective indiscriminately could introduce a false inductive bias.
- **Project implication:** keep Adam-v2-style objectives optional and dataset gated. The first joint classifier should establish whether supervised sharing works before adding anatomical self-supervision.

## 4. Foundation X

**Citation:** Islam, N. U., Ma, D., Pang, J., Velan, S. S., Gotway, M., and Liang, J. (2025). *Foundation X: Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis*. WACV 2025. https://openaccess.thecvf.com/content/WACV2025/papers/Islam_Foundation_X_Integrating_Classification_Localization_and_Segmentation_through_Lock-Release_Pretraining_WACV_2025_paper.pdf  
**Code:** https://github.com/JLiangLab/Foundation_X

- **Problem and insufficiency of prior work:** public chest X-ray supervision spans image-level classes, boxes, and masks. A model trained on only one annotation form leaves other expert knowledge unused; unconstrained joint updates can also interfere.
- **Model/data flow:** shared features feed classification, localization, and segmentation branches. Cyclic learning determines which dataset/task is active, while Lock-Release controls which parameters are trainable during stages of pretraining.
- **Knowledge sharing:** the shared representation receives knowledge from heterogeneous annotation types while task branches keep their output structures explicit.
- **Loss and schedule:** classification, localization, and segmentation branches retain task-appropriate objectives. The defining schedule alternates cyclic task exposure with locked and released parameter groups; official scripts specify the exact implementation.
- **Strongest evidence:** ablations compare the integrated branches and pretraining strategies, and downstream experiments test whether multi-annotation pretraining benefits localization and segmentation rather than only classification.
- **Main limitation for this project:** the paper’s heterogeneity is primarily annotation type within one imaging modality. MedMNIST’s central mismatch is dataset, modality, label space, and 2D/3D dimensionality.
- **Project implication:** if simple joint learning shows negative transfer, test staged freezing/releasing as a controlled ablation. Do not add Lock-Release before establishing the simpler baseline.

## 5. Models Genesis

**Citation:** Zhou, Z., Sodha, V., Pang, J., Gotway, M. B., and Liang, J. (2021). *Models Genesis*. Medical Image Analysis, 67, 101840. https://doi.org/10.1016/j.media.2020.101840  
**Open full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7726094/

- **Problem and insufficiency of prior work:** 2D natural-image pretraining cannot fully represent volumetric anatomy, and labelled 3D medical datasets are comparatively scarce.
- **Model/data flow:** randomly cropped 3D sub-volumes undergo non-linear intensity transformation, local shuffling, inner cutout, or outer cutout. An encoder–decoder restores the original sub-volume; the encoder transfers to classification and the full network to segmentation.
- **Knowledge sharing:** four restoration schemes are consolidated into one encoder–decoder rather than using a separate tower for each pretext task.
- **Loss and schedule:** reconstruction minimizes mean squared error between restored and original sub-volumes. Each crop can receive up to three transformations, with inner and outer cutout mutually exclusive.
- **Strongest evidence:** the combined restoration scheme is evaluated on five 3D target applications and is compared with individual transformations, random initialization, and 2D transfer approaches.
- **Main limitation for this project:** the demonstrated benefit is transfer from volumetric self-supervision, not simultaneous supervised optimization across six incompatible MedMNIST3D label spaces.
- **Project implication:** retain native 3D operators and consider restoration pretraining only after the independent and joint supervised baselines are measured.

## Cross-paper synthesis

The five papers support one conservative principle: share representations, but keep incompatible supervision explicit. Ark and Ark+ motivate dataset-specific heads and controlled task exposure. Adam-v2 shows how auxiliary objectives can encode anatomical structure when that structure is actually present. Foundation X motivates staged sharing when gradients from heterogeneous tasks interfere. Models Genesis demonstrates why the six 3D datasets should retain volumetric processing.

The proposed research sequence is therefore:

1. Reproduce independent baselines using unchanged official definitions.
2. Train a 12-dataset 2D model and a six-dataset 3D model with one shared encoder per dimensionality, one head per dataset, balanced sampling, and normalized task losses.
3. Measure negative transfer per dataset against the five-seed independent baseline.
4. Add one mechanism at a time—dataset conditioning, adapters, gradient-conflict mitigation, or staged Lock-Release—only when a measured failure motivates it.
5. For the unified 18-dataset setting, use 2D and 3D stems that project to a shared width, followed by shared blocks and dataset-specific heads.
6. Compare at matched and explicitly reported optimization budgets; report per-dataset metrics, macro averages, worst-dataset change, parameter count, memory, and wall-clock time.

## Anticipated questions and defensible answers

### Ark+

1. **Why not merge all labels into one vector?** Because identical indices across datasets do not imply identical clinical concepts; separate heads preserve each label space and remove manual ontology assumptions.
2. **What does cyclic training try to solve?** It controls how the shared model sees heterogeneous tasks and reduces simultaneous gradient competition; whether it helps MedMNIST must be tested.
3. **What is the key transfer risk?** Ark+ is chest-X-ray specific, whereas MedMNIST spans much broader modalities and dimensionalities.

### Foundation Ark

1. **What is “accruing and reusing” knowledge?** It means updating a common representation from multiple public datasets so later tasks can benefit from information learned earlier while their supervision remains explicit.
2. **Why is openness scientifically relevant?** Public data, code, and weights make the training evidence inspectable and allow independent reproduction.
3. **What comparison is essential here?** Joint learning must be compared with independent models under transparent data and optimization budgets; otherwise more updates may be mistaken for better sharing.

### Adam-v2

1. **How do its three branches differ?** Localizability discriminates anatomical patterns, composability builds a whole from parts, and decomposability recovers parts from the whole.
2. **Why not apply it to every MedMNIST dataset?** Some datasets lack a meaningful spatial anatomy hierarchy, so the inductive bias may be inappropriate.
3. **Where would it enter this project?** As optional pretraining or an auxiliary objective after the supervised joint baseline, followed by a dataset-gated ablation.

### Foundation X

1. **What is Lock-Release?** It is a staged training policy that freezes and later unfreezes parameter groups while tasks are learned cyclically.
2. **Why could it reduce interference?** It limits which shared parameters a task can immediately overwrite, then permits controlled joint adaptation.
3. **Why is it not the first model here?** Its additional schedule can obscure whether ordinary shared features and separate heads already solve the problem.

### Models Genesis

1. **Why is native 3D processing important?** Slice-wise 2D processing discards through-plane anatomical context that the 3D restoration objective explicitly learns.
2. **What prevents a trivial identity solution?** Inputs are transformed before restoration; the paper also reports that identity mapping is not a strong representation-learning objective.
3. **What does the paper not prove?** It does not prove that joint supervised training across the six MedMNIST3D datasets will avoid negative transfer.

## Independent multi-task MedMNIST work

### A Shared-Backbone Approach for Multi-Task MedMNIST Classification (SAIIT 2026)

Gavril, Arhire and Iftene, arXiv:2609.06838, submitted 6 September 2026.
Code: `github.com/GavrilStefan-Dorian/A-Shared-Backbone-Approach-for-Multi-Task-MedMNIST-Classification`

- **Problem and insufficiency of prior work:** independent per-dataset models duplicate
  parameters and cannot reuse structure, while a single shared classifier conflates
  unrelated label spaces across modalities and class distributions.
- **Model/data flow:** one shared backbone with a task-specific linear head per dataset,
  evaluated over 11 heterogeneous MedMNIST datasets. Backbones compared were ConvNeXt-Tiny,
  ResNet-18 and EfficientNet-B0.
- **Strongest evidence:** placed sixth in the *Tensor Reloaded: Multi-Task MedMNIST*
  competition with a harmonic-mean macro-F1 of 0.73294 (ConvNeXt-Tiny with label
  smoothing). Code and a runnable notebook are released.
- **Non-obvious finding worth carrying into this project:** the authors report a
  *resolution domain shift between the MedMNIST API and the evaluation environment*, and
  state that resolving it, together with architecture-specific regularisation,
  substantially improved performance. Any joint model here must verify that the
  resolution seen at training time matches evaluation time.
- **Main limitation for this project:** it covers 11 two-dimensional datasets, reports
  macro-F1 rather than AUC, and does not address the 2D/3D bridging problem. It is
  therefore evidence that selective sharing works, not a substitute for the unified
  18-dataset design.
- **Project implication:** the shared-backbone-plus-task-heads pattern is now an
  established baseline rather than an open idea. The contribution claimed here must be
  the 2D/3D dimension-specific stems, explicit dataset conditioning, task-normalised
  losses, and quantified negative transfer. A joint model that only shares a backbone
  would no longer be novel.

## Evidence boundary

This literature record supports the research design only. No proposed mechanism is reported as a measured project result. All performance statements must be added only after a traceable run produces predictions and official evaluator outputs.
