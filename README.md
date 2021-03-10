# Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning

Official implementation of [Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning](https://arxiv.org/abs/2103.01315). The paper reports state-of-the-art results on five popular few-shot learning benchmarks.

In many real-world problems, collecting a large number of labeled samples is infeasible. Few-shot learning (FSL) is the dominant approach to address this issue, where the objective is to quickly adapt to novel categories in presence of a limited number of samples. FSL tasks have been predominantly solved by leveraging the ideas from gradient-based meta-learning and metric learning approaches. However, recent works have demonstrated the significance of powerful feature representations with a simple embedding network that can outperform existing sophisticated FSL algorithms. In this work, we build on this insight and propose a novel training mechanism that simultaneously enforces equivariance and invariance to a general set of geometric transformations. Equivariance or invariance has been employed standalone in the previous works; however, to the best of our knowledge, they have not been used jointly. Simultaneous optimization for both of these contrasting objectives allows the model to jointly learn features that are not only independent of the input transformation but also the features that encode the structure of geometric transformations. These complementary sets of features help generalize well to novel classes with only a few data samples. We achieve additional improvements by incorporating a novel self-supervised distillation objective. Our extensive experimentation shows that even without knowledge distillation our proposed method can outperform current state-of-the-art FSL methods on five popular benchmark datasets.

This repository is implemented using PyTorch and it includes code for running the few-shot learning experiments on CIFAR-FS, FC-100, miniImageNet and tieredImageNet datasets.

<p align="center">
  <img src="/figures/conceptual-1.png" width="500">
</p>
<p>
  <em>Approach Overview: Shapes represent different transformations and colors represent different classes. While the invariant features provide better discrimination, the equivariant features help us learn the internal structure of the data manifold. These complimentary representations help us generalize better to new tasks with only a few training samples. By jointly leveraging the strengths of equivariant and invariant features, our method achieves significant improvement over baseline (bottom row).</em>
</p>

<p align="center">
  <img src="/figures/training.png" width="800">
</p>
<p>
  <em>Network Architecture during Training: A series of transformed inputs (transformed by applying transformations T1...TM) are
provided to a shared feature extractor fΘ. The resulting embedding is forwarded to three parallel heads fΦ, fΘ and fΩ that focus on
learning equivariant features, discriminative class boundaries, and invariant features, respectively. The resulting output representations are
distilled from an old copy of the model (teacher model on the right) across multiple-heads to further improve the encoded representations.
Notably, a dedicated memory bank of negative samples helps stabilize our invariant contrastive learning.</em>
</p>

# Dependencies
