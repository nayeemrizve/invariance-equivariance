# Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning

Official implementation of "Exploring Complementary Strengths of Invariant and Equivariant Representations for Few-Shot Learning". [Go to the Support Web Site](https://support.west-wind.com). The paper reports state-of-the-art results on five popular few-shot learning benchmarks.

Real-world contains an overwhelmingly large number of object classes, learning all of which at once is impossible. Few shot learning is a promising learning paradigm due to its ability to learn out of order distributions quickly with only a few samples. Recent works show that simply learning a good feature embedding can outperform more sophisticated meta-learning and metric learning algorithms. In this paper, we propose a simple approach to improve the representation capacity of deep neural networks for few-shot learning tasks. We follow a two-stage learning process: First, we train a neural network to maximize the entropy of the feature embedding, thus creating an optimal output manifold using self-supervision as an auxiliary loss. In the second stage, we minimize the entropy on feature embedding by bringing self-supervised twins together, while constraining the manifold with student-teacher distillation. Our experiments show that, even in the first stage, auxiliary self-supervision can outperform current state-of-the-art methods, with further gains achieved by our second stage distillation process.

This official code provides an implementation for our SKD. This repository is implemented using PyTorch and it includes code for running the few-shot learning experiments on CIFAR-FS, FC-100, miniImageNet and tieredImageNet datasets.


