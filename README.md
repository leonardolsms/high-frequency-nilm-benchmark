# A Reproducible Comparison of Classical and Deep Learning Models for High-Frequency NILM Using Harmonic Features

This repository provides the full experimental framework, code, and results
associated with the paper:

**A Reproducible Comparison of Classical and Deep Learning Models for
High-Frequency NILM Using Harmonic Features**

submitted to the SBRC conference.

---

## 📌 Overview

Non-Intrusive Load Monitoring (NILM) seeks to decompose aggregate electrical consumption into appliance-level profiles without the need for
intrusive sub-metering. Recent advances in sensing technologies and signal acquisition systems have enabled high-frequency measurements, making it possible to extract detailed harmonic and power quality features that significantly
enhance appliance discrimination. Inspired by the high-frequency harmonic dataset proposed by [Dinar et al. 2025], this work presents a comprehensive and
fully reproducible comparison of classical machine learning and deep learning
models for NILM under a unified experimental protocol. Nine learning architectures are evaluated, ranging from linear and instance-based models to ensemble learners, shallow neural networks, and a sequence-to-point deep learning
approach. To ensure feasibility under realistic computational constraints, the
deep learning model is trained using reduced temporal windows while preserving its original architectural principles. Experimental results demonstrate that
classical ensemble methods and shallow neural networks achieve performance
comparable to, and in some cases superior to, more complex deep learning models when informative harmonic features are available. These findings reinforce
the relevance of feature engineering in high-frequency NILM and highlight the
importance of reproducible benchmarks for fair model assessment. All code,
preprocessing steps, and evaluation procedures are publicly released to support
transparency and future research.

## Data

The dataset used in this study is publicly available and was obtained from the
high-frequency NILM repository maintained by [Dinar et al. 2025] The dataset can be
accessed at https://github.com/fariddinar/nilm-dataset (accessed on 20 December 2025).


---

## 📂 Repository Structure

```text
notebook/    -> Jupyter notebook with all experiments
data/        -> Instructions to obtain the dataset
figures/     -> Final figures used in the paper
paper/       -> LaTeX source of the camera-ready paper
results/     -> Tables with numerical results

