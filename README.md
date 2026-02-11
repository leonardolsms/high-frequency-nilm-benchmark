# A Reproducible Comparison of Classical and Deep Learning Models for High Frequency NILM Using Harmonic Features

This repository provides the full experimental framework, code, and results
associated with the paper:

**A Reproducible Comparison of Classical and Deep Learning Models for
High Frequency NILM Using Harmonic Features**

submitted to the SBRC conference.

---

## 📌 Overview

Non Intrusive Load Monitoring (NILM) has become a key enabling
technology for smart metering and smart grid infrastructures, particularly as
high frequency electrical measurements are increasingly processed at the net-
work edge to reduce latency, communication overhead, and privacy risks. In
such distributed environments, the choice of learning models directly affects
system level properties, including inference latency, computational cost, and
scalability.
This paper presents a fully reproducible benchmark comparing classical ma-
chine learning and deep learning models for high frequency NILM, with a par-
ticular emphasis on the role of harmonic features extracted from aggregate elec-
trical measurements. Nine learning architectures are evaluated, namely Linear
Regression, k-Nearest Neighbors, Decision Trees, Random Forests, Gradient
Boosting, XGBoost, LightGBM, a Multilayer Perceptron, and a Seq2Point deep
learning model under a unified experimental pipeline explicitly designed with
edge oriented constraints in mind.
Rather than addressing supervised appliance level disaggregation, this work
investigates the contribution of high-frequency harmonic features to electrical
power modeling within smart grid monitoring scenarios. Using a large scale,
high resolution dataset and targeted ablation studies, we demonstrate that har-
monic components account for the dominant share of predictive information,
while lightweight classical models achieve performance comparable to more
complex architectures. These results highlight the practical relevance of har-
monic analysis for scalable, edge oriented energy monitoring systems.

***Keywords:*** Non Intrusive Load Monitoring; Edge Computing; Smart Metering;
Distributed Systems; High Frequency Measurements; Harmonic Analysis.

## Reproducibility

To ensure full reproducibility of the experimental results, all source code used in
this study is publicly available. The repository includes data preprocessing routines,
model training scripts, and performance evaluation pipelines for all nine evaluated mod-
els. A fully executable Google Colab notebook is provided to facilitate replication of
the experiments under identical conditions.T

## Dataset

The dataset used in this study is publicly available and was obtained from the
high-frequency NILM repository maintained by [Dinar et al. 2025] The dataset can be
accessed at https://github.com/fariddinar/nilm-dataset (accessed on 20 December 2025).

## Dataset and Feature Space

The dataset used in this study is a high frequency NILM dataset based on harmonic feature
analysis, originally introduced by [Dinar et al. 2025]. The experiments are conducted
using a high frequency electrical dataset composed of synchronized measurements of
electrical quantities and harmonic components. For each acquisition session and sensor
phase channel, the dataset provides root mean square current (Irms), root mean square
voltage (Vrms), power factor, apparent power, and the magnitudes of the first 32 current
harmonics. Appliance- level ground truth is not available in the dataset; therefore, the
experimental analysis focuses on the contribution of harmonic features to electrical power
modeling rather than supervised load disaggregation.
All measurements are acquired using a sensing system based on the ATM90E36A
energy metering integrated circuit, which computes harmonic components through an
onboard Discrete Fourier Transform applied to an 8 kHz-sampled signal. Feature values
are logged at two second intervals and collected across multiple measurement sessions
and phases, enabling the analysis of both steady-state and dynamic operating conditions.
The dataset is partitioned at the session level, ensuring that samples from the same
acquisition session do not appear simultaneously in training and testing sets. Feature
normalization is performed using statistics computed exclusively from the training data
and then applied to the test data.


---

## 📂 Repository Structure

```text
notebook/    -> Jupyter notebook with all experiments
data/        -> Instructions to obtain the dataset
figures/     -> Final figures used in the paper
paper/       -> LaTeX source of the camera-ready paper
results/     -> Tables with numerical results

