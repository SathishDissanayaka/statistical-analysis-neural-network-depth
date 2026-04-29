# 🧠 A Deep Statistical Investigation of Depth vs Representational Capacity in Neural Networks

### Investigating whether neural network depth truly improves learning capacity

**Experiment ID:** `depth_experiment_v5`
**Dataset:** Fashion-MNIST
**Total Runs:** 1,800 models
**Team:** Team Deviants

---

## 📌 Overview

This project explores a fundamental question in deep learning research:

> **Does increasing neural network depth improve representational capacity?**

We conduct a controlled experimental study that isolates **depth effects from parameter scaling**, addressing a key limitation in many prior works where depth and parameter count are conflated.

---

## 🎯 Key Insight

> **Depth alone does NOT increase representational capacity. Parameters do.**

* Increasing depth without increasing parameters → ❌ No improvement
* Increasing depth with parameter growth → ✅ Significant improvement
* Core issue → parameter distribution bottleneck across layers

---

## 🧪 Experimental Design

### 🔁 Two-Regime Framework

We isolate the effect of depth using two controlled regimes:

| Regime          | Depth | Parameters | Purpose                     |
| --------------- | ----- | ---------- | --------------------------- |
| **iso_param**   | ✅     | ❌ Fixed    | Pure depth effect           |
| **fixed_width** | ✅     | ✅ Growing  | Real-world scaling behavior |

---

### 📊 Configuration

* **Depths tested:** 2, 4, 6, 8, 12, 16
* **Seeds per config:** 10
* **Corruption levels:** 0.0, 0.6, 1.0

### 🔢 Total Experiments

* 900 runs (iso_param)
* 900 runs (fixed_width)
* **Total: 1,800 models**

---

## 📂 Dataset & Quality Assurance

* Fully balanced factorial design
* No missing values
* No failed training runs
* Accuracy range: 0.10 → 0.99
* Early stopping applied (median epoch ≈ 35)

---

## ⚙️ Feature Engineering

We engineered **9 hypothesis-driven features**:

* `gen_gap` → overfitting signal
* `epoch_fraction` → training efficiency
* `total_flops` → compute usage
* `loss_drop_10` → convergence strength
* `depth_group`, `corruption_group` → stratification
* `log_n_params` → parameter scaling
* `log_total_flops` → normalized compute
* ⭐ `depth_per_param` → key bottleneck indicator

---

## 🔍 Key Findings

### 1️⃣ Depth Alone Does NOT Increase Capacity

**iso_param regime:**

* Accuracy: ~67–70% (flat trend)
* Spearman: r = -0.08 (not significant)
* Kruskal-Wallis: p = 0.118

👉 **Fail to reject null hypothesis**

---

### 2️⃣ Parameters Drive Capacity

* Correlation: r ≈ 0.97
* OLS: strong positive coefficients (β ≈ +2.3 to +2.9)
* Highly statistically significant

👉 **Parameters are the dominant factor**

---

### 3️⃣ Depth Helps Only With Parameter Growth

**fixed_width regime:**

* Accuracy gain: +5.7%
* Spearman: r = +0.64 (p < 0.001)
* Significant group differences (Kruskal-Wallis)

👉 Depth effect is interaction-based, not intrinsic

---

## 🧠 Bottleneck Explanation

### 🔑 Key Metric

`depth_per_param = depth / number_of_parameters`

* iso_param → ratio increases → ❌ bottleneck
* fixed_width → ratio stable → ✅ efficient scaling

### Insight:

> Increasing depth spreads parameters too thin across layers, reducing per-layer representational strength.

---

## 📈 Statistical Analysis

### Why Non-Parametric Methods?

* ❌ Shapiro-Wilk: normality violated
* ❌ Levene’s test: variance inequality

### Methods Used:

* Kruskal-Wallis test
* Dunn’s post-hoc (Holm correction)
* Rank-biserial correlation

---

## 🤖 Predictive Modeling

### Models Applied:

* Linear: OLS, Lasso
* Non-linear: Decision Tree, Random Forest
* Explainability: SHAP, Partial Dependence

### 🔥 Consensus Across Models:

| Model                | Finding                    |
| -------------------- | -------------------------- |
| OLS                  | Depth coefficient negative |
| Lasso                | Depth removed entirely     |
| Decision Tree        | Depth importance ≈ 0       |
| Random Forest + SHAP | Parameters dominate        |

---

## 🧾 Final Conclusion

> ❌ Depth ≠ representational capacity
> ✅ Parameters = representational capacity

### 🧠 Core Principle

> **“Depth is a vessel for parameters, not a source of capacity.”**

---

## 🚀 Practical Implications

* Prefer well-parameterized shallow models
* Be cautious when interpreting depth scaling laws
* Use `depth_per_param` as a diagnostic metric
* Fine-tuned small models can outperform deeper under-parameterized networks

---

## 🛠️ Tech Stack

* Python (NumPy, Pandas, Scikit-learn)
* SciPy (statistical testing)
* Matplotlib / Seaborn (visualization)
* SHAP (model explainability)

---

## 📁 Outputs

* `data_preprocessed.csv`
* Regime-wise datasets
* Trained models
* Statistical test reports
* Feature importance analyses

---

## 👥 Team

**Team Deviants**
Deep Learning Research Group
