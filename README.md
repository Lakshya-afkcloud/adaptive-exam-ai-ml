# Adaptive Exam AI-ML System (NAVIKA)

An AI-driven **real-time adaptive examination system** that dynamically adjusts question difficulty during an ongoing test by analyzing both **cognitive performance** and **behavioral patterns** of learners.

This project implements a **dual-loop adaptive engine** that improves assessment accuracy, fairness, and efficiency compared to traditional static or correctness-only adaptive tests.

---

## 🔍 Problem Addressed
Conventional online exams rely on static or accuracy-only adaptive models, which:
- Fail to capture real learner behavior
- Reward lucky guessing
- Penalize anxious or slow-but-capable students
- Increase test length without improving evaluation quality

This system solves these limitations using **real-time multi-parameter intelligence**.

---

## 🧠 System Overview (NAVIKA Architecture)

The system operates using a **Dual-Loop Model**:

### 1️⃣ Offline Intelligence Layer (Learning Phase)
- Processes historical student interaction data
- Extracts behavioral features such as:
  - Accuracy
  - Response Speed (Speed Factor)
  - Consistency
- Applies **unsupervised learning (K-Means clustering)** to discover behavioral cohorts
- Trains a cohort model that acts as the system’s **adaptive brain**

### 2️⃣ Online Adaptive Engine (Execution Phase)
- Runs during a live examination
- Continuously updates:
  - Student ability using psychometric modeling (IRT-based estimation)
  - Behavioral state using real-time response patterns
- Selects the **next optimal question** based on both skill level and behavior

This approach prevents score inflation from guessing and enables recovery paths for struggling learners.

---

## 🗄️ Database & Dataset Description

The project uses a **hybrid educational dataset** constructed from multiple real-world learning platforms to ensure robustness and generalization.

### Dataset Characteristics:
- Anonymized student interaction logs
- Each record includes:
  - Student ID
  - Question ID
  - Correctness (binary)
  - Response Time (seconds)

### Data Sources (Aggregated & Normalized):
- Large-scale mathematics learning datasets
- Skill-based practice logs
- Timed problem-solving records

### Key Processing Steps:
- Temporal normalization of response times
- Speed Factor computation for cross-question comparison
- Feature engineering for behavioral profiling
- Train–test split for offline modeling and live simulation

⚠️ **Note:**  
Raw datasets, images, trained models, and intermediate outputs are intentionally excluded from the repository to keep it lightweight and reproducible.

---

## 🧪 Key Functional Modules
- Data preprocessing and normalization
- Behavioral feature extraction
- Cohort modeling using unsupervised learning
- Real-time adaptive navigation engine
- Evaluation and diagnostic analysis

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- NumPy / Pandas
- Machine Learning (Clustering, Feature Engineering)
- Psychometric Modeling (IRT-inspired logic)

---

## 📌 Key Outcomes
- Reduces unnecessary questions while maintaining evaluation reliability
- Differentiates true ability from guessing behavior
- Improves fairness for slow or anxious learners
- Generates richer performance analytics beyond final scores

---

## 🚀 Future Scope
- Integration with live examination platforms
- Multi-modal behavioral signals (keystroke dynamics, fatigue detection)
- Explainable AI for transparent assessment decisions
- Scalable deployment for large-scale online exams

---
