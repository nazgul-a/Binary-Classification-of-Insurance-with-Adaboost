# 🏦 Binary Classification of Insurance with Adaboost (Kaggle Competition)

This project explores **ensemble learning** by first implementing **AdaBoost manually from scratch** and then comparing it with modern boosting algorithms (**XGBoost, LightGBM, CatBoost**).  

The scratch version achieved only ~0.5 accuracy (similar to random guessing), but it clearly demonstrated how boosting works step by step.  
In contrast, the library models performed much better after **hyperparameter fine-tuning**, highlighting the practical power of these tools.  

---

## 📊 Dataset  
- **Source:** Binary insurance classification dataset (Kaggle-inspired).  
- **Target:** Predict whether a customer will purchase insurance.  

---

## 🔧 Methods  
- Manual implementation of AdaBoost (weak learners + weight updates).  
- Comparison with boosting libraries:  
  - XGBoost (`XGBClassifier`)  
  - LightGBM (`LGBMClassifier`, with early stopping)  
  - CatBoost (`CatBoostClassifier`)  
- Hyperparameter tuning for model optimization.  

---

## 💡 Key Insights  
- Scratch AdaBoost: ~0.5 accuracy, confirming implementation gaps.  
- Modern boosting libraries: strong performance after tuning.  
- Key lesson: **conceptual understanding + applied practice** gives a full picture of how boosting works.  

---

## 🚀 Skills Demonstrated  
- Algorithm implementation from scratch (AdaBoost).  
- Ensemble methods in machine learning.  
- Hyperparameter tuning with XGBoost, LightGBM, and CatBoost.  
- Critical comparison of theoretical vs. practical implementations.  

---

## 📄 Full Analysis  
See the complete report with code and plots here: **[analysis.md](Binary-Classification-of-Insurance-with-Adaboost/Binary-Classification-of-Insurance-with-Adaboost.md)**.  
