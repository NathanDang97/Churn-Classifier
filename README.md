# 📊 Churn Prediction for Subscription Services

This project (rooted from a Data Science Challenge powered by Coursera) tackles the industry-critical problem of **customer churn prediction** for a video streaming service using a real-world, imbalanced dataset. The goal is to identify customers at high risk of canceling their subscriptions — enabling targeted retention strategies.

Through extensive **feature engineering**, **model ensembling**, and **class imbalance handling**, the final solution achieved a **ROC AUC score of 0.75**, placing it in the **91st percentile** on the leaderboard.

## 🔍 Key Highlights

- **Ensembled model** combining a Neural Network, XGBoost, and Random Forest for robust predictive performance.
- **Feature engineering** included interaction terms, log transformations, behavioral ratios, and bucketed features.
- **Class imbalance handled** without oversampling, using `class_weight`, `scale_pos_weight`, and early stopping techniques.
- **Manual hyperparameter tuning** focused on high-impact parameters to optimize performance with minimal compute cost.

## 📈 Final Performance

- **ROC AUC:** 0.75  
- **Leaderboard Percentile:** 91st
