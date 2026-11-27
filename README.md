# Reducing COD Return Losses at Daraz Pakistan with AI
**Order Cancellation Prediction & Profit Maximization System**

## Overview
Cash on Delivery (COD) accounts for roughly 75-80% of e-commerce retail orders in Pakistan (~78% in our dataset). However, doorstep refusals and order cancellations create two-way reverse logistics costs and tied-up inventory.

Using 494,463 historical orders from Pakistan's Largest E-Commerce Dataset, this project builds a checkout-time risk scoring pipeline with LightGBM to predict order cancellation or refund probability and optimize the intervention threshold for net profit.

---

## Key Results: Before vs. After AI (Validation Slice)

| Metric | Before AI (Unfiltered COD) | After AI (Threshold = 0.22) | Impact |
| :--- | :---: | :---: | :--- |
| Checkout Policy | Ship all COD orders unchecked | Flag orders with risk >= 0.22 for prepaid nudge or OTP verification | Proactive risk filtering |
| Logistics Loss | PKR 23.5M | PKR 6.3M | -PKR 17.2M (-73%) |
| Net Profit | PKR 3,499,494 | PKR 12,954,085 (~PKR 13.0M) | +PKR 9,454,591 (+270%) |

> Business assumptions: 12% platform commission on completed orders, PKR 450 two-way reverse logistics loss per failed or returned order, 0.25 residual cost factor on flagged cancellations, and 0.70 retention factor when risky orders convert to prepaid or pass verification.

---

## Methodology & Modeling Summary

1. **Exploratory Data Analysis (EDA):**
   - Filtered 1,048,575 raw records down to 494,463 completed, canceled, or refunded orders.
   - Observed a 52.74% baseline cancellation or refund rate (260,780 failed orders vs. 47.26% completed).
2. **Feature Engineering:**
   - Pricing and discounts: `order_value`, `qty`, `unit_price_est`, `discount_amt`, `discount_ratio`
   - Payment and channel: `is_cod` (separating Cash on Delivery from prepaid wallets or cards)
   - Temporal and category signals: `order_hour`, `order_dow`, `is_late_night` (11 PM to 7 AM), `cat_code`
3. **Model Performance (LGBMClassifier):**
   - Validation ROC-AUC: `0.8012`
   - Validation PR-AUC: `0.8273`
   - Top risk drivers: `order_value`, `unit_price_est`, `cat_code`, and `discount_ratio`
4. **Profit-Maximizing Cutoff:**
   - Swept decision thresholds from `0.15` to `0.85` and selected `0.22` as the cutoff maximizing net PKR profit.

---

## Repository Structure

```text
├── README.md                 # Project summary, methodology, and Before vs. After impact
├── ayeshanoor05.ipynb        # End-to-end data loading, feature engineering, LightGBM training, and profit simulation
├── ayeshanoor05.pdf          # 12-slide executive presentation deck
└── charts/                   # Exported visualizations 
    ├── status_donut.png
    ├── feature_trends_4panel.png
    ├── top_features.png
    ├── risk_distribution.png
    └── profit_simulation.png
