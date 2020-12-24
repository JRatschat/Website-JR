---
date: "2020-01-27T00:00:00Z"
external_link: ""
image:
  caption: "Photo by [Christopher Burns](https://unsplash.com/@christopher__burns) on [Unsplash](https://unsplash.com/photos/Kj2SaNHG-hg)"
# links:
# - icon: Medium
#   icon_pack: fab
#   name: Medium
#   url: https://medium.com/@j.ratschat
slides: ""
summary: I predicted and interpreted cross-selling purchase probabilities using XGBoost and SHAP values in R.
categories: ["Data Science", "Research"]
tags:
- Research
- Data Science
title: Boosting Gradient Boosting Interpretability
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
---

## About the paper
This research paper was written in summer 2020 within the Master's seminar *Data Mining in Marketing: Data Driven Customer Analytics with Machine Learning*. In partial fulfillment of the requirements of the seminar, I predicted and interpreted cross-selling purchase probabilities using XGBoost and SHAP values in R. Read the full paper here: [Boosting Gradient Boosting Interpretability: Predicting and Interpreting Cross-Selling Purchase Probabilities of a Large German Savings Bank](https://github.com/JRatschat/Boosting-Gradient-Boosting-Interpretability/blob/master/Boosting_Gradient_Boosting_Interpretability.pdf).

## Introduction and Findings
Powerful analytical methods lead to more efficient and effective data-driven marketing (Wedel & Kannan 2016). Especially machine learning has become popular in this field due to the high predictiveness of its algorithms. A challenge is, however, that complex machine learning models are generally black-box models. Hence, data goes in, and results come out, but it is unknown or hidden to its users how these models came up with its results. Increasing interpretability is vital because it grows users' confidence and trust in machine learning models. Users do not adopt models that fail to do so (Ribeiro et al. 2016). Also, enhanced interpretability extends the knowledge derived from models. Therefore, eliminating the tradeoff between a model's accuracy and a model's interpretability has gained many researchers' attention (Ribeiro et al. 2016, Lundberg & Lee 2017, Chen et al. 2018, Lipton 2018).

In this paper, I use a data set from a large German savings bank to predict cross-selling purchase probabilities and decisions in the customer base. This paper aims to (1) to accurately predict whether an already existing customer will open a checking account and (2) to explore which effect the features have on the prediction to enhance the interpretability of the model. The paper leverages one of the leading gradient boosting algorithms, namely XGBoost, to reach these goals. It has been used with great success in many machine learning and data mining challenges (Chen & Guestrin 2016). Moreover, I implement SHapley Additive exPlanations (SHAP) values to tackle the lack of boosted trees' interpretability (Friedman 2001).

Regarding the first research question, a hyperparameter-tuned XGBoost model's predictive accuracy proves superior compared to a benchmark logit model. A disadvantage, however, is that more complex models are more computationally expensive than simple models. Concerning the second research question, it becomes clear that SHAP values enable its users to critically examine complex models and understand how dependent variables were predicted. Through this method, users gain further knowledge about the importance, extent, and direction of feature variables on the target variable. Although causal statements cannot be made through this approach, it still helps users gain trust in the model, find ways to improve the model and get a new understanding of the data. When using ordinary feature impact tools, this would not be feasible to such an extent. 

The analysis unfolds that this so-called black-box model applies among other trends discovered in the research of RFM-models (Bauer 1988, Miglautsch 2000). For example, customers that have recently acquired another product have a higher predictive value of opening a checking account than customers who have not. Also, the more active customers are (as measured by logins), the higher is the prediction value. Other trends found in the data are that younger customers exhibit a higher prediction probability than older customers and that checking account ads always lead to a positive effect on the prediction, although varying. 

The paper's main conclusion is that XGBoost models have their place in practice for predicting cross-selling purchase probabilities and decisions. One of the most significant disadvantages - lack of interpretability - can be mitigated with SHAP values that greatly expand the transparency, explainability, interpretability of complex tree-based models.
