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
In this paper, a dataset from a large German savings bank is used to predict crossselling purchase probabilities and decisions in the customer base. The goals of this paper are (1) to accurately predict whether an already existing customer will open a checking account and (2) to explore which effect the features have on the prediction to enhance the interpretability of the model. To reach these goals, the paper leverages one of the leading gradient boosting algorithms there is, namely XGBoost. It has been used with great success on many machine learning and data mining challenges. Among the advantages of XGBoost are that its models are easily scalable, tend to avoid overfitting, and can be used for a wide range of problems (Chen & Guestrin 2016, p. 785-786). To tackle the lack of easy interpretability of boosted trees (Friedman 2001, p. 1229–1230), this paper implements SHapley Additive exPlanations (SHAP) values to explain the output of the XGBoost model.

When accuracy is the main goals, then one should use more complex models. Although tuning an XGBoost model is computationally expensive, once it is tuned, it becomes computationally cheap. It becomes apparent that such prediction systems have their place in marketing analytics departments. Instead of only predicting the likelihood of buying a checking account, marketing departments could extend this approach to any product offering to compare purchasing probabilities and target customers with the products for which they have the highest probability to buy. Through this strategy, it could become feasible to target the right customers and to ultimately increase profitability. For this concept to be successful, several additional topics need to be addressed. One has to assess which customers have the potential to be profitable while excluding customers that lead to losses (Shah et al. 2012). Also, one must answer the question when and where a customer should be targeted.

Generally, SHAP values enable its users to critically examine complex models and to understand how dependent variables were predicted. Through this method, users gain further knowledge about importance, extent and direction of feature variables on the target variable. Although causal statements cannot be made through this approach, it still helps users to gain trust in the model, to find ways of improving the model, and to get a new understanding of the data. Therefore, this enhanced interpretability should increase user adoption. When only using tools from the xgboost package, this would not be feasible to such an extent. The trust of the analysis through SHAP values is increased because suggested underlying trends of the data have been amongst others discovered by RFM-models (Bauer 1988, Miglautsch 2000). Customers that have recently acquired another product have an higher prediction value than customers that have not bought another product for a longer period of time. Also, the more active customers are (as measured by logins), the higher is the prediction that these customers cross-buy a checking account. Other important trends found in the data are that younger customers exhibit an higher prediction probability than older customers and that checking account ads always lead to a positive effect on the prediction, although with varying extent. This paper shows that giro_mailing does lead to a negative interaction effect on the prediction when appearing with younger or recent customers. This could indicate that the model finds autoresponse within the group of younger or recent customers and punishes this effect with a negative interaction value.

XGBoost models should be used in practice for predicting cross-selling purchase probabilities and decisions. One of the biggest disadvantages - lack of interpretability - can be mitigated through the use of SHAP values that greatly expand the transparency, explainability, interpretability of complex tree-based models.