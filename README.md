# A Deep Learning Approach to Predict Recanalization First-Pass Effect following Mechanical Thrombectomy in Patients with Acute Ischemic Stroke

[Haoyue Zhang](https://github.com/zhanghaoyue), [Jennifer S. Polson](https://github.com/jspolson), Zichen Wang, Kambiz Nael, Neal M. Rao, William F. Speier and Corey W. Arnold

[Click here to read the paper](https://www.ajnr.org/content/45/8/1044.long)

## Abstract
BACKGROUND AND PURPOSE: Following endovascular thrombectomy in patients with large-vessel occlusion stroke, successful recanalization from 1 attempt, known as the first-pass effect, has correlated favorably with long-term outcomes. Pretreatment imaging may contain information that can be used to predict the first-pass effect. Recently, applications of machine learning models have shown promising results in predicting recanalization outcomes, albeit requiring manual segmentation. In this study, we sought to construct completely automated methods using deep learning to predict the first-pass effect from pretreatment CT and MR imaging.

MATERIALS AND METHODS: Our models were developed and evaluated using a cohort of 326 patients who underwent endovascular thrombectomy at UCLA Ronald Reagan Medical Center from 2014 to 2021. We designed a hybrid transformer model with nonlocal and cross-attention modules to predict the first-pass effect on MR imaging and CT series.

RESULTS: The proposed method achieved a mean 0.8506 (SD, 0.0712) for cross-validation receiver operating characteristic area under the curve (ROC-AUC) on MR imaging and 0.8719 (SD, 0.0831) for cross-validation ROC-AUC on CT. When evaluated on the prospective test sets, our proposed model achieved a mean ROC-AUC of 0.7967 (SD, 0.0335) with a mean sensitivity of 0.7286 (SD, 0.1849) and specificity of 0.8462 (SD, 0.1216) for MR imaging and a mean ROC-AUC of 0.8051 (SD, 0.0377) with a mean sensitivity of 0.8615 (SD, 0.1131) and specificity 0.7500 (SD, 0.1054) for CT, respectively, representing the first classification of the first-pass effect from MR imaging alone and the first automated first-pass effect classification method in CT.

CONCLUSIONS: Results illustrate that both nonperfusion MR imaging and CT from admission contain signals that can predict a successful first-pass effect following endovascular thrombectomy using our deep learning methods without requiring time-intensive manual segmentation.

Framework illustration:
![image]![F3 large](https://github.com/user-attachments/assets/b48a9360-d165-40a3-bf23-73ec81ba8c13)


For MRI preprocessing pipeline, please refer to this repository from our previous work:\
https://github.com/zhanghaoyue/stroke_preprocessing

## Citation
If you find our work useful, please consider citing:
```
@article{zhang2024deep,
  title={A Deep Learning Approach to Predict Recanalization First-Pass Effect following Mechanical Thrombectomy in Patients with Acute Ischemic Stroke},
  author={Zhang, Haoyue and Polson, Jennifer S and Wang, Zichen and Nael, Kambiz and Rao, Neal M and Speier, William F and Arnold, Corey W},
  journal={American Journal of Neuroradiology},
  year={2024},
  publisher={Am Soc Neuroradiology}
}
```


### note

The model code is under /model/model.py. The corresponding final model is res25.
Others are my comparison models.

You can easily load the model to your own code framework. Data loader is an example for you to modify your data loader. 


