# 逻辑回归模型-客户流失预警系统 

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)

## 项目简介

本项目基于**逻辑回归算法**构建了一个电信客户流失预警模型。通过对客户基本信息、服务使用情况、合同与账单信息等多维度数据进行分析，预测客户流失风险，帮助企业提前采取挽留措施，降低客户流失率。

## 数据集说明

本项目使用电信客户流失数据集（Telco Customer Churn）`https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data`，
包含 **7,043 位客户**的 **21 个特征字段**：

### 特征分类

| 类别 | 字段 |
|------|------|
| **客户基本信息** | 客户ID、性别、老年公民、伴侣、家属 |
| **服务使用情况** | 在网时长、电话服务、多条线路、互联网服务、在线安全、在线备份、设备保护、技术支持、流媒体电视、流媒体电影 |
| **合同与账单** | 合同、无纸化账单、支付方式、月费、总费用 |
| **目标变量** | 客户流失（Yes/No） |

### 数据特点

- **数据集规模**：7,043 条记录，21 个字段
- **流失率**：约 26.5%（1,869 位客户已流失）
- **数据质量**：存在少量缺失值（总费用列），已进行处理

## 技术栈

- **Python 3.11+**
- **pandas** - 数据处理与分析
- **numpy** - 数值计算
- **matplotlib / seaborn** - 数据可视化
- **scikit-learn** - 机器学习建模

## 项目结构

```
.
├── README.md                              # 项目说明文档
├── customer_churn_prediction.ipynb        # Jupyter Notebook 主程序
├── churn_prediction_model.pkl             # 模型信息
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # 数据集
└── scaler.pkl                             # 模型信息
```

## 核心功能

### 1. 数据预处理

- **列名清理**：去除列名中的前后空格
- **缺失值处理**：将总费用转换为数值类型，删除包含缺失值的记录
- **标签编码**：将客户流失（Yes/No）转换为数值（1/0）
- **One-Hot 编码**：对 15 个分类变量进行编码，生成 30 个特征

### 2. 模型构建

- **算法**：逻辑回归（Logistic Regression）
- **类别平衡**：使用 `class_weight='balanced'` 处理类别不平衡问题
- **数据划分**：训练集 80%（5,625条），测试集 20%（1,407条）
- **特征标准化**：使用 StandardScaler 进行标准化

### 3. 模型评估

| 指标 | 数值 |
|------|------|
| **准确率 (Accuracy)** | 72.64% |
| **精确率 (Precision)** | 49.09% |
| **召回率 (Recall)** | 79.68% |
| **F1 分数 (F1-Score)** | 60.75% |
| **AUC-ROC** | **83.50%** |

### 4. 风险等级划分

根据预测概率将客户分为三个风险等级：

| 风险等级 | 流失概率 | 建议措施 |
|----------|----------|----------|
| **高风险** | ≥ 70% | 立即采取挽留措施，提供专属优惠 |
| **中风险** | 40% - 70% | 定期跟进，提升服务满意度 |
| **低风险** | < 40% | 维持正常服务，定期关怀 |

### 5. 可视化分析

- 客户流失分布图
- 混淆矩阵热力图
- ROC 曲线
- Top 15 重要特征分析
- 风险等级分布图

## 使用方法

```bash
jupyter notebook customer_churn_prediction.ipynb
```

按顺序运行各个代码单元格，可以交互式地查看数据分析过程和结果。

## 关键发现

1. **高召回率**：模型能够识别出 79.68% 的潜在流失客户，适合用于预警场景
2. **AUC-ROC 达 83.50%**：模型具有良好的区分能力
3. **关键影响因素**：
   - 合同类型（月付客户更容易流失）
   - 在网时长（新用户流失风险更高）
   - 互联网服务类型（光纤用户流失率较高）
   - 支付方式（电子支票用户风险较高）

## 业务建议

1. **重点关注高风险客户**：针对预测为高风险的客户，主动提供挽留方案
2. **优化合同策略**：鼓励客户选择长期合同，降低月付比例
3. **提升新用户体验**：加强入网前3个月的服务跟进
4. **增值服务推广**：推广在线安全、技术支持等附加服务，提高客户粘性

## 模型保存与加载

```python
import joblib

# 保存模型
joblib.dump(model, 'churn_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 加载模型
loaded_model = joblib.load('churn_prediction_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
```

## 贡献与反馈

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 许可证

本项目采用 MIT 许可证。

---

## 联系方式

**谙弆悕博士（Ailan Anjuxi）**

- 邮箱：anjuxi.ME@outlook.com
- SIP电话：sip:anjuxi@sip.linphone.org

---

*本项目为数据科学与机器学习教学演示用途，数据集来源于公开数据源。*

