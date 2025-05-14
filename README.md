# collections of practices
 平时做的练习合集

### 01-客户流失预测系统 – IBM Telco 数据集

- 使用 Pandas/EDA/特征工程处理原始客户数据，构建了多维度特征（用户粘性、服务活跃度等）

- 训练了逻辑回归、决策树、随机森林、XGBoost、LightGBM 等模型，LightGBM 获得最佳性能（AUC ≈ 0.85）

- 引入 SHAP 可解释性方法，识别出“合约类型”、“月费用”、“家庭情况”等关键流失因子

- 可视化混淆矩阵与 ROC 曲线，设计多模型对比表格并生成业务汇报材料

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, LightGBM, XGBoost, SHAP, EDA, Classification, Model Interpretability, Feature Engineering, ROC Curve, AUC, Business Analysis



### 02-Amazon商品数据，多模态特征融合

构建了一个简单的基于图神经网络（GNN）的多模态推荐系统，融合了用户评论文本特征、商品图像特征以及用户-商品交互信息，用于完成**评分预测**和**Top-K 推荐任务**。

- 文本特征：提取用户评论的 BERT 语义向量（768维）

- 图像特征：提取商品图片的 ResNet 视觉向量（2048维）

- 用户特征：初始化为零向量，参与 GNN 表达学习



### 03-销售预测系统 M5 Competition Subset (DS Focus)

选取 M5 竞赛数据中 `dept_id=FOODS_1` 的子集（随机抽取10类商品），基于历史销量数据，构建销售预测系统。

完成包含滞后特征、滚动均值、傅里叶季节性、价格变动、节日窗口、SNAP交互等特征的工程构建。

对比建模了 ARIMA、Facebook Prophet、LightGBM（Optuna调参）、GRU神经网络等模型，评估模型性能差异。

使用 PyTorch 构建 GRU 模型，捕捉时间序列的长短期依赖关系，整体在部分商品上超过传统模型表现。



### 04-一个超级简单的Finetuning



### 05-uplifting model + 因果推断（ATE+CATE）

基于 Hillstrom Email Marketing Campaign 数据集，构建和评估多种 Uplift 模型，用于量化营销邮件对客户转化行为的因果影响。该任务不仅涉及传统的预测建模，还深入探讨了因果推断方法在营销场景中的应用。

- 数据预处理与EDA：包括异常值处理、特征工程与分组策略
- Uplift 模型结构：
  - 经典方法：S-learner, T-learner, X-learner, DR-learner, CausalForest
  - 深度模型：PyTorch 自定义 TARNet 网络（无 TensorFlow 依赖）
- 指标评估：
  - AUUC（Area Under Uplift Curve）
  - Qini 曲线下的面积 AUQC
  - Lift@K（排名前 30% uplift 得分人群的提升率）
  - KRCC（Kendall Rank Correlation Coefficient）

SKILLS

- Python, Pandas, Scikit-learn, EconML, CausalML
- PyTorch（深度 Uplift 网络实现）
- 数据可视化：Matplotlib, Seaborn, uplift-curves

