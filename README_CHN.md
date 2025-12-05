# LIDC-IDRI Patch Research

基于 ConceptCLIP 的可解释性肺结节恶性度预测研究

## 项目简介

本项目探索使用 **ConceptCLIP**（一种针对生物医学领域的视觉-语言模型）进行肺结节良恶性分类。核心思路是：

1. **零样本特征提取**：使用预训练的 ConceptCLIP 模型，无需微调
2. **概念驱动**：定义放射学概念（如毛刺征、分叶、钙化等），计算图像与概念的相似度分数
3. **可解释性**：使用简单的逻辑回归模型，通过特征系数量化每个概念对预测的贡献

### 研究问题

- **RQ1**: 如何设计有效的放射学概念以提升下游分类性能？
- **RQ2**: ConceptCLIP 学习到的概念表示是否具有临床可解释性？

### 主要结果

- 20 个概念配置达到最优性能：**AUC=0.794, Accuracy=0.730**
- 通过 L1 正则化特征选择识别出 **12 个统计显著的概念特征**
- 所有显著特征均与临床知识一致

## 数据集

使用 **LIDC-IDRI** 数据集构建结节级 2D patch 数据集：

- **结节数量**: 678 (恶性: 397, 良性: 281)
- **患者数量**: 440
- **图像数量**: 2532 patches

数据集构建代码：[LIDC-IDRI_patch_generation](https://github.com/aquamarineaqua/LIDC-IDRI_patch_generation)

## 项目结构

```
LIDC-IDRI_patch_research/
├── 1_Database_and_Dataloader_create.ipynb  # 数据预处理与特征提取
├── 2_Read_database.ipynb                    # 读取数据库与基础分类
├── 3_For_20_concept.ipynb                   # 20概念实验
├── 4_For_30_concept.ipynb                   # 30概念实验
├── 5_For_image_embeddings.ipynb             # 图像嵌入基线实验
├── 6_For_20_concept_L1_feature_selection.ipynb  # L1特征选择与可解释性分析
└── datasets/                                # 数据集文件夹
    └── curation2/lidc_patches_all/          # patch 图像数据（请自行解压压缩文件）
```

## 安装依赖

### Python 版本
- Python >= 3.10

### 必需库

```bash
pip install torch torchvision
pip install transformers
pip install huggingface_hub
pip install pandas numpy
pip install matplotlib
pip install scikit-learn
pip install statsmodels
pip install h5py
pip install tqdm
pip install pillow
pip install seaborn
pip install jupyter
```

或使用以下命令一次性安装：

```bash
pip install torch torchvision transformers huggingface_hub pandas numpy matplotlib scikit-learn statsmodels h5py tqdm pillow seaborn jupyter
```

## 使用指南

### 1. 数据准备

确保 `datasets/curation2/lidc_patches_all/` 目录下包含：
- `all_patches_metadata.csv`: 元数据文件
- 各患者文件夹（如 `LIDC-IDRI-0001/`）: 包含 patch 图像

**注意**：首次运行需要登录 HuggingFace：
```python
from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")
```

### 2. Notebook 功能说明

| Notebook | 功能描述 |
|----------|----------|
| `1_Database_and_Dataloader_create.ipynb` | **数据预处理与特征提取流水线**：筛选面积≥50mm²的结节，二分类标签生成，加载ConceptCLIP提取图像/文本特征，存储到HDF5数据库 |
| `2_Read_database.ipynb` | **基础分类实验（10概念）**：读取HDF5特征，Prompt Ensembling，计算概念分数，结节级聚合，5折交叉验证逻辑回归 |
| `3_For_20_concept.ipynb` | **20概念实验**：定义20个放射学概念类别（98个sub-concept），生成文本嵌入，图像-概念相似度计算，分类评估 |
| `4_For_30_concept.ipynb` | **30概念实验**：扩展到30个概念类别（145个sub-concept），对比不同概念数量的分类性能 |
| `5_For_image_embeddings.ipynb` | **Baseline实验**：使用纯图像嵌入（Mean-Max拼接，2304维）进行分类，对比Logistic Regression、SVM、XGBoost |
| `6_For_20_concept_L1_feature_selection.ipynb` | **特征选择与可解释性分析**：LASSO特征选择，统计显著性检验（p-value），Bootstrap稳定性评估，Case Study可视化 |

### 3. 推荐运行流程

```
Step 1: 数据预处理
└── 1_Database_and_Dataloader_create.ipynb
    ├── 输出: curated_metadata.csv
    └── 输出: conceptclip_features.h5

Step 2: 概念实验
├── 2_Read_database.ipynb（10概念）
├── 4_For_30_concept.ipynb
└── 3_For_20_concept.ipynb

Step 3: 可解释性分析
└── 6_For_20_concept_L1_feature_selection.ipynb
    ├── LASSO 特征选择
    ├── 统计检验
    └── Case Study 可视化

（可选）Baseline 对比
└── 5_For_image_embeddings.ipynb
```

### 4. 输出文件说明

| 文件 | 描述 |
|------|------|
| `curated_metadata.csv` | 筛选后的元数据 |
| `conceptclip_features.h5` | 原始10概念特征数据库 |
| `conceptclip_features_20.h5` | 20概念特征数据库 |
| `conceptclip_features_30.h5` | 30概念特征数据库 |
| `image_features/df_image_features.csv` | 图像嵌入特征 |
| `image_features/df_nodule_features_concept20_minmax.csv` | 结节级概念分数 |
