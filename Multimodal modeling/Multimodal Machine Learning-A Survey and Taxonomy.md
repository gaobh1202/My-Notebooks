# Multimodal Machine Learning: A Survey and Taxonomy

## Abstract:

A research problem is characterized as multimodal when it includes multiple such modalities.

将多模态机器学习面临的主要问题划分为：representation, translation, alignment, fusion, and co-learning.

## 1. Introduction

- **Representation:** 

  How to represent and summarize multimodal data in a way that exploits the **complementarity** and **redundancy** of multiple modalities.

  如何在充分利用模态互补性以及冗余性的前提下，学习到有效的多模态表征。（*这里的表征是指单一模态的表征还是fusion后的表征？？*）

- **Translation:**

  How to translate (map) data from one modality to another.

  一种模态到另一种模态数据的映射可能是不确定的，比如针对一副图像，并没有标准确定准确的语言描述。
  
- **Alignment:** 

  How to identify the direct relations between (sub)elements from two or more different modalities.

  如何将不同模态下的具体数据进行对应？比如，一个文本的菜单，将各步骤对应到视频的不同部分。

- **Fusion:**

  Join information from two or more modalities to perform a prediction.

  不同模态的数据具有不同的预测能力与噪音

- **Co-learning:**

  Transfer knowledge between modalities, their representation, and their predictive models.

  例如：***co-learning***（如何利用从一个模态学习到的知识到另一个数据有限的模态的训练）

  zero-shot learning.

## 2. Application:

- **Audio-Visual speech recognition：**

  结合视觉信息来提升语音识别的效果。实验说明，视觉信息最大的优点体现在语音信号是noisy的。

  *the captured interactions between modalities were **supplementary** rather than **complementary*** 

  多模态的信息是提升模型的稳健性，而不是提高识别效果

- **Multimedia content indexing and retrieval**

  区别于传统基于关键字的多媒体视频索引，新方法基于内容直接对多媒体索引。

  该领域衍生出一些子问题，如*automatic shot-boundary detection*和*video summarization*

  相关的数据集为*multimedia event detection (MED) task*

- **Multimodal interaction:** understand human multi-modal behaviors during social interactions

  该领域旨在对人的情绪、行为等进行识别

- Media description:

  - image captioning：为输入图像生成文本描述。

    一个重要挑战是，如何评估生成的描述的质量？

  - Visual question-answering：回答关于图像的问题。

## 3. Multimodal Representation:

**多模态数据表示面临的问题**

- 如何结合来自异构数据源的数据
- 如何处理不同程度的噪声
- 如何处理丢失的数据

**多模态数据表示应该具有的特点**

- 特征空间的相似性能够反映概念之间的相似性
- 当有模态缺失时，该表示也能够容易获得
- 给定观察数据，能够填补缺失的模态

**将多模态数据划分为两类：** **joint** and **coordinated**

- joint representation：将单一模态数据映射到相同的特征空间
- coordinated representation：分别处理不同模态的数据，但在过程中施加相似性的约束，从而使不同模态的表示处于合作的空间（coordinate space）

![image-20221031170637898](C:\Users\Administrator\Desktop\Notebooks\Paper Reading\Multimodal modeling\image1.png)

### 3.1 Joint Representation

联合表示大多用在训练和推理阶段都有多模态数据的任务中。最简单的例子就是concatenation，也被称作early fusion

#### **Neural** **Networks**

神经网络的每一层被假设能够以更抽象的方式来表示数据，因此，通常选择模型的最后一层，或倒数第二层作为数据的表示。而采用神经网络实现多模态表示的做法通常是，先对不同模态利用网络层映射到一个联合空间，再将这个联合表示通过网络层进行映射，或直接用于预测。

由于神经网络的训练需要大量标记数据，因此一个常见的作法是利用自编码器在无监督数据上进行预训练。

Ref. 151使用堆叠的去噪自编码器来表示每个模态，再用另一个自编码器得到多模态表示

优点是：神经网络具有出色的性能，并且可以再无监督数据集上进行预训练

缺点是：无法自然地处理丢失数据，神经网络难训练。

#### Probabilistic graphical models

使用潜在随机变量来构建数据表示。最常用的方法是深度玻尔兹曼机，由受限玻尔兹曼机堆叠而成

图模型的好处是不需要使用监督数据

Ref. 104为每个模态构建DBN，再将其组合为联合表示。

多模态DBMs能够通过使用隐藏单元的二进制层合并两个或多个无向图来从多个模态中学习联合表示。

***由于模型的无定向性质，它们允许每个模态的低级表示在联合训练后相互影响。***

使用multimodal DBMs来学习多模态表示的好处是：便于处理丢失数据。并且可以用于生成某一模态的样本，即使是在缺失其他模态的情况下。训练时也可以采用无监督的训练方式

缺点是难训练

#### Sequence Representation

用于表示如句子，视频，音频等可变长度的序列数据

RNN和LSTM等模型展现了较好的效果，RNN的隐藏状态可以被视作数据的表示

### 3.2 Coordinated Representation

不是将模态投影到一个联合空间中，而是为每个模态学习单独的表示形式，并通过约束来协调（coordinate）它们。

#### Similarity models

相似性模型最小化协作空间中不同模态之间的距离

Ref. 221&222构建了图像与注释之间的协作空间，根据图像和文本的特征来构建线型图，从而使得相关联的多模态样本之间的内积更高（对应余弦距离更小）

神经网络模型也被广泛用于构建*协作表示*。比如：使用相似内积和ranking损失来构建深度视觉语义嵌入（DeVISE）；利用<subject, verb, object>成分语言模型和深度视频模型来构建视频和句子之间的协作空间，用于视频描述等任务

#### Structured coordinated space models

根据任务类型不同（如跨模态检索、图像字幕等）施加不同的限制

结构化协作空间常用于跨模态hashing，即将高维数据压缩至二进制代码，并且相同对象具有相似的code，常用于跨模态检索领域

另一个应用是图像与语言的order-embedding

结构化协调空间的一种特殊情况是基于规范相关分析 (CCA) 的情况。CCA计算线性投影，该线性投影最大化了两个随机变量 (在我们的情况下为模态) 之间的相关性，并增强了新空间的正交性。常用于跨模态检索。深度CCA作为核CCA的替代方案，解决了可伸缩性的问题（随数据规模变化而变）

CCA, KCCA, DCCA通常只捕捉模态之间共享的内容；深度规范相关自动编码器，由于包括了自编码器，所以能够捕捉模态特有的特征；语义相关性最大化方法 [248] 还鼓励语义相关性，同时保留相关性最大化和所得空间的正交性-这导致了CCA和跨模态散列技术的结合

### 3.3 Discussion

联合表示适用于在训练和推理阶段都有多种模态数据的任务，并被扩展到多种模态的数据表示（>2）

协作表示适用于推理阶段只有单一模态的任务（如图文检索、翻译），并且大多只限制于两种模态

## 4.Translation

从一个模态映射到另一个模态，其中一个典型的问题就是“图像与视频字幕”，不仅需要充分理解图像内容、识别显著部分，还需要生成符合语法规则的描述。

将多模态翻译任务分为example-based和generative两类，其中基于示例的方法依赖于字典，而生成类方法依赖于模型。（NeRF类似于生成式方法）

### 4.1 Example-based

根据其训练数据（字典）的不同，区分为retrieval-based和combination-based

##### Retrieval-based models:

在字典中查询最接近的样本作为翻译结果，查询过程可以发生在unimodal空间或中间语义空间。

**Unimodal retrieval：**可以应用于视觉-语音合成，图像描述；优点是只需要单一模态的表示，缺点是通常需要额外的步骤（如re-ranking），并且单一模态的相似性不能保证翻译的结果。

**Intermediate semantic：**将不同模态的数据映射到一个语义空间（类似于协调表示），再在该空间内进行检索；优点是可以进行双边翻译，但缺点是依赖于语义空间的好坏

##### Combination-based models

通过组合一系列相似的样本来获得翻译结果

#### 4.2 Generative approaches

可以划分为grammar-based models, encoder-decoder, continuous generation models

grammar-based：设置一定的规则来固定结果的结构

encoder-decoder：

## 5.Alignment

multimodal alignment：寻找多个模态的实例的各子成分之间的关系和关联。比如在图像和描述之间，寻找图像的某个区域和描述的某个部分之间的对应关系

可以分为显式和隐式，隐式通常作为其他任务的一个子过程

Multimodal alignment存在的问题：1) 很少有带有显式注释对齐的数据集; 2) 很难设计模态之间的相似性度量; 3) 可能存在多种可能的对齐方式，并且并非一种模态中的所有元素在另一种模态中都具有对应关系

## 6.Fusion

多模态融合的优点：1. 在多种模态中观察同一现象可能获得更稳健的预测；2. 多模态可能捕获到互补信息；3. 当单一模态缺失时，多模态系统仍然可以运行

将多模态融合方法分为model-agnostic(不依赖特定的ML模型) & model-based

#### 6.1 Model-agnostic approaches

可以分为：early (i.e., feature-based), late (i.e., decision-based) and hybrid fusion

early fusion是特征层的融合，利用各模态浅层特征之间的相关性和相互作用

late fusion是决策层的融合，利用平均、投票等方法，该类型方法更灵活，并且在模态缺失时也能工作

#### 6.2 Model-based approaches

**Multiple kernel learning:**

多核学习是核支持向量机的扩展，针对不同模态采取不同的核。多核学习的误差函数是凸的，因此可以采用一般的优化方法，并且既可以用于回归也可以用于分类；缺点是推理时间长、内存占用大

**Graphical models:**

大多数图模型可以分成两类：生成式——建模联合概率密度；判别式——建模条件概率密度

一些常见工作如：隐马尔科夫模型，条件随机场

图模型的好处是能够利用数据的时空结构，适合于时序问题，并且可以融入专家知识

**Neural Networks：**

好处是能够处理大量的数据，性能好，可以端到端地训练，能够学习到更复杂的决策边界；

缺点是缺乏可解释性

#### 6.3 Discussion:

存在的问题：

- 信号可能不是时序对齐的（比如一个是连续的信号，一个是离散的事件）
- 难以建立模型来利用补充信息（supplementary information），而不是仅利用互补信息（complementary information）
- 可能存在不同类型、不同level的噪音

## 7.Co-learning

联合学习常用于多模态学习中，从一种resource rich数据源中学习知识来帮助resouce poor数据的建模（比如缺少标注数据，输入有噪声，标签不可信等）

将Co-learning分为parallel, non-parallel, hybrid

Parallel方法要求不同模态的训练数据来自同一实体（比如在audio-visual recognition中video和speech都来自同一个人）

Non-parallel通过使用类别间的重叠来实现co-learning

Hybrid方法中模态间通过共享模态或数据集进行桥接

![image-20221110105329165](C:\Users\Administrator\Desktop\Notebooks\Paper Reading\Multimodal modeling\image-2.png)

#### 7.1 Parallel data

可以分为两种co-training和representation learning

Co-training：创造更多的有标签样本

Transfer learning：多模态深度玻尔兹曼机和多模态自编码器等能够在不同模态的表示之间迁移信息

#### 7.2 Non parallel data

非并行数据不要求模态之间属于同一实体，只要共享类别或概念。能够帮助更好地理解语义，甚至实现对unseen object的检测

**Transfer learning**：迁移学习通常和多模态协同表示一起使用。比如**Ref.61**通过将CNN视觉特征与在单独的大型数据集上训练的word2vec文本 [141] 协调，使用文本来改善图像分类的视觉表示。

**Conceptual *grounding***：指的是学习语义含义或概念，不仅基于语言，还基于视觉，声音甚至气味等其他形式。grounding通常是寻找特征之间的共同潜在空间或分别学习单一模态表示再concatenating

**Zero-shot learning**：单模态ZSL关注物体的组成部分或属性，从而预测unseen的目标；多模态ZSL借助辅助模态来帮助首要模态进行识别，但与non-parallel有冲突。

#### 7.3 Hybrid data

Hybrid模式下，两个非并行模态由共享模态或数据集进行桥接
