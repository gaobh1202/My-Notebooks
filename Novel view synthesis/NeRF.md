# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

# 前言：

NeRF是Neural Radiance Fields（神经辐射场）的缩写。

其中，Radiance Field的本质为一映射函数，在该论文中表示将**输入**（三维空间坐标，视角）映射到**输出**（volume density，RGB颜色）
$$
g_{\theta}(\mathbf{x}, \mathbf{d})=(\sigma, \mathbf{c})
$$
如此一来，该映射函数即Radiance Field相当于原三维空间的一种**隐式表示**。因为通过该函数，能够获得一个给定坐标点的体密度与RGB，在通过渲染方法即可生成2D图像。

隐式表示是相较点云、体素等模型的，该论文中通过MLP来拟合~~该映射函数~~三维场景，因此不存在类似点云一样的三维模型，所以称作隐式表示。

传统方法是根据输入的不同视角图片先重建出场景的三维表示，再渲染得到特定视角的图像。

**如何理解文章中提到的camera ray？**

**拍摄到的是各种视角下的二维图像，如何转换成输入的坐标和视角？**

# Abstract

论文提出一种通过优化底层连续体积场景函数（underlying continuous volumetric scene function）来生成新视角图像的方法。

通过MLP来表示三维场景，输入为空间坐标和视角，输出为体密度（volume density）和emitted radiance

# 1. Introduction:

将一个静态的场景表示为一个连续的5-D函数，函数的输出为三维场景中每个点*（由输入x, y, z体现）*在每个方向下*（由输入的视角体现）*的radiance emitted（实际是RGB）和volume density*（理解为不透明度）*

为了生成一个特定视角下的图像，该论文共分为三步：

- 利用穿过场景的camera ray来生成样本集，即3D点
- 训练一个神经网络，来反映（样本点+视角）到（RGB+密度）
- 利用体渲染方法，将这些RGB+密度渲染成2D图像。

利用实际观察到的图像与通过渲染生成的图像来作为训练的误差，采用梯度下降的训练方式。

这种基本的实现方式不能获得一个有效的高分辨率表征，并且在样本的要求上是低效的。本文通过将基本的5D输入转换成位置编码的方式来提高MLP的特征表达能力。*（transforming input 5D coordinates with a positional encoding）*，并提出一种分层采样方法来减少所需样本数量。

本文的**主要贡献：**

- 提出一种神经辐射场方法，来高精度的表示一个连续的场景，并通过MLP对其参数化
- 基于体渲染方法，来优化RGB图像的表征。其中包括分层采样方法来更好地分配MLP的空间。
- 一种位置编码方式，来将5D坐标映射到更高维空间，从而更好的优化MLP

![image-20220911160426947](C:\Users\hp\Desktop\Notebooks\Paper Reading\Novel view synthesis\image-1.png)

随机采样100个input view

# 2. Related work

近年来一个有前景的研究方向是直接用MLP的权重来编码物体object或场景scene，即直接学习一个从空间坐标到implicit representation的映射。

然而与三角网格、体素网格等离散化方法相比，该类方法难以重现具有复杂几何形状的场景。

NeRF的工作主要包括基于神经网络的重建，和基于volume density的二维图像渲染方法，因此相关工作主要从这两个方面展开。

- **Neural 3D shape representations:** 

  有一些研究工作探索了通过深度神经网络来建模连续的3D形状。但这些方法在简单形状上会出现过度平滑的现象，效果不好。**本文通过引入视角输入，来将原有的3-D输入扩展到5-D，从而获得更高精度的表征。**

- **View Synthesis and image-based rendering:** 

  当给定密集的视角样本时，通过光场插值即可重构视角。而当前研究的重点是利用稀疏的视角样本来生成高质量的视角。

  - 一种是基于网格的方法（mesh-based）。通过*可微分光栅器*或*路径追踪器*可以直接优化网格表示，利用梯度下降法再现一组输入图像。

    然而这种方法难优化，难以达到局部极小。并且需要一组提前设定的网格模板，不具有普适性。

  - 另一种方法是采用体积表示（volumetric representation）。这种方法能够表示复杂的形状，并且适合梯度下降。

    早期的体积方法直接用RGB图像为体素网格着色，后来开始通过训练深度网络来预测应该的体积表示。

    但该类方法在渲染获得较高分辨率图像时，需要更精细的采样，导致算法的时间&空间复杂度较高。

    **本文通过在MLP的参数中编码一个连续的体积来解决这一问题。？？？**

# 3. Neural Radiance Field Scene Representation

本节描述了通过神经网络来隐式的建模一个3-D场景的思想。

**神经辐射场场景表示和可微渲染过程的简介图**

![image-20220911160558393](C:\Users\hp\Desktop\Notebooks\Paper Reading\Novel view synthesis\image-2.png)

1. 首先沿着相机射线生成5-D坐标；
2. 训练MLP来预测RGB和体积密度
3. 利用体渲染方法生成图像
4. 由于体积函数是连续、可微的，所以可以采用梯度下降的方法来训练。

**如何由2-D图像获得需要的5D坐标以及输出的真实值？**答：2-D图像相当于一个观察者，其上每个像素发出一个camera ray，从射线上采样获得样本点。而这个样本点并没有真实的RGB和density，需要将其渲染成一个2维图像，再将这个渲染获得的二维图像与实际输入的二维图像之间作差，得到误差函数。再渲染二维图像时，所用到的数据是MLP输出的各个点的RGB与density

在实际中，将3D位置表示为笛卡尔单位向量

为了使场景表征是多视角一致的，限制网络对于**体积密度的预测只与空间坐标有关**，而对于**RGB颜色的预测与空间坐标和视角同时相关**。

具体的实现方式为：首先MLP输入三维空间坐标，此时输出是体积密度和一个256-D的特征向量；再将特征向量与输入视角进行拼接，送至全连接层，之后输出预测的RGB颜色

![image-20220912181138435](C:\Users\hp\Desktop\Notebooks\Paper Reading\Novel view synthesis\image-3.png)

a图和b图展示了在一个场景中，同一个位置处在不同视角下颜色的不同。

c图描述了一个固定位置在整个半球空间视角下颜色的分布情况。

# 4. Volume Rendering with Radiance Fields:

**本节简述**：叙述了如何由MLP输出的RGB与density来计算2-D图像上每个像素的颜色，以及如何在计算机中离散化的实现。

**Volume density：**表示射线在x位置的无穷小粒子处终止的微分概率（The volume density σ(x) can be interpreted as ***the differential probability of a ray terminating at an infinitesimal particle at location x***）

在下式中，C(r)表示渲染得到的颜色，直观上是一条射线经过tn->tf后的结果。其中T(t)表示由tn->tf的累计透过率，即射线从tn到t不撞击任何其他粒子的概率。

![image-20220912202931000](C:\Users\hp\Desktop\Notebooks\Paper Reading\Novel view synthesis\image-4.png)

针对预期的2-D图片上的每个像素都需要估计上述的积分

但在计算机中，需要对连续的积分项进行离散化。本文对该积分进行了数值估计。

**分层采样方法：**文章中提到，*确定性积分（deterministic quadrature, 定积分？）会限制表征的分辨率，原因在于MLP只会在固定的离散位置集处查询*（**什么意思？？**）。所以论文中提出了一种*分层采样方法*。具体的，先将tn->tf均匀分成N份，再从每一份中均匀随机的抽取一个样本。

即使是使用离散的样本集来估计积分，通过分层采样法，也能够获得场景的连续表示，因为在MLP的优化过程中是在连续的位置进行评估。（***什么意思？？？因为是随机采样所以能够获得连续的表示？？？***）

![image-20220912205540258](C:\Users\hp\Desktop\Notebooks\Paper Reading\Novel view synthesis\image-5.png)

最终通过上式来估计表示颜色的积分项。这个由（c, σ）集来计算C(r)的公式是可微的。

# 5. Optimizing a Neural Radiance Field:

引入了两个改进来实现高分辨率复杂场景的表示。

第一个是输入坐标的位置编码，帮助MLP表示高频函数；

第二个是分层抽样过程，使我们能够有效地对这种高频表示进行抽样。

## 5.1 Positional encoding

尽管神经网络可以作为通用的函数逼近器，但直接将xyzθΦ作为输入将导致渲染时对颜色和形状的高频变化不敏感（**可能是输入特征的粒度太粗，从而导致输出对于输入的变化不明显？？**）

已有的研究说明神经网络倾向于学习低频函数，并且在将输入数据传递到网络之前，使用**高频函数将输入数据映射到更高的维度空间**，可以更好地拟合包含高频变化的数据。

![image-20220912230019961](C:\Users\hp\Desktop\Notebooks\Paper Reading\Novel view synthesis\image-6.png)

位置编码的具体方式如上，首先将xyz标准化到[-1, 1]范围，然后利用上述公式分别映射至2L维的高维空间。在实验中，xyz的L为10，视角d的L为4.

## 5.2 Hierarchical volume sampling

在渲染时，如果是密集地沿camera ray采样N个点是低效的，因为可能存在一些自由空间和被遮挡的区域被反复采样。

该论文提出一种分层采样策略，通过**按最终渲染的预期效果分配样本**来提高渲染效率。

具体的步骤如下：

- 训练两个网络，分别是”coarse粗糙的“和”fine好的“
- 对于粗糙网络，按前文提到的分层抽样方法选取Nc个样本，然后来评估该网络（就是通过这些点来渲染一个2D图像）
- 在评估粗糙网络时，每个像素处的颜色可以表示为加权和的形式（如下），而该形式相当于一个分段常数概率密度函数PDF。再利用反变化采样法（traverse transform sampling）从这个分布中采样Nf个样本

![image-20220912231656788](C:\Users\hp\Desktop\Notebooks\Paper Reading\Novel view synthesis\image-7.png)

- 将Nc+Nf个样本共同作为样本集来评估fine网络，并计算获得最终的渲染颜色

这个过程将更多的样本分配给我们希望包含可见内容的区域。
