# Text Detection总结

## 方法

### 1. 传统方法

#### 1.1 Connected-Component Methods

#### 1.2 Sliding-Window Methods

### 2. Bottom-Up 方法

基本思路是回归出每个字符或者pixel的坐标或者位置，之后根据字符间的上下文关系预测文字区域。主要的论文包括:

#### PixelLink: Detecting scene text via instance segmentation

D Deng, H Liu, X Li, D Cai, AAAI 2018.

- Paper [Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16469/16260)
- Code [Link](https://github.com/ZJULearning/pixel_link)
- 简述：基于实例分割的回归文本区域的方法。本文将同一个实例中的pixel相连起来，而不需要定位的分支。

#### Detecting oriented text in natural images by linking segments

B Shi, X Bai, S Belongie,  CVPR 2017

- Paper [Link](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shi_Detecting_Oriented_Text_CVPR_2017_paper.pdf)
- Code [Link](https://github.com/dengdan/seglink)
- 简述：本文主要回归每个单独的Segment（一般为一个单词或者文本行），以及每个segment之间的link，从而回归出文字区域。其具有20FPS的检测效率。

#### Multi-oriented text detection with fully convolutional networks.

Z. Zhang, C. Zhang, W. Shen, C. Yao, W. Liu, and X. Bai. CVPR 2016

- Paper [Link](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Multi-Oriented_Text_Detection_CVPR_2016_paper.pdf)
- Code [Link](https://github.com/stupidZZ/FCN_Text)
- 简述：回归出了一系列字符级别的Text block的candidate，并使用基于FCN的分割网络预测每个区域的Salient map，并使用另外一个FCN网络预测每个字符的中心位置。

#### Scene text detection via holistic, multi-channel prediction

C Yao, X Bai, N Sang, X Zhou, S Zhou， arXiv 2016

- Paper [Link](https://arxiv.org/pdf/1606.09002)
- Code Link
- 简述：考虑字符、单词以及文本行的长距离上下文信息，使用语义分割的方法构建每个文字区域的信息。

#### Mask textspotter:An end-to-end trainable neural network for spotting text with arbitrary shapes.

P Lyu, M Liao, C Yao, W Wu, ECCV 2018

- Paper [Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.pdf)
- Code [Link](https://github.com/MhLiao/MaskTextSpotter)
- 简述：本文主要预测每个字符的probability map，并把识别任务视作一个语义分割的任务。之后合成整个detection map。本文基于Mask RCNN，分别使用两个支路同时实现文本检测与识别。

#### Wordsup: Exploiting word annotations for character based text detection

H Hu, C Zhang, Y Luo, Y Wang，ICCV 2017

- Paper [Link](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hu_WordSup_Exploiting_Word_ICCV_2017_paper.pdf)
- Code Link
- 简介：本文主要探究一种弱监督的，基于单词标注级别的文本检测方法。本文的annotation生成的方式也是Bottom-Up的。本文在基于Word的标注的同时，也进行Character Grouping生成字符区域，并生成识别结果。

#### CRAFT: Character Region Awareness for Text Detection

Y Baek, B Lee, D Han, S Yun，CVPR 2019

- Paper [Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf)
- Code [Link](https://github.com/clovaai/CRAFT-pytorch)
- 简介：本文基于WordSup，通过分割字符级别的区域，实现对任意形状的文本区域的分割。本文也是弱监督的文字框架。

### 3. Proposal-based

#### CTPN: Detecting Text in Natural Image with Connectionist Text Proposal Network

Z Tian, W Huang, T He, P He, Y Qiao,  ECCV 2016

- Paper [Link](https://arxiv.org/pdf/1609.03605)
- Code [Link](https://github.com/opconty/pytorch_ctpn)
- 简介：本文回归出$h \times 16$的小框，使用基于Character 的proposal并且使用了4倍上采样以适应多尺度的检测标准。在回归小框之后，使用BiLSTM回归出小框与小框之间的上下文关系，通过定义neighbour规则以确定text region的边界最终回归出Text Region。但其缺乏对于弯曲文本以及任意形状文本的检测方法。

#### Textboxes: A fast text detector with a single deep neural network

M Liao, B Shi, X Bai, X Wang, W Liu， AAAI 2017

- Paper [Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14202/14295)
- Code [Link]()
- 简介：本文针对场景文本检测中存在的目标多尺度问题，提出了一种基于FCN的框架，根据不同word的长宽比来确定不同类型的文本实例。

#### Textboxes++: A single-shot oriented scene text detector

M Liao, B Shi, X Bai， TOIP 2018

- Paper [Link](https://arxiv.org/pdf/1801.02765.pdf)
- Code [Link](https://github.com/MhLiao/TextBoxes_plusplus)
- 简介：本文在Textboxes的基础上，解决了多方向以及多尺度的文本检测问题。此外，该种方法简化了后处理方法，仅仅保留了NMS，从而保证了检测的效率。

#### Deep matching prior network: Toward tighter multi-oriented text detection

Y Liu, L Jin， CVPR 2017

- Paper [Link](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Deep_Matching_Prior_CVPR_2017_paper.pdf)
- Code [Link](https://github.com/Yuliang-Liu/Curve-Text-Detector)
- 简介：主要解决多方向的、不同大小颜色以及尺寸的text region，进行更紧密的bounding box的检测。另外，不同于传统的smooth L1 loss，本文使用了Ln loss，使得整个模型具有更强的鲁棒性和稳定性。

#### Rotation-sensitive regression for oriented scene text detection.

M Liao, Z Zhu, B Shi, G Xia， CVPR 2018

- Paper [Link](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liao_Rotation-Sensitive_Regression_for_CVPR_2018_paper.pdf)
- Code [Link](https://github.com/MhLiao/RRD)
- 简介：本文主要解决了multi-orient的文本检测问题。本文在传统的Bounding box回归的基础上增加了一个rotation-sensitive feature，旋转传统的卷积核。其在旋转的场景文本数据集中取得了较好的效果。

#### Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes

- Paper [Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Look_More_Than_Once_An_Accurate_Detector_for_Text_of_CVPR_2019_paper.pdf)
- Code [Link]()
- 简介：本文提出了一种LOMO模型，用以解决矩形bbox的感受野的适应性问题，以及长文本和任意形状文本的问题。重点包括直接回归器（DR）, 迭代调整模块（IRM）以及形状表达模块（SEM）。该方法在弯曲文本数据集上取得了较好的结果。

### 4. Segmentation-based

#### EAST: An Efficient and Accurate Scene Text Detector

Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, and Jiajun Liang, CVPR 2017

- Paper [Link](EAST: An Efficient and Accurate Scene Text Detector)
- Code [Link](https://github.com/argman/EAST)
- 简介：本文使用了一种基于multi-channel的FCN的框架，得到了每张图片的text region的mask，通过NMS之后得到多方向性的文本行的bounding box。这种方法既可以做单词级别的预测也可以做文本行级别的预测。该方法在准确率和检测速度上都具有显著优势。

#### Learning Shape-Aware Embedding for Scene Text Detection

Z Tian, M Shu, P Lyu, R Li, C Zhou，CVPR 2019

- Paper [Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Learning_Shape-Aware_Embedding_for_Scene_Text_Detection_CVPR_2019_paper.pdf)
- Code [Link]()
- 简介：本文主要解决了文字检测问题中的弯曲文本检测问题，提出了一种Shape Aware Embedding的策略，通过学习这种loss，以实现同种Text instance在Embedding之后的距离小，而不同种的距离大。另外加上Segmentation Masks的Dice loss，最终实现整个网络结果的训练。本文主要预测1个binary score map, 4个bbox顶点以及1个角度的feature map，通过dice loss进行优化。

#### Textsnake: A flexible representation for detecting text of arbitrary shapes

S Long, J Ruan, W Zhang, X He， CVPR 2018

- Paper [Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shangbang_Long_TextSnake_A_Flexible_ECCV_2018_paper.pdf)
- Code [Link](https://github.com/princewang1994/TextSnake.pytorch)
- 简介：本文主要介绍了一种检测任意形状文本的方法。具体而言，在分割出文字区域之后，会根据文字中心位置同时回归出文字的center line，并根据其几何形状进行预测。主要回归的是文字的尺寸$r$以及倾角$\theta$。



## 数据集

目前的文本检测数据集还是有很多的，可以参考https://www.ctolib.com/HCIILAB-Scene-Text-Detection.html

### 1. 场景检测数据集

#### ICDAR series(ICDAR 13/15/17/19)

是ICDAR竞赛所提供的官方数据集，该比赛是有关端到端OCR的比赛，包括VQA，多语言检测、视频文本检测等多个项目。

ICDAR数据提供的每个bounding box的标注样式为：

$$\{x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, l, c\}$$

其中$(x_i, y_i)$表示第$i$个顶点，$l$表示的是何种语言，$c$是文字识别的结果。

#### COCO-Text

是基于COCO目标检测数据集的基础上，针对其中存在的场景文字而制作的文本检测数据集。主要是实际场景检测，为实拍场景。其每个bounding box的标注样式为：

$$\{x_{min}, y_{min}, x_{max}, y_{max}\}$$

这也是VOC、MS COCO等目标检测数据集所采用的标注方法，即仅仅标注bounding box的两个顶点，而默认其为矩形框。

#### CTW

其是由清华大学-腾讯联合实验室提出的中文实际场景检测的数据集，其以倾斜文本和任意形状的文本居多，包括街景、路标、车牌等。

#### SCUT-CTW1500

1500张的中文街景文本图片，其包含14个顶点的多边形Bounding box标注。每张图片中包含至少一个弯曲实例，其中1000张是训练集，500张是测试集。

#### MSRA-TD500

包括300张训练图片以及200张训练图片，包括中英文的文本实例。他们的标注格式为：

$$(idx, dl, x, y, w, h, \theta)$$

其中$idx$表示的是图片的编号，$dl$表示difficult label，$\theta$表示与水平方向的夹角的余弦值。

### 2. 合成文本检测数据集

### SynthText

目前包括80w张图片，其主要是在实拍场景的基础上添加的一些合成文字。一般合成数据集是用来预训练底层网络的。

上述80w张合成数据集是官方提供的，官方也提供了自动生成合成文本数据集的代码：https://github.com/ankush-me/SynthText。

## 评估方法

目前文字检测采用的评估方法也是基于目标检测的。因此需要补充一下目标检测的几种评价指标。在目前有监督的目标检测方法中，通常在一张图片中标注出$M$个Ground truth bounding box，也就是$\mathcal{G} = \{G_1, G_2, \dots, G_M\}$，而检测模型通常在经过后处理后会生成$N$个预测的待选框，也就是$\mathcal{D} = \{D_1, D_2, \dots, D_N\}$，那么评估的主要目标是模型预测的框$D_j$与Ground Truth框$G_i$的相近程度。

<img src="pics\eval1.png" alt="img" style="zoom:80%;" />

那么借助$IoU$的定义，True Positive（TP）以及False Positive（FP）的定义如图所示。另外，False Negative（FN） 定义为

$$IoU(i,j) = 0,  \forall D_j \in \mathcal{D}$$

#### Precision

$$Precision = \frac{TP}{TP + FP}$$

#### Recall

$$Recall = \frac{TP}{TP+FN}$$

#### H-means

全称是harmonic mean（调和平均数）of precision and recall：

$$Hmean = 2\frac{Recall \times Precision}{Recall + precision}$$

这里的$Recall$以及$Precision$的定义如之前所述。

#### mAP

这是从目标检测中Average Precision的计算中得到的。

#### TIoU: Tightness-aware Evaluation Protocol for Scene Text Detection

Yuliang Liu, Lianwen Jin, Zecheng Xie, Canjie Luo, Shuaitao Zhang, Lele Xie, CVPR 2019

- Paper [Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Tightness-Aware_Evaluation_Protocol_for_Scene_Text_Detection_CVPR_2019_paper.pdf)
- Code [Link](https:
  //github.com/Yuliang-Liu/TIoU-metric)
- 简介：本文在总结了IC03以及IC13/15的评估方法的基础上，指出了上述基于固定的IoU阈值确定True Positive bounding boxes的方法存在引入背景噪声、部分文字缺失的缺陷。因此本文引入了Tightness-aware的IoU评估方法，并提出了基于TIoU的Precision-Recall计算方法。这种评估方法在文本检测领域取得了较好的结果，由于其更紧的检测边界因此使用该种方法能更符合人的视觉直观感受。



## 其他资源

- [Scene Text Detection Resources](https://github.com/HCIILAB/Scene-Text-Detection)
- [Scene Text Understanding](https://github.com/tangzhenyu/Scene-Text-Understanding)
- [Scene Text Papers](https://github.com/Jyouhou/SceneTextPapers)
- [Scene Text Detection Datasets](https://www.ctolib.com/HCIILAB-Scene-Text-Detection.html)





