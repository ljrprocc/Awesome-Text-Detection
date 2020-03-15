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
- 简介：本文在Textboxes的基础上，。。。

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

### 4. Segmentation-based

#### Learning Shape-Aware Embedding for Scene Text Detection

Z Tian, M Shu, P Lyu, R Li, C Zhou

- Paper [Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Learning_Shape-Aware_Embedding_for_Scene_Text_Detection_CVPR_2019_paper.pdf)
- Code [Link]()
- 简介：





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

目前包括80w张图片，拥有

## 评估方法

## 其他资源

- [Scene Text Detection Resources](https://github.com/HCIILAB/Scene-Text-Detection)
- [Scene Text Understanding](https://github.com/tangzhenyu/Scene-Text-Understanding)
- [Scene Text Papers](https://github.com/Jyouhou/SceneTextPapers)
- [Scene Text Detection Datasets](https://www.ctolib.com/HCIILAB-Scene-Text-Detection.html)





