# Awesome Deepfakes Detection![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

A list of Deepfakes Detection datasets, tools, papers and code. If this list help you in your research, a star is my pleasure.

If you want to contribute to this list, welcome to send me a pull request or contact me :) .

This repo only collect papers related to Deepfake Detection. If you are also interested in Deepfakes, please refer to: [Awesome Deepfakes](https://github.com/Daisy-Zhang/Awesome-Deepfakes).

## Contents

- [Datasets](#datasets)
  - [Video Datasets](#video-datasets)
  - [Image Datasets](#image-datasets)
- [Benchmark](#benchmark)
- [Competition](#competition)
- [Tools](#tools)
- [Papers](#papers)
  - [CVPR](#cvpr)
  - [ICCV](#iccv)
  - [ECCV](#eccv)
  - [ICML](#icml)
  - [IJCAI](#ijcai)
  - [AAAI](#aaai)
  - [NIPS](#nips)
  - [ICLR](#iclr)
  - [MM](#mm)
  - [ICME](#icme)
  - [TPAMI](#tpami)
  - [TIFS](#tifs)
  - [Other](#other)

## Benchmark

For better comparison and research purpose, we also collect the benchmark of all the SOTA methods we can get on Celeb-DF, DFDC and FaceForensic++ datasets in video-level. We use AUC score (%) as the metrics. The results are presented as follows. Results in *italics* indicate they were conducted in [Yuezun Li](https://arxiv.org/abs/1909.12962) or [Alexandros Haliassos](https://arxiv.org/pdf/2012.07657.pdf), not in their original paper.

|                                                           | Celeb-DF(v2) |  DFDC  | FaceForensic++ | Year |                           *Note*                            |
| :-------------------------------------------------------: | :----------: | :----: | :------------: | :--: | :---------------------------------------------------------: |
|      [Two-Stream](https://arxiv.org/abs/1803.11276)       |    *53.8*    | *61.4* |     *70.7*     | 2017 |   FF++ only on DF subset. Use provided pre-trained model.   |
|         [Meso4](https://arxiv.org/abs/1809.00888)         |    *54.8*    | *75.3* |     *84.7*     | 2018 |                       Same as above.                        |
|    [MesoInception4](https://arxiv.org/abs/1809.00888)     |    *53.6*    | *73.2* |     *83.0*     | 2018 |                       Same as above.                        |
|          [FWA](https://arxiv.org/abs/1811.00656)          |    *56.9*    | *72.7* |     *80.1*     | 2018 |                       Same as above.                        |
|        [DSP-FWA](https://arxiv.org/abs/1811.00656)        |    *64.6*    | *75.5* |     *93.0*     | 2018 |                       Same as above.                        |
|  [VA-MLP](https://ieeexplore.ieee.org/document/8638330)   |    *55.0*    | *61.9* |     *66.4*     | 2019 |                       Same as above.                        |
| [VA-LogReg](https://ieeexplore.ieee.org/document/8638330) |    *55.1*    | *66.2* |     *78.0*     | 2019 |                       Same as above.                        |
|     [Xception-raw](https://arxiv.org/abs/1901.08971)      |    *48.2*    | *49.9* |     *99.7*     | 2019 |                       Same as above.                        |
|     [Xception-c23](https://arxiv.org/abs/1901.08971)      |    *65.3*    | *72.2* |     *99.7*     | 2019 |                       Same as above.                        |
|     [Xception-c40](https://arxiv.org/abs/1901.08971)      |    *65.5*    | *69.7* |     *95.5*     | 2019 |                       Same as above.                        |
|      [Multi-Task](https://arxiv.org/abs/1906.06876)       |    *54.3*    | *53.6* |     *76.3*     | 2019 |                       Same as above.                        |
|      [CapsuleNet](https://arxiv.org/abs/1910.12467)       |    *57.5*    | *53.3* |     *96.6*     | 2019 |                       Same as above.                        |
|       [CNN-Spot](https://arxiv.org/abs/1912.11035)        |    *75.6*    | *72.1* |     *65.7*     | 2019 | FF++ only on FaceShifter HQ subset. All pretrained on FF++. |
|      [Face X-ray](https://arxiv.org/abs/1912.13458)       |    *79.5*    | *65.5* |     *92.8*     | 2019 |                       Same as above.                        |
|        [CNN-RNN](https://arxiv.org/abs/1905.00582)        |    *69.8*    | *68.9* |     *80.8*     | 2019 |                       Same as above.                        |
|     [LipsForensics](https://arxiv.org/abs/2012.07657)     |     82.4     |  73.5  |      97.1      | 2020 |                       Same as above.                        |
|      [Two-Branch](https://arxiv.org/abs/2008.03412)       |     76.7     |   -    |      93.2      | 2020 |                              -                              |
|      [Patch-based](https://arxiv.org/abs/2008.10588)      |    *69.6*    | *65.6* |     *57.8*     | 2020 |                       Same as above.                        |
|   [FD<sup>2</sup>Net](https://arxiv.org/abs/2011.09737)   |      -       | 66.09  |     99.45      | 2020 |                              -                              |
|         [LRNet](https://arxiv.org/abs/2104.04480)         |     56.9     |   -    |      99.9      | 2021 |                      Trained on FF++.                       |



## Datasets

### Video Datasets

* **UADFV**: "Exposing Deep Fakes Using Inconsistent Head Poses." [Paper](https://arxiv.org/abs/1811.00661)
* **EBV**: "In Ictu Oculi: Exposing AI Generated Fake Face Videos by Detecting Eye Blinking." [Paper](https://arxiv.org/abs/1806.02877)    [Download](http://www.cs.albany.edu/~lsw/downloads.html)
* **Deepfake-TIMIT**: "DeepFakes: a New Threat to Face Recognition? Assessment and Detection." [Paper](https://arxiv.org/abs/1812.08685)    [Download](https://conradsanderson.id.au/vidtimit/)
* **DFFD**: "On the Detection of Digital Face Manipulation." [Paper](http://cvlab.cse.msu.edu/pdfs/dang_liu_stehouwer_liu_jain_cvpr2020.pdf)    [Download](http://cvlab.cse.msu.edu/dffd-dataset.html)
* **DeepfakeDetection**: [Download](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
* **Celeb-DF (v1)**: "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics." [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf)    [Download](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1)
* **Celeb-DF (v2)**: "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics." [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf)    [Download](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v2)
* **DFDC**: "The DeepFake Detection Challenge (DFDC) Dataset." [Paper](https://arxiv.org/abs/2006.07397)    [Download](https://www.kaggle.com/c/deepfake-detection-challenge/data) 
* **FaceForensic++**: "FaceForensics++: Learning to Detect Manipulated Facial Images." [Paper](https://arxiv.org/abs/1901.08971)    [Download](https://github.com/ondyari/FaceForensics)
* **FFIW-10K**: "Face Forensics in the Wild." [Paper](https://arxiv.org/abs/2103.16076)    [Download](https://github.com/tfzhou/FFIW)
* **Deeper Forensic-1.0**: "DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection." [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_DeeperForensics-1.0_A_Large-Scale_Dataset_for_Real-World_Face_Forgery_Detection_CVPR_2020_paper.pdf)    [Download](https://github.com/EndlessSora/DeeperForensics-1.0)
* **Wild Deepfake**: "WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection." [Paper](https://arxiv.org/abs/2101.01456)    [Download](https://github.com/deepfakeinthewild/deepfake-in-the-wild)
* **ForgeryNet**: "ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis." [Paper](https://arxiv.org/abs/2103.05630)    [Download](https://github.com/yinanhe/forgerynet)
* **WLDR**: "Protecting World Leaders Against Deep Fakes." [Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)

|                     | Real Videos | Fake Videos | Year |                             Note                             |
| :-----------------: | :---------: | :---------: | :--: | :----------------------------------------------------------: |
|        UADFV        |     49      |     49      | 2018 |                      focus on head pose                      |
|         EBV         |      -      |     49      | 2018 |                    focus on eye blinking                     |
|   Deepfake-TIMIT    |     320     |     640     | 2018 |                      GAN-Based methods                       |
|        DFFD         |    1,000    |    3000     | 2019 |                     mutiple SOTA methods                     |
|  DeepfakeDetection  |     363     |    3,608    | 2019 | collect from actors with publicly available generation methods |
|    Celeb-DF (v2)    |     590     |    5639     | 2019 |                         high quality                         |
|        DFDC         |   23,564    |   104,500   | 2019 |                  DFDC competition on Kaggle                  |
|   FaceForensic++    |    1,000    |    5,000    | 2019 |              five different generation methods               |
|      FFIW-10K       |   10,000    |   10,000    | 2019 |                  mutiple faces in one frame                  |
|        WLDR         |      -      |      -      | 2019 |            person of interest video from Youtube             |
| DeeperForensics-1.0 |   50,000    |   10,000    | 2020 |                 add real-world perturbations                 |
|    Wild-Deepfake    |    3,805    |    3,509    | 2021 |                    collect from Internet                     |
|     ForgeryNet      |   99,630    |   121,617   | 2021 |     8 video-level generation methods, add perturbations      |



### Image Datasets

* **DFFD**: "On the Detection of Digital Face Manipulation." [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dang_On_the_Detection_of_Digital_Face_Manipulation_CVPR_2020_paper.pdf)    [Download](http://cvlab.cse.msu.edu/project-ffd.html)
* **FFHQ**: "A Style-Based Generator Architecture for Generative Adversarial Networks." [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf)    [Download](https://github.com/NVlabs/ffhq-dataset)
* **iFakeFaceDB**: "GANprintR: Improved Fakes and Evaluation of the State of the Art in Face Manipulation Detection." [Paper](https://arxiv.org/abs/1911.05351)    [Download](https://github.com/socialabubi/iFakeFaceDB)
* **100k Faces Generated by AI (Online)**: [Download](https://generated.photos/datasets)
* **DFGC**: "DFGC 2021: A DeepFake Game Competition." [Paper](https://arxiv.org/abs/2106.01217)    [Dowload](https://github.com/bomb2peng/DFGC_starterkit)
* **ForgeryNet**: "ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis." [Paper](https://arxiv.org/abs/2103.05630)    [Download](https://github.com/yinanhe/forgerynet)

|             | Real Images |    Fake Images     | Year |                         Note                          |
| :---------: | :---------: | :----------------: | :--: | :---------------------------------------------------: |
|    DFFD     |   58,703    |      240,336       | 2019 |                 mutiple SOTA methods                  |
|    FFHQ     |      -      | 70,000 (GAN-based) | 2019 |            generated by GAN-based methods             |
| iFakeFaceDB |      -      | 87,000 (StyleGAN)  | 2020 |                 generated by StyleGAN                 |
| 100k Faces  |      -      | 100,000 (StyleGAN) | 2021 |                 generated by StyleGAN                 |
|    DFGC     |    1,000    |      N*1,000       | 2021 | DFGC 2021 competition, fake images generated by users |
| ForgeryNet  |  1,438,201  |     1,457,861      | 2021 |  7 image-level generation methods, add perturbations  |



## Competition

|              Name               |                             Link                             | Year |                             Note                             |
| :-----------------------------: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: |
|  Deepfake Detection Challenge   | [Website](https://www.kaggle.com/c/deepfake-detection-challenge) | 2019 | 1. video-level detection.<br>2. use DFDC datasets.<br/> 3. the first worldwide competition.<br/> 4. more than 2,000 teams participated. |
|     DeepForensics Challenge     | [Website](https://competitions.codalab.org/competitions/25228) | 2020 | 1. video-level detection.<br/> 2. use DeeperForensics-1.0 datasets.<br/> 3. simulates real-world scenarios. |
|    Deepfake Game Competition    | [Website](https://competitions.codalab.org/competitions/29583) | 2021 | 1. both image-level generation and video-level detection track.<br/> 2. use Celeb-DF(v2) datasets. |
| Face Forgery Analysis Challenge | [Website](https://competitions.codalab.org/competitions/33386) | 2021 | 1. both image-level and video-level detection track.<br/> 2. addtional temporal localization track.<br/> 3. use ForgeryNet dataset. |



## Tools

* **Sensity**: [Website](https://sensity.ai/)
* **Deepware**: [Website](https://deepware.ai/)
* **Baidu Security**: [Website](http://weishi.baidu.com/product/deepfake)
* **DeepReal**: [Website](https://deepfakes.real-ai.cn/)



## Papers

### CVPR

* "Learning Second Order Local Anomaly for General Face Forgery Detection", CVPR 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Fei_Learning_Second_Order_Local_Anomaly_for_General_Face_Forgery_Detection_CVPR_2022_paper.pdf)
* "DeepFake Disrupter: The Detector of DeepFake Is My Friend", CVPR 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_DeepFake_Disrupter_The_Detector_of_DeepFake_Is_My_Friend_CVPR_2022_paper.pdf)
* "Protecting Celebrities from DeepFake with Identity Consistency Transformer", CVPR 2022: [Paper](https://arxiv.org/abs/2203.01318v3) [Github](https://github.com/LightDXY/ICT_DeepFake)

* "Detecting Deepfakes with Self-Blended Images", CVPR 2022: [Paper](https://arxiv.org/abs/2204.08376) [Github](https://github.com/mapooon/SelfBlendedImages)
* "Self-supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection", CVPR 2022: [Paper](https://arxiv.org/abs/2203.12208) [Github](https://github.com/liangchen527/SLADD)
* "Voice-Face Homogeneity Tells Deepfake", CVPR 2022: [Paper](https://arxiv.org/abs/2203.02195#:~:text=Detecting%20forgery%20videos%20is%20highly,fit%20well%20on%20certain%20data.) [Github](https://github.com/xaCheng1996/VFD)

* "End-to-End Reconstruction-Classification Learning for Face Forgery Detection", CVPR 2022: [Paper](https://vision.sjtu.edu.cn/files/cvpr22_RECCE.pdf)
* "Robust Image Forgery Detection Against Transmission Over Online Social Networks", CVPR 2022: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9686650) [Github](https://github.com/HighwayWu/ImageForensicsOSN)
* "Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection", CVPR 2022: [Paper](https://arxiv.org/pdf/2201.07131.pdf)
* "Exploring Frequency Adversarial Attacks for Face Forgery Detection", CVPR 2022: [Paper](https://arxiv.org/pdf/2203.15674.pdf)

* "ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis", CVPR 2021: [Paper](https://arxiv.org/abs/2103.05630)    [Github](https://github.com/yinanhe/forgerynet)
* "Representative Forgery Mining for Fake Face Detection", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Representative_Forgery_Mining_for_Fake_Face_Detection_CVPR_2021_paper.pdf)    [Github](https://github.com/crywang/RFM)
* "MagDR: Mask-Guided Detection and Reconstruction for Defending Deepfakes", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_MagDR_Mask-Guided_Detection_and_Reconstruction_for_Defending_Deepfakes_CVPR_2021_paper.pdf)
* "Improving the Efficiency and Robustness of Deepfakes Detection Through Precise Geometric Features", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Improving_the_Efficiency_and_Robustness_of_Deepfakes_Detection_Through_Precise_CVPR_2021_paper.pdf)    [Github](https://github.com/frederickszk/LRnet)
* "Multi-Attentional Deepfake Detection", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Multi-Attentional_Deepfake_Detection_CVPR_2021_paper.html)    [Github](https://github.com/yoctta/multiple-attention)
* "DeepFakes Evolution: Analysis of Facial Regions and Fake Detection Performance", arXiv 2021: [Paper](https://arxiv.org/pdf/2004.07532.pdf?ref=https://githubhelp.com)
* "Lips Don't Lie: A Generalisable and Robust Approach To Face Forgery Detection", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Haliassos_Lips_Dont_Lie_A_Generalisable_and_Robust_Approach_To_Face_CVPR_2021_paper.pdf)
* "Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Spatial-Phase_Shallow_Learning_Rethinking_Face_Forgery_Detection_in_Frequency_Domain_CVPR_2021_paper.pdf)
* "Frequency-Aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Frequency-Aware_Discriminative_Feature_Learning_Supervised_by_Single-Center_Loss_for_Face_CVPR_2021_paper.pdf)
* "Generalizing Face Forgery Detection With High-Frequency Features", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Generalizing_Face_Forgery_Detection_With_High-Frequency_Features_CVPR_2021_paper.pdf)
* "Face Forgery Detection by 3D Decomposition", CVPR 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Face_Forgery_Detection_by_3D_Decomposition_CVPR_2021_paper.pdf)
* "Global Texture Enhancement for Fake Face Detection in the Wild", CVPR 2020: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Global_Texture_Enhancement_for_Fake_Face_Detection_in_the_Wild_CVPR_2020_paper.pdf)
* "On the Detection of Digital Face Manipulation", CVPR 2020: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dang_On_the_Detection_of_Digital_Face_Manipulation_CVPR_2020_paper.pdf)    [Github](https://github.com/JStehouwer/FFD_CVPR2020)
* "Face X-Ray for More General Face Forgery Detection", CVPR 2020: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.pdf)
* "CNN-generated images are surprisingly easy to spot... for now", CVPR 2020: [Paper](https://arxiv.org/abs/1912.11035)    [Github](https://github.com/PeterWang512/CNNDetection)
* "CORE: Consistent Representation Learning for Face Forgery Detection", CVPR Workshop 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022W/WMF/papers/Ni_CORE_COnsistent_REpresentation_Learning_for_Face_Forgery_Detection_CVPRW_2022_paper.pdf) [Github](https://github.com/niyunsheng/CORE)
* "FReTAL: Generalizing Deepfake Detection using Knowledge Distillation and Representation Learning", CVPR Workshop 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021W/WMF/papers/Kim_FReTAL_Generalizing_Deepfake_Detection_Using_Knowledge_Distillation_and_Representation_Learning_CVPRW_2021_paper.pdf)    [Github](https://github.com/alsgkals2/FReTAL-Generalizing_Deepfakes_using_Knowledge_Distillation_and_Representation_Learning)
* "Towards Untrusted Social Video Verification to Combat Deepfakes via Face Geometry Consistency", CVPR Workshop 2020: [Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Tursman_Towards_Untrusted_Social_Video_Verification_to_Combat_Deepfakes_via_Face_CVPRW_2020_paper.pdf)
* "OC-FakeDect: Classifying Deepfakes Using One-class Variational Autoencoder", CVPR Workshop 2020: [Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Khalid_OC-FakeDect_Classifying_Deepfakes_Using_One-Class_Variational_Autoencoder_CVPRW_2020_paper.pdf)
* "Exposing DeepFake Videos By Detecting Face Warping Artifacts", CVPR Workshop 2019: [Paper](https://arxiv.org/abs/1811.00656)
* "Recurrent Convolutional Strategies for Face Manipulation Detection in Videos", CVPR Workshop 2019: [Paper](https://arxiv.org/abs/1905.00582)
* "Two-Stream Neural Networks for Tampered Face Detection", CVPR Workshop 2017: [Paper](https://arxiv.org/abs/1803.11276)

### ICCV

* "Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data" ICCV 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Artificial_Fingerprinting_for_Generative_Models_Rooting_Deepfake_Attribution_in_Training_ICCV_2021_paper.pdf) [Github](https://github.com/ningyu1991/ArtificialGANFingerprints)
* "KoDF: A Large-scale Korean DeepFake Detection Dataset", ICCV 2021: [Paper](https://arxiv.org/abs/2103.10094)
* "Exploring Temporal Coherence for More General Video Face Forgery Detection", ICCV 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Exploring_Temporal_Coherence_for_More_General_Video_Face_Forgery_Detection_ICCV_2021_paper.pdf)
* "ID-Reveal: Identity-aware DeepFake Video Detection", ICCV 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cozzolino_ID-Reveal_Identity-Aware_DeepFake_Video_Detection_ICCV_2021_paper.pdf) [Github](https://github.com/grip-unina/id-reveal)
* "Joint Audio-Visual Deepfake Detection", ICCV 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Joint_Audio-Visual_Deepfake_Detection_ICCV_2021_paper.pdf)
* "Learning Self-Consistency for Deepfake Detection", ICCV 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Learning_Self-Consistency_for_Deepfake_Detection_ICCV_2021_paper.pdf)
* "OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild", ICCV 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Le_OpenForensics_Large-Scale_Challenging_Dataset_for_Multi-Face_Forgery_Detection_and_Segmentation_ICCV_2021_paper.pdf)
* "Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints", ICCV 2019: [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Attributing_Fake_Images_to_GANs_Learning_and_Analyzing_GAN_Fingerprints_ICCV_2019_paper.pdf)    [Github](https://github.com/ningyu1991/GANFingerprints)

### ECCV

* "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues", ECCV 2020: [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570086.pdf)
* "Two-branch Recurrent Network for Isolating Deepfakes in Videos", ECCV 2020: [Paper](https://arxiv.org/abs/2008.03412)

### ICML

* "Leveraging Frequency Analysis for Deep Fake Image Recognition", ICML 2020: [Paper](http://proceedings.mlr.press/v119/frank20a/frank20a.pdf)    [Github](https://github.com/RUB-SysSec/GANDCTAnalysis)

### IJCAI

* "Region-Aware Temporal Inconsistency Learning for DeepFake Video Detection", IJCAI 2022: [Paper](https://www.ijcai.org/proceedings/2022/0129.pdf)
* "Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations", IJCAI 2022: [Paper](https://www.ijcai.org/proceedings/2022/0107.pdf)
* "Detecting Deepfake Videos with Temporal Dropout 3DCNN", IJCAI 2021: [Paper](https://www.ijcai.org/proceedings/2021/0178.pdf)
* "Dynamic Inconsistency-aware DeepFake Video Detection", IJCAI 2021: [Paper](https://www.ijcai.org/proceedings/2021/0102.pdf)
* "An Examination of Fairness of AI Models for Deepfake Detection", IJCAI 2021: [Paper](https://www.ijcai.org/proceedings/2021/0079.pdf)
* "Beyond the Spectrum: Detecting Deepfakes via Re-Synthesis", IJCAI 2021: [Paper](https://www.ijcai.org/proceedings/2021/0349.pdf)    [Github](https://github.com/SSAW14/BeyondtheSpectrum)
* "FakeSpotter: A Simple yet Robust Baseline for Spotting AI-Synthesized Fake Faces", IJCAI 2020: [Paper](https://www.ijcai.org/Proceedings/2020/0476.pdf)

### AAAI

* "Delving into the Local: Dynamic Inconsistency Learning for DeepFake Video Detection", AAAI 2022: [Paper](https://www.aaai.org/AAAI22Papers/AAAI-1978.GuZ.pdf)
* "Deepfake Network Architecture Attribution", AAAI 2022: [Paper](https://arxiv.org/abs/2202.13843)
* "Dual Contrastive Learning for General Face Forgery Detection", AAAI 2022: [Paper](https://arxiv.org/abs/2112.13522)
* "CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes", AAAI 2022: [Paper](https://arxiv.org/abs/2105.10872)
* "ADD: Frequency Attention and Multi-View based Knowledge Distillation to Detect Low-Quality Compressed Deepfake Images", AAAI 2022: [Paper](https://arxiv.org/abs/2112.03553)
* "Exploiting Fine-grained Face Forgery Clues via Progressive Enhancement Learning", AAAI 2022: [Paper](https://arxiv.org/pdf/2112.13977.pdf)
* "FInfer: Frame Inference-based Deepfake Detection for High-Visual-Quality Videos", AAAI 2022: [Paper](https://www.aaai.org/AAAI22Papers/AAAI-3317.HuJ.pdf)
* "FrePGAN: Robust Deepfake Detection Using Frequency-level Perturbations", AAAI 2022: [Paper](https://arxiv.org/pdf/2202.03347.pdf)
* "Domain General Face Forgery Detection by Learning to Weight", AAAI 2021: [Paper](https://www.aaai.org/AAAI21Papers/AAAI-589.SunK.pdf)    [Github](https://github.com/skJack/LTW)
* "Local Relation Learning for Face Forgery Detection", AAAI 2021: [Paper](https://arxiv.org/pdf/2105.02577.pdf)

### NIPS

* "WaveFake: A Data Set to Facilitate Audio Deepfake Detection", NIPS 2021: [Paper](https://arxiv.org/abs/2111.02813)
* "AOT: Appearance Optimal Transport Based Identity Swapping for Forgery Detection", NIPS 2020: [Paper](https://papers.nips.cc/paper/2020/file/f718499c1c8cef6730f9fd03c8125cab-Paper.pdf)

### ICLR
* "Responsible Disclosure of Generative Models Using Scalable Fingerprinting", ICLR 2022: [Paper](https://openreview.net/pdf?id=sOK-zS6WHB)

### MM

* "Evaluation of an Audio-Video Multimodal Deepfake Dataset using Unimodal and Multimodal Detectors", ACM MM 2021: [Paper](https://arxiv.org/abs/2109.02993)
* "Spatiotemporal Inconsistency Learning for DeepFake Video Detection", ACM MM 2021: [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475508)
* "Video Transformer for Deepfake Detection with Incremental Learning", ACM MM 2021: [Paper](https://arxiv.org/ftp/arxiv/papers/2108/2108.05307.pdf)
* "Metric Learning for Anti-Compression Facial Forgery Detection", ACM MM 2021: [Paper](https://arxiv.org/pdf/2103.08397.pdf)
* "CoReD: Generalizing Fake Media Detection with Continual Representation using Distillation", ACM MM 2021: [Paper](https://arxiv.org/pdf/2107.02408.pdf)
* "FakeTagger: Robust Safeguards against DeepFake Dissemination via Provenance Tracking", ACM MM 2021: [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475518)
* "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset", ACM MM 2021 Workshop: [Paper](https://openreview.net/pdf?id=TAXFsg6ZaOl) [Github](https://github.com/hasamkhalid/FakeAVCeleb)
* "Not made for each other- Audio-Visual Dissonance-based Deepfake Detection and Localization", ACM MM 2020: [Paper](https://dl.acm.org/doi/10.1145/3394171.3413700)    [Github](https://github.com/abhinavdhall/deepfake/)
* "Sharp Multiple Instance Learning for DeepFake Video Detection", ACM MM 2020: [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3414034)
* "DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms", ACM MM 2020: [Paper](https://dl.acm.org/doi/10.1145/3394171.3413707)
* "Emotions Don't Lie: An Audio-Visual Deepfake Detection Method using Affective Cues", ACM MM 2020: [Paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413570)

### ICME

* "DEEPFAKE VIDEOS DETECTION USING SELF-SUPERVISED DECOUPLING NETWORK", ICME 2021: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9428368)
* "DLFMNET: END-TO-END DETECTION AND LOCALIZATION OF FACE MANIPULATION USING MULTI-DOMAIN FEATURES", ICME 2021: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9428450) [Github](https://github.com/LightningChan/DLFMNet)
* "DEFAKEHOP: A LIGHT-WEIGHT HIGH-PERFORMANCE DEEPFAKE DETECTOR", ICME 2021: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9428361) [Github](https://github.com/hongshuochen/DefakeHop)
* "FSSPOTTER: Spotting Face-Swapped Video by Spatial and Temporal Clues", ICME 2020: [Paper](https://ieeexplore.ieee.org/document/9102914)

### TPAMI

* "DeepFake Detection Based on Discrepancies Between Faces and their Context", TPAMI 2021: [Paper](https://ieeexplore.ieee.org/document/9468380/)
* "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals", TPAMI 2021: [Paper](https://ieeexplore.ieee.org/document/9141516/)

### TIFS

* "Detection of Fake and Fraudulent Faces via Neural Memory Networks", TIFS 2021: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9309253)
* "Preventing DeepFake Attacks on Speaker Authentication by Dynamic Lip Movement Analysis", TIFS 2021: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9298826)

### Other

* "Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors", ICPR 2022: [Paper](https://arxiv.org/abs/2204.08612)
* "Generalized Facial Manipulation Detection with Edge Region Feature Extraction", WACV 2022: [Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Kim_Generalized_Facial_Manipulation_Detection_With_Edge_Region_Feature_Extraction_WACV_2022_paper.pdf)
* "Supervised Contrastive Learning for Generalizable and Explainable DeepFakes Detection": WACV 2022: [Paper](https://openaccess.thecvf.com/content/WACV2022W/XAI4B/papers/Xu_Supervised_Contrastive_Learning_for_Generalizable_and_Explainable_DeepFakes_Detection_WACVW_2022_paper.pdf)
* "Deepfake Caricatures: Amplifying attention to artifacts increases deepfake detection by humans and machines", arXiv 2022: [Paper](https://arxiv.org/pdf/2206.00535.pdf)
* "Seeing is Living? Rethinking the Security of Facial Liveness Verification in the Deepfake Era", arXiv 2022: [Paper](https://arxiv.org/pdf/2202.10673.pdf)
* "FaceSigns: Semi-Fragile Neural Watermarks for Media Authentication and Countering Deepfakes", arXiv 2022: [Paper](https://arxiv.org/pdf/2204.01960.pdf)
* "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization", arXiv 2022: [Paper](https://arxiv.org/pdf/2204.06228.pdf)
* "Crowd–powered Face Manipulation Detection: Fusing Human Examiner Decisions", arXiv 2022: [Paper](https://arxiv.org/pdf/2201.13084.pdf)
* "Face Forgery Detection Based on the Improved Siamese Network", Security and Communication Networks 2022: [Paper](https://www.hindawi.com/journals/scn/2022/5169873/)
* "Audio-Visual Person-of-Interest DeepFake Detection", arXiv 2022: [Paper](https://arxiv.org/abs/2204.03083)
* “DeepFake Detection for Human Face Images and Videos: A Survey”, IEEE Access 2022: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9712265)
* "Deepfake Detection: A Systematic Literature Review", IEEE Access 2022: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9721302)
* "Improving Generalization of Deepfake Detection with Domain Adaptive Batch Normalization", Adversarial Learning for Multimedia 2021: [Paper](https://dl.acm.org/doi/abs/10.1145/3475724.3483603)
* "One Detector to Rule Them All: Towards a General Deepfake Attack Detection Framework", WWW 2021: [Paper](https://arxiv.org/abs/2105.00187)
* "Deepfakes Generation and Detection: State-of-the-art, open challenges, countermeasures, and way forward", arXiv 2021: [Paper](https://arxiv.org/ftp/arxiv/papers/2103/2103.00484.pdf)
* "MSTA-Net: Forgery Detection by Generating Manipulation Trace Based on Multi-scale Self-texture Attention", TCSVT 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9643421)
* "Training Strategies and Data Augmentations in CNN-based DeepFake Video Detection", WIFS 2020: [Paper](https://arxiv.org/pdf/2011.07792.pdf)
* "How Do the Hearts of Deep Fakes Beat? Deep Fake Source Detection via Interpreting Residuals with Biological Signals", IJCB 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9304909)
* "Deepfake Detection using Spatiotemporal Convolutional Networks", arXiv: [Paper](https://arxiv.org/abs/2006.14749)    [Github](https://github.com/oidelima/Deepfake-Detection)
* "A Convolutional LSTM based Residual Network for Deepfake Video Detection", arXiv: [Paper](https://arxiv.org/abs/2009.07480)
* "Spatio-temporal Features for Generalized Detection of Deepfake Videos", submitted to CVIU: [Paper](https://arxiv.org/abs/2010.11844)
* "Exploiting Visual Artifacts to Expose Deepfakes and Face Manipulations", WACVW 2019: [Paper](https://ieeexplore.ieee.org/document/8638330)
* "Interpretable and Trustworthy Deepfake Detection via Dynamic Prototypes", WACV 2021: [Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Trinh_Interpretable_and_Trustworthy_Deepfake_Detection_via_Dynamic_Prototypes_WACV_2021_paper.pdf)
* "MesoNet: a Compact Facial Video Forgery Detection Network", WIFS 2018: [Paper](https://arxiv.org/abs/1809.00888)    [Github](https://github.com/DariusAf/MesoNet)
* "Multi-task Learning For Detecting and Segmenting Manipulated Facial Images and Videos", BATS 2019: [Paper](https://arxiv.org/abs/1906.06876)
* "Use of a Capsule Network to Detect Fake Images and Videos", arXiv: [Paper](https://arxiv.org/abs/1910.12467)
* "What makes fake images detectable? Understanding properties that generalize", arXiv: [Paper](https://arxiv.org/abs/2008.10588)
* "PRRNet: Pixel-Region relation network for face forgery detection", Pattern Recognition 2021: [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320321001370)
* "ForensicTransfer: Weakly-supervised Domain Adaptation for Forgery Detection", arXiv: [Paper](https://arxiv.org/abs/1812.02510)
* "Identity-Driven DeepFake Detection", CoRR 2020: [Paper](https://arxiv.org/abs/2012.03930)
* "Detecting Deep-Fake Videos from Appearance and Behavior", WIFS 2020: [Paper](https://arxiv.org/pdf/2004.14491.pdf%20%7c29.pdf)
* "Exposing Fake Faces Through Deep Neural Networks Combining Content and Trace Feature Extractors", IEEE Access 2021: [Paper](https://ieeexplore.ieee.org/abstract/document/9531572)
* "FaceGuard: Proactive Deepfake Detection", CoRR 2021: [Paper](https://arxiv.org/pdf/2109.05673v1.pdf)
* "DeepFake-o-meter: An Open Platform for DeepFake Detection", SP Workshops 2021: [Paper](https://arxiv.org/abs/2103.02018)
* "DeepfakeUCL: Deepfake Detection via Unsupervised Contrastive Learning", IJCNN 2021: [Paper](https://arxiv.org/abs/2104.11507)
* "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection", CoRR 2021: [Paper](https://arxiv.org/abs/2104.09770)
