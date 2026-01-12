# Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal

[Jinpei Guo](https://jp-guo.github.io/), [Zheng Chen](https://zhengchen1999.github.io/), [Wenbo Li](https://fenglinglwb.github.io/),  [Yong Guo](https://www.guoyongcs.com/), and [Yulun Zhang](http://yulunzhang.com/),  "Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal", ICCV, 2025

[[paper](https://arxiv.org/pdf/2502.09873)] [[supplementary material](https://github.com/jp-guo/CODiff/releases/tag/v1)]

#### 🔥🔥🔥 News

- **2025-02-14:** This repo is released.

---

> **Abstract:** Diffusion models have demonstrated remarkable success in image restoration tasks. However, their multi-step denoising process introduces significant computational overhead, limiting their practical deployment. Furthermore, existing methods struggle to effectively remove severe JPEG artifact, especially in highly compressed images. To address these challenges, we propose CODiff, a compression-aware one-step diffusion model for JPEG artifact removal. The core of CODiff is the compression-aware visual embedder (CaVE), which extracts and leverages JPEG compression priors to guide the diffusion model. Moreover, We propose a dual learning strategy for CaVE, which combines explicit and implicit learning. Specifically, explicit learning enforces a quality prediction objective to differentiate low-quality images with different compression levels. Implicit learning employs a reconstruction objective that enhances the model's generalization. This dual learning allows for a deeper and more comprehensive understanding of JPEG compression. Experimental results demonstrate that CODiff surpasses recent leading methods in both quantitative and visual quality metrics.

![](figs/codiff.png)

---

<!-- ![vis-main-top](images/vis-main-top.png) -->

CODiff reconstruction demos on JPEG images with QF=1

[<img src="figs/comp/img_029_1.png" width="270px"/>](https://imgsli.com/MzQ5NDc4) [<img src="figs/comp/img_081_1.png" width="270px"/>](https://imgsli.com/MzQ5NDc5) [<img src="figs/comp/img_082_1.png" width="270px"/>](https://imgsli.com/MzQ5NDgw)

CODiff reconstruction demos on JPEG images with QF=5

[<img src="figs/comp/img_017_5.png" width="270px"/>](https://imgsli.com/MzQ5MzYx) [<img src="figs/comp/img_053_5.png" width="270px"/>](https://imgsli.com/MzQ5MzYz) [<img src="figs/comp/img_054_5.png" width="270px"/>](https://imgsli.com/MzQ5MzY0)

CODiff reconstruction demos on JPEG images with QF=10

[<img src="figs/comp/img_001_10.png" width="270px"/>](https://imgsli.com/MzQ5NDgx) [<img src="figs/comp/img_065_10.png" width="270px"/>](https://imgsli.com/MzQ5NDgy) [<img src="figs/comp/img_070_10.png" width="270px"/>](https://imgsli.com/MzQ5NDgz)

---

## Contents

- [x] [Setup](#Setup)
- [x] [Training](#Training)
- [x] [Testing](#Testing)
- [x] [Results](#Results)
- [x] [Citation](#Citation)
- [x] [Acknowledgements](#Acknowledgements)

---

## <a name="setup"></a> Setup

### Environment
The implementation is primarily developed on top of [OSEDiff](https://github.com/cswry/OSEDiff)'s code foundation.
```aiignore
conda env create -f environment.yml
conda activate codiff
```

### Models
Please download the following models and place them in the `model_zoo` directory.
1. [SD-2.1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
2. [CODiff](https://drive.google.com/file/d/1SHZQs8fu3K419jlENoY8qdTKbjAOL0i7/view?usp=drive_link)
3. [CaVE](https://drive.google.com/file/d/1SZo5UMSMEcYy4Wr-OIQgTk8hUXiFMIbK/view?usp=drive_link)

---

## <a name="training"></a> Training
Training consists of two stages. In the first stage, we train CaVE. In the second stage, we freeze the parameters of CaVE and fine-tune CODiff's UNet with LoRA.
### First Stage
Update the training [configuration file](https://github.com/jp-guo/CODiff/tree/main/options/cave.json) with appropriate values, then run:
```aiignore
python main_train_cave.py
```

### Second Stage
Update the [configuration file](https://github.com/jp-guo/CODiff/tree/main/options/codiff.json) with appropriate values, specify the CaVE checkpoint from the first stage in [train_codiff.sh](https://github.com/jp-guo/CODiff/tree/main/train_codiff.sh), 
and launch training:
```aiignore
bash train_codiff.sh
```

---

## <a name="Testing"></a> Testing

Specify the paths to the CaVE and CODiff checkpoints, as well as the dataset directory in [test_codiff.sh](https://github.com/jp-guo/CODiff/tree/main/test_codiff.sh), then run:
```aiignore
bash test_codiff.sh
```

---

## <a name="results"></a> Results

We achieved state-of-the-art performance on LIVE-1, Urban100 and DIV2K-val datasets. Detailed results can be found in the paper.

<details>
<summary>&ensp;Quantitative Comparisons (click to expand) </summary>
<li> Quantitative results on LIVE-1 dataset from the main paper. 
<p align="center">
<img src="figs/live1-table.png" >
</p>
</li>
<li> Quantitative results on Urban100 dataset from the main paper. 
<p align="center">
<img src="figs/urban100-table.png" >
</p>
</li>
<li> Quantitative results on DIV2K-val dataset from the main paper. 
<p align="center">
<img src="figs/div2k-val-table.png" >
</p>
</li>
</details>
<details open>
<summary>&ensp;Visual Comparisons (click to expand) </summary>
<li> Visual results on LIVE-1 dataset from the main paper.
<p align="center">
<img src="figs/live1-fig.png">
</p>
</li>
<li> Visual results on Urban100 dataset from the main paper.
<p align="center">
<img src="figs/urban100-fig.png" >
</p>
</li>
<li> Visual results on DIV2K-val dataset from the main paper.
<p align="center">
<img src="figs/div2k-val-fig.png" >
</p>
</li>
</details>

## <a name="citation"></a> Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@article{guo2025compression,
    title={Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal},
    author={Guo, Jinpei and Chen, Zheng and Li, Wenbo and Guo, Yong and Zhang, Yulun},
    journal={arXiv preprint arXiv:2502.09873},
    year={2025}
}
```

## <a name="acknowledgements"></a> Acknowledgements

This code is built on [FBCNN](https://github.com/jiaxi-jiang/FBCNN) and [OSEDiff](https://github.com/cswry/OSEDiff).
