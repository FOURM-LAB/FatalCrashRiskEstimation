## Unveiling Roadway Hazards: Enhancing Fatal Crash Risk Estimation Through Multiscale Satellite Imagery and Self-Supervised Cross-Matching [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10313931)] [[Dataset]()]
#### [Gongbo Liang](http://www.gb-liang.com), [Janet Zulu](https://www.linkedin.com/in/janetzulu/), [Xin Xing](https://xtrigold.github.io), [Nathan Jaocbs](https://jacobsn.github.io/) 

### Abstract
Traffic accidents threaten human lives and impose substantial financial burdens annually. Accurate estimation of accident fatal crash risk is crucial for enhancing road safety and saving lives. This paper proposes an innovative approach that utilizes multi-scale satellite imagery and self-supervised learning for fatal crash risk estimation. By integrating multi-scale imagery, our network captures diverse features at different scales, encompassing observations of surrounding environmental factors in low-resolution images that cover larger areas and learning detailed ground-level information from high-resolution images. One advantage of our work is its sole reliance on satellite imagery data, making it an efficient and practical solution, especially when other data modalities are unavailable. With the ability to accurately estimate fatal crash risk, our method exhibits a potential for enhancing road safety, optimizing infrastructure planning, preventing accidents, and ultimately saving lives. 

### 🛠️ Installation
1. Create a Conda environment for the code:
   ```
   conda create --name FCRE python=3.8
   ```
2. Activate the environment:
   ```
   conda activate FCRE
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### 👨‍💻 Getting Started 
#### 🔥 Training Models
1. Setup all the parameters of interest inside `config.py` before launching the training script.
2. Run pre-training by calling:
    ```bash
    python pretrain.py
    ```
3. Run fine-tuning by calling:
    ```bash
    python finetune.py
    ```

#### ❄️ Pretrained Models
Download pretrained models from the given links below:

|Model Type|Download Url|
|----------|--------|
|CVE-MAE|[Link](https://wustl.box.com/s/o1ooaunhaym7v1qj3yzj3vof0lskxyha)|
|CVE-MAE-Meta| [Link](https://wustl.box.com/s/fudo44eznjwejcp3vql14by20rqqayfy)|
|CVM-MAE| [Link](https://wustl.box.com/s/xuezslrnjxyz1d1ngtzvnm5ck2il4nx8)|
|CVM-MAE-Meta| [Link](https://wustl.box.com/s/c3nfbdmcigiogqskemyc4h5soveiya8n)|

### 📑 Citation
```bibtex
@ARTICLE{liang2024unveiling,
  author={Liang, Gongbo and Zulu, Janet and Xing, Xin and Jacobs, Nathan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Unveiling Roadway Hazards: Enhancing Fatal Crash Risk Estimation Through Multiscale Satellite Imagery and Self-Supervised Cross-Matching}, 
  year={2024},
  volume={17},
  pages={535-546},
  doi={10.1109/JSTARS.2023.3331438}}
```
