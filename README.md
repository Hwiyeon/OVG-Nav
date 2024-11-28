# OVG-Nav: Commonsense-Aware Object Value Graph for Object Goal Navigation

This repository contains a Pytorch implementation of our RA-L (2024), ICRA@40 (2024) paper:

### *[Commonsense-Aware Object Value Graph for Object Goal Navigation](https://ieeexplore.ieee.org/document/10478188)* <br>
Hwiyeon Yoo, Yunho Choi, Jeongho Park, Songhwai Oh <br>
Seoul National University

## Abstract
Object goal navigation (ObjectNav) is the task of finding a target object in an unseen environment. It is one of the fundamental challenges in visual navigation as it requires both structural and semantic understanding. In this paper, we present OVG-Nav, a novel ObjectNav framework that leverages a topological graph structure called object value graph (OVG), which contains visual observations and commonsense prior knowledge. The high-level planning of OVG-Nav prioritizes subgoal nodes for exploration based on a metric called object value, which reflects the closeness to the target object. Here, we propose OVGNet, a model designed to predict the object values of each node of an OVG using observed features along with commonsense knowledge. The structure of highlevel planning using OVG and low-level action decisions reduces sensitivity to accumulating sensor noises, leading to robust navigation performance. Experimental results show that OVGNav outperforms the baseline in success rate (SR) and success rate weighted by path length (SPL) in the MP3D dataset both in accurate sensing and noisy sensing. In addition, we show that the OVG-Nav can be transferred to the real-world robot successfully.
## Example

![ExampleVideo](demo/OVG-Nav_image.gif)

Note that the top-down map and pose information are only used for visualization, not for the graph generation. 



## Installation



The source code is developed and tested in the following setting. 
- Python 3.7
- pytorch 1.9.1
- detectron2
- habitat-sim 0.2.2
- habitat 0.2.2

Please refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim.git) and [habitat-lab](https://github.com/facebookresearch/habitat-lab.git) for installation.

To start, we prefer creating the environment using conda:

```
conda create -n OVG-Nav python=3.7
conda activate OVG-Nav
pip install transformers
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch -y
pip install ftfy regex tqdm

pip install git+https://github.com/openai/CLIP.git
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
pip install gitpython
conda install habitat-sim=0.2.2 withbullet headless -c conda-forge -c aihabitat
pip install opencv-python scikit-learn scikit-image scikit-fmm quaternion

git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout v0.2.2
pip install -e .
pip install -r requirements.txt
python setup.py develop --all
pip install seaborn
```


## Citation
If you find this code useful for your research, please consider citing:
```Bibtex
@article{yoo2024commonsense,
    title={Commonsense-Aware Object Value Graph for Object Goal Navigation},
    author={Yoo, Hwiyeon and Choi, Yunho and Park, Jeongho and Oh, Songhwai},
    journal={IEEE Robotics and Automation Letters},
    year={2024},
    publisher={IEEE}
}
```

