# MTNet - Mobility Tree Network for Next POI Recommendation

This is the code implementation for our paper [Learning Time Slot Preferences via Mobility Tree for Next POI Recommendation  (AAAI2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28697).

## Cite our work
```
@article{Huang_Pan_Cai_Zhang_Yuan_2024,
  title={Learning Time Slot Preferences via Mobility Tree for Next POI Recommendation}, volume={38},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/28697},
  DOI={10.1609/aaai.v38i8.28697},
  abstractNote={Next Point-of-Interests (POIs) recommendation task aims to provide a dynamic ranking of POIs based on users’ current check-in trajectories. The recommendation performance of this task is contingent upon a comprehensive understanding of users’ personalized behavioral patterns through Location-based Social Networks (LBSNs) data. While prior studies have adeptly captured sequential patterns and transitional relationships within users’ check-in trajectories, a noticeable gap persists in devising a mechanism for discerning specialized behavioral patterns during distinct time slots, such as noon, afternoon, or evening. In this paper, we introduce an innovative data structure termed the ``Mobility Tree’’, tailored for hierarchically describing users’ check-in records. The Mobility Tree encompasses multi-granularity time slot nodes to learn user preferences across varying temporal periods. Meanwhile, we propose the Mobility Tree Network (MTNet), a multitask framework for personalized preference learning based on Mobility Trees. We develop a four-step node interaction operation to propagate feature information from the leaf nodes to the root node. Additionally, we adopt a multitask training strategy to push the model towards learning a robust representation. The comprehensive experimental results demonstrate the superiority of MTNet over eleven state-of-the-art next POI recommendation models across three real-world LBSN datasets, substantiating the efficacy of time slot preference learning facilitated by Mobility Tree.}, number={8},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Huang, Tianhao and Pan, Xuan and Cai, Xiangrui and Zhang, Ying and Yuan, Xiaojie},
  year={2024}, month={Mar.},
  pages={8535-8543} }
```

## Requirements
```
pip install -r requirements.txt
```

## Training, Validation and Testing
- Adjust the configuration in `config.py`  
- Run `main.py`

## FAQ
- Q: How about the total running time?    
  A: NYC dataset runs for about 1 hour on NVIDIA GeForce RTX 2060. For Gowalla and TKY datasets, this time is about 2.5 hours.
  
- Q: Where are the TKY and Gowalla dataset?    
  A: See [Issue](https://github.com/Skyyyy0920/MTNet/issues/4).
