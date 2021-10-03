## Visuomotor Affordance Learning (VAL)

Paper: http://arxiv.org/abs/2106.00671

Website: https://sites.google.com/view/val-rl

### Repos
- bullet-manipulation: https://github.com/anair13/bullet-manipulation-affordances

### Branches
- bullet-manipulation: rig_affordances
- bullet-objects: sasha-master

### Scripts
- examples/val/full1.py
- examples/val/pretrained1.py (for top drawer, bottom drawer [button])
- examples/val/pretrained_objects1.py (for tray, pnp)

### Datasets
Pretrained VQVAE: https://drive.google.com/file/d/1ohfdGOi6zJc8nxz1Bp9c_XXdhNbH8flS/view?usp=sharing

Premade data is available - ask Ashvin for access.

Object (tray, pnp) Experiment:
- Dataset Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/3dof_gr_demo_collector.py
- Goal Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/rig_presample_goals.py

Drawer + Button Experiment:
- Dataset Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/3dof_afford_demo_collector.py
- Goal Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/rig_afford_presample_goals.py

Tray Experiment:
- Dataset Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/3dof_tray_demo_collector.py
- Goal Collector: https://github.com/JannerM/bullet-manipulation/blob/rig_affordances/shapenet_scripts/rig_tray_presample_goals.py

### Baselines:
- VQVAE: experiments/sasha/awac_exps/goal_reaching/combined.py
- CCVAE: experiments/sasha/awac_exps/affordance_baseline/ccvae.py
- VAE: experiments/sasha/awac_exps/affordance_baseline/vae.py
