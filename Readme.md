#### Q-function Self ATtention  （Q-SAT） 

This codebase accompanies paper <Q-SAT:  Value Factorization with Self-Atention for Deep Multi-Agent Reinforcement Learning>.

Q-SAT is written based on  [PyMARL](https://github.com/oxwhirl/pymarl) codebases which are open-sourced.

##### Installation instructions

1. Install the following environments from corresponding links:

- Level Based Foraging (LBF): https://github.com/semitable/lb-foraging
- StarCraft Multi-Agent Challenge (SMAC): https://github.com/oxwhirl/smac
- Google Research Football (GRF): https://github.com/google-research/football

2. Install [PyMARL](https://github.com/oxwhirl/pymarl)  as instructed.

##### Run an experiment

Note that *regularization* here corresponds to hyper-parameter *k* in paper.

```python

# VDN-SAT
cd q-sat-master
python3 src/main.py --config=vdn_sat --env-config=sc2 with env_args.map_name=corridor regularization=0.0005

# QMIX-SAT
cd q-sat-master
python3 src/main.py --config=qmix_sat --env-config=sc2 with env_args.map_name=corridor regularization=0.0005

# QPLEX-SAT
cd q-sat-qplex
python3 src/main.py --config=qplex_sat --env-config=sc2 with env_args.map_name=corridor regularization=0.0005

# Q-SAT-Net
cd q-sat-master
python3 src/main.py --config=q_sat_0 --env-config=sc2 with env_args.map_name=corridor regularization=0.0005
```

