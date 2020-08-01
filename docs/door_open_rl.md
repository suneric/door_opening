# Reinforcement learning for door opening task

install tensorflow
- [tensorflow gpu support "Ubuntu 18.04 (CUDA 10.1)"](https://www.tensorflow.org/install/gpu)
libcudnn7 version should be 7.6.5.32-1+cuda10.1 (use 'table' key to check aviable version)
- then install tensorflow
```
pip install tensorflow==2.1
```
if pip unable to recognize the version, upgrade pip with
```
pip install pip --upgrade
```
- view learning data in tensorboard
```
python -m tensorboard.main --logdir=/path/model
```

install gym
```
pip install gym
```
package dor_trainer,


# Reference
- [gazebo_rl](https://github.com/deePurrobotics/gazebo_rl.git)
- [pursuit_evasion "dev branch"](https://github.com/linZHank/pursuit_evasion)
