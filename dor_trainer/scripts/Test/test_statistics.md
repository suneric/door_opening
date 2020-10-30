# Trained Model Test for validate robustness, generalization

# Training Physical Environment
## door property
- dimension (width x depth x height) : 0.9144 x 0.04445 x 2.032 meters
- mass: 41.3256 kg
- hinge:
  - spring stiffness: 1
  - spring reference: 1
  - damping: 0.1
  - friction: 0.1
- initial open angle: 0.1 radian    
## mobile base
- dimension (length x width x height): 50 x 50 x 17.5 centimeters
- hook length: 0.25 centimeters
- mass: 34 kg   
- wheel-ground contact friction coefficients:
  - mu1: 0.1
  - mu2: 0.1
- mobile control:
  - type: skid steer drive
  - torque: 30
## scene color
- door color: Gazebo/Yellow
- Wall color: Gazebo/White


# Test
## 100 test cases
### door pulling task
- training performance
![](training_pull.svg)

- policy test

|models |success rate |average steps |minimum steps |maximum steps |average cost |lowest cost | highest cost |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
|dqn without noise | 98% | 25 | 17 | 54 | 1.722 m | 1.154 m | 3.387 m |
|ppo without noise | 97% | 15 | 14 | 19 | 1.264 m | 1.160 m | 1.483 m |
|ppo with noise    | 96% | 18 | 16 | 26 | 1.236 m | 1.098 m | 1.542 m |

- trajectory cost

| dqn without noise | ppo without noise | ppo with noise |
|:---|:---:|---:|
|![](pull_dqn_no_noise.png) | ![](pull_ppo_no_noise.png) | ![](pull_ppo_noise.png) |

### door traversing task
- training performance
![](training_traverse.svg)

- policy test

|models |success rate |average steps |minimum steps |maximum steps |average cost |lowest cost | highest cost | average value |lowest value |highest value|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|dqn without noise | 74% | 26 | 17 | 48 | 1.997 m | 1.378 m | 3.241 m | 95.922 | 68.331 | 115.080 |
|ppo without noise | | | | | | | | | | |
|ppo with noise    | 87% | 23 | 17 | 47 | 1.953 m | 1.525 m | 2.906 m | 92.125 | 42.564 | 105.229 |

- trajectories

| dqn without noise | ppo without noise | ppo with noise |
|:---|:---:|---:|
|![](traverse_dqn_no_noise.png) |  |  |

### door pushing task
- training performance
![](training_push.svg)

- policy test

|models |success rate |average steps |minimum steps |maximum steps |average cost |lowest cost | highest cost |average value |lowest value |highest value|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|dqn without noise | 99% | 18 | 16 | 20 | 1.905 m | 1.732 m| 2.064 m | 75.001 | 75.001 | 75.001|
|ppo without noise | 98% | 17 | 16 | 19 | 1.953 m | 1.781 m | 2.295 m | 98.263 | 93.044 | 102.513 |
|ppo with noise    | 98% | 17 | 17 | 21 | 1.937 m | 1.840 m | 2.076 m | 98.516 | 93.192 | 106.429 |

- trajectories

|dqn without noise |ppo without noise |ppo with noise |
|:---|:---:|---:|
| link missing | ![](push_ppo_no_noise.png) | ![](push_ppo_noise.png) |
