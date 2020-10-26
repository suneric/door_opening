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
- dqn without noise
  - success rate: 98/100
  - minimum steps: 17
  - maximum steps: 54
  - average steps: 25
  - lowest cost: 1.154 m
  - highest cost: 3.387 m
  - average cost: 1.722 m
- ppo without noise
  - success rate: 97/100
  - minimum steps: 14
  - maximum steps: 19
  - average steps: 15
  - lowest cost: 1.16 m
  - highest cost: 1.483 m
  - average cost: 1.264 m
- ppo with noise
  - success rate: 96/100
  - minimum steps: 16
  - maximum steps: 26
  - average steps: 18
  - lowest cost: 1.098 m
  - highest cost: 1.542 m
  - average cost: 1.236 m  
