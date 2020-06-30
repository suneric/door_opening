# door_opening
simulate opening a door by a mobile robot in gazebo

## door handle detection
follow [repo](https://github.com/mengqlTHU/DoorDetector) and download [weights](https://drive.google.com/file/d/1i9E9pTPN5MtRxgBJWLnfQl2ypCv92dXk/view)
and put the weights in folder 'dor_control/classfier/yolo'


## reference
1. [setup and configuration of the navigation stack on a robot](https://wiki.ros.org/navigation/Tutorials/RobotSetup)
2. [How to make better maps using gmapping](https://answers.ros.org/question/269280/how-to-make-better-maps-using-gmapping/?answer=269293#post-id-269293)


## Navigation Stack
To navigate a robot, we need a map, a localization module, a path planning module. These components are sufficient if the map fully reflects the envrionment, the envrionment is static and there are no errors in the estimation. However, the envrionment changes (e.g. opening/closing doors), it is dynamic (things might appear/disapper from the perception range of the robot) and the estimation is "noisy". Thus we need to complement the design with other components that address these issues, namely obstable-detection/avoidance, local map refinement, based on the most recent sensor reading.
- Building a map: ROS uses GMapping which implements a particle filter to track the robot trajectories. you need to record a bag with /odom, /scan and /tf while driving the robot around in the envrionment, play the bag and the gmapping-node and save it. (the map is an occupancy map and it is represented as an image showing the blueprint of the enviornment and a confiuguration file '.yaml' that gives meta information about the map)
- Localizing a robot: ROS implements the Adaptive Monte Carlo Localization algorithm (AMCL) which uses a particle filter to track the position of the robot, with each pose presented by a particle which is moved according to movement measured by the odometry. The localization is integrated in ROS by emmiting a transform from a map-frame to the odom frame that corrects the odometry. To query the robot position according to the localization you should ask the transform of base_footprint in the map frame. (ACML relies on a laser)  
