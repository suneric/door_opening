# Path Planning with 2D Laser scanner
## Build a Map
A map can be built with ROS slam gmapping package with 2D laser scan.
```
roslaunch dor_control build_room_map.launch
```
After the auto map build is complete, save the map with
```
rosrun map_server map_saver -f [/path/mymap.yaml]
```

## Localization
the localization can be achieved by using ROS amcl package.

## Path planning on an exisiting map
The path planning can be done by using ROS move_base along with global planner and local planner.
```
roslaunch dor_gazebo execute_tasks.launch
```


## References
- [ROS slam gmapping](http://wiki.ros.org/gmapping)
- [ROS amcl](http://wiki.ros.org/amcl?distro=noetic)
- [ROS map server](http://wiki.ros.org/map_server)
- [ROS move base](http://wiki.ros.org/move_base)
- [ROS Navigatin Tuning guide](http://kaiyuzheng.me/documents/navguide.pdf)
