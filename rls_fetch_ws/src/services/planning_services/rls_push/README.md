## RLS Push Meta Packages
This meta package includes ros packages for deep planar pushing

### Meta Packages Summary
The meta pacakges is comprised of three ros packages
* ```rls_push```: Push-Net main programs
* ```rls_push_vision```: perception modules for pcl segmentation, image processing and etc
* ```rls_push_msg```: defined msg, srv and action used by the previous two packages

### Tabletop Segmentation
Given a single object on a planar surface, Fetch uses AR marker to localize itself with the planar surface. The tabletop segmentation service takes in a definition of a table and PCL, and returns a set of points of object's PCL in the type of ```visualization_msgs/Marker```.

* make sure the AR marker is visible to Fetch
* launch AR marker detector (A)
```$ roslaunch aruco_ros single_marker_detect.launch```
* launch tabletop service and publish static transformation (B)
```$ roslaunch rls_push_vision fetch_tabletop_segmentation.launch```
* run test program
```$ rosrun rls_push_vision segment_test.py```

### Push-Net
Given an object on a flat plane, Fetch pushes it from an initial configuration to a goal. The goal is specified either in 2D position or 2D orientation.

* launch A and B above
* launch moveit
```roslaunch fetch_moveit_config move_group.launch```
* run push-net
```python push_net_main.py```


