# Published by 3d_pose_estimation: computes the 3D pose of the target and the yaw relative to the robot centre and the centre of the target bounding box, so that the robot can face the target.

std_msgs/Header header


# 3D position of the target, in coordinates
geometry_msgs/Point position

# Orientation of the target, in coordinates
float32 yaw