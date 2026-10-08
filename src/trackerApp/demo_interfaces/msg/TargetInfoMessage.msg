# Published by the Perception node only on frames where the target is confirmed. Used by pose_3d_estimation

std_msgs/Header header

# target bounding box, xyxy convention
float32[4] bounding_box


# Distance in metres, already computed
float32 depth_m

# Name of the image and depth camera topics
string camera_rgb_topic
string camera_depth_topic