# Pubblicazione fatta da 3d_pose_estimation, calcola la posa 3d del target e lo yaw in relazione al centro del robot e il centro boundig box del target, per fare in modo che il robot possa guardare il target.

std_msgs/Header header


#Posizione 3D del target in coordinate 
geometry_msgs/Point position

# Orientamento del target in coordinate 
float32 yaw