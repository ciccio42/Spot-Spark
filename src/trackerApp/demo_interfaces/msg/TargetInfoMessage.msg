# Pubblicato dal nodo di Percezione Solo sui frame in cui il target è confermato. Usato da pose_3d_estimation

std_msgs/Header header

#bounding box del target con convenzione xyxy
float32[4] bounding_box


# Distanza in metri gia calcolata 
float32 depth_m

# Nome del topic della camera di immagine e di depth
string camera_rgb_topic
string camera_depth_topic