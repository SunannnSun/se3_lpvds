import os
import rosbag
import numpy as np
from datetime import datetime

bag_name = "demo_2024-08-07-11-43-13"
bag_path = "/home/figueroa-lab11/franka_ws/src/demo_recorded/" + bag_name + ".bag"
output_file = "dataset/kine/" + bag_name

bag = rosbag.Bag(bag_path)
timestamps = []
topic_data = []

for topic, msg, t in bag.read_messages(topics=["/franka_state_controller/O_T_EE"]):
    # Assuming the message is a list or array of floats, change this if the message structure is different

    position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    ori = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    state = {'position':position, 'orientation':ori, 't':t.to_sec()}
    topic_data.append(state)


bag.close()

topic_array = np.array(topic_data)

np.save(output_file, topic_array)