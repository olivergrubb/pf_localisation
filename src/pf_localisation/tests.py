import rosbag
import numpy as np
import matplotlib.pyplot as plt

"""
This test records the estimate pose and ground truth
pose of the robot at each time step and calculates the
root mean square error between the two.
"""

def rmse_of_ground_truth_and_estimate(test_bag):
    raw_ground_truth_poses = []
    raw_estimate_poses = []
    raw_clock = []

    with rosbag.Bag(test_bag) as bag:
        for msg in bag.read_messages(topics=['/base_pose_ground_truth', '/amcl_pose', '/clock']):
            if msg.topic == "/base_pose_ground_truth":
                raw_ground_truth_poses.append(msg.message)
            elif msg.topic == "/amcl_pose":
                raw_estimate_poses.append(msg.message)
            else:
                raw_clock.append(msg.message)

    time_stamps_with_pose = [(f"{pose.header.stamp.secs}.{pose.header.stamp.nsecs}", pose) for pose in raw_ground_truth_poses]
    error_values =[]
    time_values = []

    current_time = 0
    first_time = 0.0

    for stamp in time_stamps_with_pose:
        for pose in raw_estimate_poses:
            if f"{pose.header.stamp.secs}.{pose.header.stamp.nsecs}" == stamp[0] and float(stamp[0]) > current_time:
                if first_time == 0.0:
                    first_time = float(stamp[0])
                x_pos_err = stamp[1].pose.pose.position.x - (pose.pose.pose.position.x - 15)
                y_pos_err = stamp[1].pose.pose.position.y - (pose.pose.pose.position.y -15)
                error_values.append(np.sqrt(x_pos_err**2 + y_pos_err**2))
                time_values.append(float(stamp[0]) - first_time)
                current_time = float(stamp[0])
    
    
    return time_values, error_values

def average_time(time_file):
    times = []
    with open(time_file) as f:
        for line in f:
            times.append(float(line))
    return np.mean(times)

#print(average_time("time_test_results/estimate_dbspan_results.txt"))
time, error = rmse_of_ground_truth_and_estimate("Test bag file")
#time2, error2 = rmse_of_ground_truth_and_estimate("/home/ollie/catkin_ws/src/pf_localisation/src/pf_localisation/bag_files/residual_mpro_ground_truth_poses_and_estimated_poses.bag")
#time3, error3 = rmse_of_ground_truth_and_estimate("/home/ollie/catkin_ws/src/pf_localisation/src/pf_localisation/bag_files/systematic_mpro_ground_truth_poses_and_estimated_poses.bag")
plt.plot(time, error, label="", color="")
#plt.plot(time2, error2, label="Residual", color="red")
#plt.plot(time3, error3, label="Systemtic", color="green")
plt.xlabel('Time')
plt.ylabel('Root Mean Square Error')
plt.title('Graph Title')
plt.legend()
plt.grid(True)
plt.show()