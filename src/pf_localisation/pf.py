from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
import random

import time


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters TODO Tune noise values
        self.ODOM_ROTATION_NOISE = 0.1 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.1 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.1 # Odometry model y axis (side-to-side) noise

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
        # ----- Particle cloud parameters
        self.NUMBER_OF_PARTICLES = 300
        self.CLOUD_X_NOISE = 0.5
        self.CLOUD_Y_NOISE = 0.5
        self.CLOUD_ROTATION_NOISE = 1

       
    def initialise_particle_cloud(self, initialpose):
        rospy.loginfo("In initialise_particle_cloud")
        rospy.loginfo(initialpose)
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        
        self.particlecloud = PoseArray()
        
        for i in range(0, self.NUMBER_OF_PARTICLES + 1):
            
            rndx = random.normalvariate(0, 1)
            rndy = random.normalvariate(0, 1)
            rndr = random.normalvariate(0, 1)
            
            pose = Pose()
            pose.position.x = initialpose.pose.pose.position.x + rndx * self.CLOUD_X_NOISE
            pose.position.y = initialpose.pose.pose.position.y + rndy * self.CLOUD_Y_NOISE
            pose.position.z = 0
            pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, rndr * self.CLOUD_ROTATION_NOISE)

            self.particlecloud.poses.append(pose)
        
        return self.particlecloud

 
    
    def update_particle_cloud(self, scan):
        rospy.loginfo("In update_particle_cloud")
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        # Get likelihood weighting of each pose
        weighted_poses = [(pose, self.sensor_model.get_weight(scan, pose)) for pose in self.particlecloud.poses]
        
        # Resample particlecloud
        new_poses = PoseArray()
        
        # Calculate normaliser for weights
        sum_of_weights = 0
        for i in range(0, len(weighted_poses)):
            sum_of_weights += weighted_poses[i][1]
        
        normaliser = 1 / sum_of_weights
        
        cdf = [weighted_poses[0][1] * normaliser]
        
        for i in range(1, len(weighted_poses)):
            cdf.append(cdf[i-1] + weighted_poses[i][1] * normaliser)

        # Select starting threshold
        threshold = random.uniform(0, 1/len(weighted_poses))
        i = 0
        
        for j in range(0, self.NUMBER_OF_PARTICLES):
            while threshold > cdf[i]:
                i += 1
            new_poses.poses.append(weighted_poses[i][0])
            threshold += 1/len(weighted_poses)
            
        # Update particlecloud
        self.particlecloud = new_poses

    def estimate_pose(self):
        rospy.loginfo("In estimate_pose")
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
         
        # Calculate mean x and y
        mean_x = sum(pose.position.x for pose in self.particlecloud.poses) / len(self.particlecloud.poses)
        mean_y = sum(pose.position.y for pose in self.particlecloud.poses) / len(self.particlecloud.poses)
        
        # Calculate standard deviation for x and y
        x_cum_sq_deviation = 0
        y_cum_sq_deviation = 0
        
        for pose in self.particlecloud.poses:
            x_cum_sq_deviation += (pose.position.x - mean_x) ** 2
            y_cum_sq_deviation += (pose.position.y - mean_y) ** 2
        
        x_std_dev = math.sqrt(x_cum_sq_deviation / len(self.particlecloud.poses))
        y_std_dev = math.sqrt(y_cum_sq_deviation / len(self.particlecloud.poses))
        
        if x_std_dev > 0.001 or y_std_dev > 0.001:
            
            # Thresholds based on position lying 1 standard deviation from mean
            x_lower_threshold = mean_x - x_std_dev
            x_upper_threshold = mean_x + x_std_dev
            y_lower_threshold = mean_y - y_std_dev
            y_upper_threshold = mean_y + y_std_dev
        
            # Filter poses based on the threshold
            filtered_poses = [pose for pose in self.particlecloud.poses if
                  (pose.position.x < x_lower_threshold or pose.position.x > x_upper_threshold) or
                  (pose.position.y < y_lower_threshold or pose.position.y > y_upper_threshold)]
            rospy.loginfo(f"x mean: {mean_x}")
            rospy.loginfo(f"Y mean: {mean_x}")
            rospy.loginfo(f"x sdv: {x_std_dev}")
            rospy.loginfo(f"y sdv: {y_std_dev}")
            print("Original number of poses:", len(self.particlecloud.poses))
            print("Number of filtered poses:", len(filtered_poses))
        else:
            filtered_poses = self.particlecloud.poses
            
        # Calculate average pose from filtered poses
        estimated_pose = Pose()
        estimated_pose.position.x = sum(pose.position.x for pose in filtered_poses) / len(filtered_poses) if len(filtered_poses) else 1
        estimated_pose.position.y = sum(pose.position.y for pose in filtered_poses) / len(filtered_poses) if len(filtered_poses) else 1
        estimated_pose.orientation.z = sum(pose.orientation.z for pose in filtered_poses) / len(filtered_poses) if len(filtered_poses) else 1
        estimated_pose.orientation.w = sum(pose.orientation.w for pose in filtered_poses) / len(filtered_poses) if len(filtered_poses) else 1
        
        return estimated_pose
