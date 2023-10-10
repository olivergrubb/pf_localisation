from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy
import numpy as np
from . resample import *
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
        
        # ----- Initial particle cloud parameters
        self.NUMBER_OF_PARTICLES = 400
        self.CLOUD_X_NOISE = 4
        self.CLOUD_Y_NOISE = 4
        self.CLOUD_ROTATION_NOISE = 1
        
        # ----- Resample particle cloud parameters
        self.RESAMPLE_X_NOISE = 0.1
        self.RESAMPLE_Y_NOISE = 0.1
        self.RESAMPLE_ROTATION_NOISE = 0.1
        
        self.STOCHASTIC_RATIO = 0.5
        self.RANDOM_EXPLORATION_RATIO = 0.3
        self.EDUCATED_ESTIMATE_RATIO = 0.2

       
    def initialise_particle_cloud(self, initialpose):
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

        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        # Get likelihood weighting of each pose
        weighted_poses = [(pose, self.sensor_model.get_weight(scan, pose)) for pose in self.particlecloud.poses]
        
        
        # Multinomial resampling
        
        #new_poses = multinomial_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO))
        
        
        # Residual resampling
        
        #new_poses = residual_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO))
        

        # Systematic resampling
        
        #new_poses = systematic_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO))
        

        # Stratified resampling
        
        #new_poses = stratified_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO))
        

        # Adaptive resampling (Still testing)
        
        #new_poses = adaptive_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO), 0.5, self.NUMBER_OF_PARTICLES/2)
        

        # reguralized resampling
        
        new_poses = reguralized_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO), 0.5)
        
        
        # Smoothed resampling
        
        #new_poses = smoothed_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO))
        

        # Residual stratisfied resampling
        
        #new_poses = residual_stratified_resampling(list(zip(*weighted_poses))[0], list(zip(*weighted_poses))[1], math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO))
        
        """ 
        # Original Ollie resampling code
        # Resample particlecloud
        new_poses = []
        
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
        
        # Resample select portion of poses
        for j in range(0, math.floor(self.NUMBER_OF_PARTICLES * self.STOCHASTIC_RATIO)):
            while threshold > cdf[i]:
                i += 1
            new_poses.append(weighted_poses[i][0])
            threshold += 1/len(weighted_poses)
        """
        # Add some estimate poses
        for k in range(0, math.floor(self.NUMBER_OF_PARTICLES * self.EDUCATED_ESTIMATE_RATIO)):
            rndx = random.normalvariate(0, 1)
            rndy = random.normalvariate(0, 1)
            rndr = random.normalvariate(0, 1)
            
            # Adding informed estimates based on highest weighted pose
            pose = Pose()
            pose.position.x = max(weighted_poses, key=lambda x: x[1])[0].position.x + rndx * self.RESAMPLE_X_NOISE
            pose.position.y = max(weighted_poses, key=lambda x: x[1])[0].position.y + rndy * self.RESAMPLE_Y_NOISE
            pose.position.z = 0
            pose.orientation = rotateQuaternion(max(weighted_poses, key=lambda x: x[1])[0].orientation, rndr * self.RESAMPLE_ROTATION_NOISE)

            new_poses.append(pose)
        
        # Add some random exploratory poses
        for l in range(0, math.floor(self.NUMBER_OF_PARTICLES * self.RANDOM_EXPLORATION_RATIO)):
            # Based off map lying on grid line y = 30 - x with thickness of approx 10 at widest
            rnd = random.normalvariate(0, 5)
            
            pose = Pose()
            # Not 0, 30 as lots of points lie off map
            x = random.uniform(3, 27)
            y = (30 - x) + rnd
            
            pose.position.x = x
            pose.position.y = y
            pose.orientation.w = random.uniform(0, 1)
            pose.orientation.z = random.uniform(-1, 1)
            
            new_poses.append(pose)
            
        # Update particlecloud
        self.particlecloud.poses = new_poses
        
    def estimate_pose(self):
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
        
        xy_values = np.array([(pose.position.x, pose.position.y) for pose in self.particlecloud.poses])
        wz_values = np.array([(pose.orientation.w, pose.orientation.z) for pose in self.particlecloud.poses])
        
        q1 = np.percentile(xy_values, 25, axis=0)
        q3 = np.percentile(xy_values, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Remove outliers
        
        non_outliers = xy_values[(xy_values >= lower_bound).all(axis=1) & (xy_values <= upper_bound).all(axis=1)]
        
        cluster_centroid = np.mean(non_outliers, axis=0)
        mean_orientation = np.mean(wz_values, axis=0)
        
        estimated_pose = Pose()
        
        estimated_pose.position.x = cluster_centroid[0]
        estimated_pose.position.y = cluster_centroid[1]
        estimated_pose.orientation.w = mean_orientation[0]
        estimated_pose.orientation.z = mean_orientation[1]
        
        return estimated_pose
