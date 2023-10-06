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
        self.ODOM_ROTATION_NOISE = 1 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 1 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 1 # Odometry model y axis (side-to-side) noise

        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
        # ----- Particle cloud parameters
        self.NUMBER_OF_PARTICLES = 200
        self.CLOUD_X_NOISE = 1
        self.CLOUD_Y_NOISE = 1
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
            
            rnd = random.normalvariate(0, 1)
            
            pose = Pose()
            pose.position.x = initialpose.pose.pose.position.x + rnd * self.CLOUD_X_NOISE
            pose.position.y = initialpose.pose.pose.position.y + rnd * self.CLOUD_X_NOISE
            pose.position.z = 0
            pose.orientation = rotateQuaternion(pose.orientation, rnd * self.CLOUD_ROTATION_NOISE)
            
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
        pass

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
        pass
