from geometry_msgs.msg import Pose
import numpy as np
import rospy

def mean_pose(poses):
    xy_values = np.array([(pose.position.x, pose.position.y) for pose in poses])
    wz_values = np.array([(pose.orientation.w, pose.orientation.z) for pose in poses])
    
    mean_position = np.mean(xy_values, axis=0)
    mean_orientation = np.mean(wz_values, axis=0)
    
    estimated_pose = Pose()
    
    estimated_pose.position.x = mean_position[0]
    estimated_pose.position.y = mean_position[1]
    estimated_pose.orientation.w = mean_orientation[0]
    estimated_pose.orientation.z = mean_orientation[1]
    
    return estimated_pose

def mean_poses_removed_outliers(poses):
    xy_values = np.array([(pose.position.x, pose.position.y) for pose in poses])
    wz_values = np.array([(pose.orientation.w, pose.orientation.z) for pose in poses])
    
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

def dbscan(poses, epslion, min_poses):
    clusters = []
    visited = set()

    for pose in poses:
        if f"{pose.position.x},{pose.position.y},{pose.orientation.w},{pose.orientation.z}" in visited:
            continue
        visited.add(f"{pose.position.x},{pose.position.y},{pose.orientation.w},{pose.orientation.z}")
        neighbours = find_neighbors(poses, pose, epslion)

        if len(neighbours) < min_poses:
            continue
        else:
            cluster = expand_cluster(pose, neighbours, epslion, min_poses, visited)
            clusters.append(cluster)
    return average_pose_of_largest_cluster(clusters)

def average_pose_of_largest_cluster(clusters):
    if clusters == []:
        return Pose()
    else:
        largest_cluster = max(clusters, key=lambda x: len(x))
        return mean_pose(largest_cluster)

def find_neighbors(poses, pose, epslion):
    neighbours = []
    for other_pose in poses:
        if other_pose == pose:
            continue
        if distance(pose, other_pose) < epslion:
            neighbours.append(other_pose)
    
    return neighbours

def expand_cluster(pose, neighbours, epslion, min_poses, visited):
    cluster = [pose]
    for neighbour in neighbours:
        if f"{neighbour.position.x},{neighbour.position.y},{neighbour.orientation.w},{neighbour.orientation.z}" not in visited:
            visited.add(f"{neighbour.position.x},{neighbour.position.y},{neighbour.orientation.w},{neighbour.orientation.z}")
            new_neighbours = find_neighbors(neighbours, neighbour, epslion)
            if len(new_neighbours) >= min_poses:
                neighbours.extend(new_neighbours)
        if neighbour not in cluster:
            cluster.append(neighbour)
    
    return cluster

def distance(pose1, pose2):
    return np.sqrt((pose1.position.x - pose2.position.x)**2 + (pose1.position.y - pose2.position.y)**2)