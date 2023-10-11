import numpy as np 
import math
#import scipy.integrate as spi
import copy
import rospy
from geometry_msgs.msg import Pose, Quaternion

#me
def multinomial_resampling(particles, weights, num_of_samples):
    num_particles = len(particles)
    sampled_particles = []

    sum_of_weights = 0
    for i in range(0, len(weights)):
        sum_of_weights += weights[i]
    
    normaliser = 1 / sum_of_weights

    normalised_weights = []
    
    for i in range(0, len(weights)):
        normalised_weights.append(weights[i] * normaliser)

    for i in range(0, num_of_samples):
        sample = np.random.choice(particles, p=normalised_weights)
        sampled_particles.append(sample)
    return sampled_particles

#me
def residual_resampling(particles, weights, num_of_samples):
    num_particles = len(particles)
    sampled_particles = []

    # Compute the cumulative distribution function (CDF)
    sum_of_weights = sum(weights)
    normalizer = 1 / sum_of_weights
    cdf = [weights[0] * normalizer]

    for i in range(1, num_particles):
        cdf.append(cdf[i-1] + weights[i] * normalizer)
    
    # Residual resampling
    r = np.random.uniform(0, 1) / num_particles

    for i in range(num_of_samples):
        target_weight = r + i / num_particles
        for j in range(num_particles):
            if target_weight <= cdf[j]:
                sampled_particles.append(particles[j])
                break

    return sampled_particles

#me
def systematic_resampling(particles, weights, num_of_samples):
    num_particles = len(particles)
    sampled_particles = []

    sum_of_weights = sum(weights)
    normaliser = 1 / sum_of_weights
    cdf = [weights[0]* normaliser]
        
    for i in range(1, num_particles):
        cdf.append(cdf[i-1] + weights[i] * normaliser)

    # Select starting threshold
    threshold = np.random.uniform(0, 1/len(weights))
    i = 0
        
    # Resample select portion of posessudo apt-get install python3-scipyS
    for j in range(0, num_of_samples):
        while threshold > cdf[i]:
            i += 1
        sampled_particles.append(particles[i])  # Corrected the line
        threshold += 1 / num_particles

    return sampled_particles

#me
def stratified_resampling(particles, weights, num_of_samples):
    num_particles = len(particles)
    sampled_particles = []

    sum_of_weights = sum(weights)
    normalizer = 1 / sum_of_weights
    cdf = [weights[0] * normalizer]
    
    for i in range(1, num_particles):
        cdf.append(cdf[i-1] + weights[i] * normalizer)
    
    # Stratified resampling
    u = [np.random.uniform(0, 1) / num_particles + (i / num_particles) for i in range(num_of_samples)]
    
    j = 0
    
    for i in range(num_of_samples):
        while j < num_particles and cdf[j] < u[i]:
            j += 1
        if j < num_particles:
            sampled_particles.append(particles[j])
        else:
            # If j exceeds the number of particles, wrap it around to 0
            j = 0
            sampled_particles.append(particles[j])

    return sampled_particles

#me
def adaptive_resampling(particles, weights, num_of_samples, weight_threshold, particle_threshold):
    num_particles = len(particles)
    low_weight_particles = []

    sum_of_weights = 0
    for i in range(0, len(weights)):
        sum_of_weights += weights[i]
    
    normaliser = 1 / sum_of_weights

    normalised_weights = []
    
    for i in range(0, len(weights)):
        normalised_weights.append(weights[i] * normaliser)

    highest_weight = 0
    lowest_weight = 1
    for i in range(0,num_particles):
        if normalised_weights[i] > highest_weight:
            highest_weight = normalised_weights[i]
        if normalised_weights[i] < lowest_weight:
            lowest_weight = normalised_weights[i]
        if normalised_weights[i]<weight_threshold:
            low_weight_particles.append(particles[i])
    rospy.loginfo("Highest weight: %f", highest_weight)
    rospy.loginfo("Lowest weight: %f", lowest_weight)
    rospy.loginfo("Number of particles: %d", num_particles)
    rospy.loginfo("Number of low weight particles: %d", len(low_weight_particles))
    if len(low_weight_particles) > particle_threshold:
        rospy.loginfo("Stratified Resampling")
        sampled_particles = stratified_resampling(low_weight_particles, weights, num_of_samples)
    
    else:
        rospy.loginfo("multinomial Resampling")
        sampled_particles = multinomial_resampling(particles, weights, num_of_samples)
    
    return sampled_particles

#me
def compute_weighted_average(particles, weights):
    num_particles = len(particles)
    weighted_sum_x = 0
    weighted_sum_y = 0
    weighted_sum_w = 0
    weighted_sum_z = 0


    for i in range(1, num_particles):
        weighted_sum_x += weights[i] * particles[i].position.x
        weighted_sum_y += weights[i] * particles[i].position.y
        weighted_sum_w += weights[i] * particles[i].orientation.w
        weighted_sum_z += weights[i] * particles[i].orientation.z
    
    weighted_avg_x = weighted_sum_x / num_particles
    weighted_avg_y = weighted_sum_y / num_particles
    weighted_avg_w = weighted_sum_w / num_particles
    weighted_avg_z = weighted_sum_z / num_particles

    return (weighted_avg_x, weighted_avg_y, weighted_avg_w, weighted_avg_z)

#me
def reguralized_resampling(particles, weights, num_of_samples, regularization_factor):
    num_particles = len(particles)
    sampled_particles = []
    weighted_average = compute_weighted_average(particles, weights)

    for i in range(0, num_of_samples):
        regularized_particle = Pose()
        regularized_particle.position.x = (1 - regularization_factor) * particles[i].position.x + regularization_factor * weighted_average[0]
        regularized_particle.position.y = (1 - regularization_factor) * particles[i].position.y + regularization_factor * weighted_average[1]
        regularized_particle.orientation.w = (1 - regularization_factor) * particles[i].orientation.w + regularization_factor * weighted_average[2]
        regularized_particle.orientation.z = (1 - regularization_factor) * particles[i].orientation.z + regularization_factor * weighted_average[3]
        sampled_particles.append(regularized_particle)

    return sampled_particles

def residual_stratified_resampling(particles, weights, sample_size):
    num_particles = len(particles)
    new_particles = [None] * num_particles

    # Calculate the number of whole samples and the remainder
    num_samples = int(sum(weights))
    remainder = sum(weights) - num_samples

    # Calculate the size of each stratum
    stratum_size = num_samples / num_particles

    # Initialize the cumulative weight and stratum counter
    cumulative_weight = 0.0
    stratum = 0

    # Ensure that the specified sample_size is within bounds
    sample_size = min(sample_size, num_samples)
    
    for i in range(num_particles):
        # Calculate the target number of samples for this stratum
        target_samples = int(stratum_size) + (1 if remainder > 0 else 0)
        remainder -= 1

        for k in range(target_samples):
            cumulative_weight += 1.0 / num_samples

            while cumulative_weight > weights[stratum]:
                cumulative_weight -= weights[stratum]
                new_particles[i] = copy.deepcopy(particles[stratum])
                stratum = (stratum + 1) % num_particles
                
                # Check if the desired sample size has been reached
                if new_particles.count(None) == num_particles - sample_size:
                    return [p for p in new_particles if p is not None]

    return [p for p in new_particles if p is not None]

#me with help learning copy import
def smoothed_resampling(particles, weights, num_of_samples):
    num_particles = num_of_samples
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Determine the number of particles to be sampled deterministically
    deterministic_count = int(num_particles * (1 - 1 / (2 * num_particles)))
    
    # Initialize resampled_particles as an empty list
    resampled_particles = []

    # Step 1: Resample deterministically
    for _ in range(deterministic_count):
        index = np.random.choice(num_particles)  # Randomly choose an index
        resampled_particles.append(copy.copy(particles[index]))
    
    remaining_weight = 1.0 - deterministic_count / num_particles

    # Step 2: Resample stochastically
    for _ in range(num_particles - deterministic_count):
        # Sample a random weight between 0 and remaining_weight
        target_weight = np.random.uniform(0, remaining_weight)  # Random number between 0 and remaining_weight

        # Initialize the cumulative weight to 0
        cumulative_weight = 0
        
        # Find the particle that corresponds to the sampled weight
        for j in range(num_particles):
            cumulative_weight += normalized_weights[j]
            if cumulative_weight >= target_weight:
                index = j
                break
        
        # Add a copy of the selected particle to resampled_particles
        resampled_particles.append(copy.copy(particles[index]))

    # Return the resampled set of particles
    return resampled_particles

#chatgpt
#I wanted to import this to compare and to read into it but none of this was written by me 
#Particle Marginal Metropolis-Hastings (PMMH)
def update_pose(pose, proposal):
    updated_pose = Pose()
    updated_pose.position.x = pose.position.x + proposal
    updated_pose.position.y = pose.position.y + proposal
    updated_pose.position.z = 0
    updated_quat = Quaternion(pose.orientation.w, 0, 0, pose.orientation.z)
    proposal_quat = Quaternion(0, 0, 0, proposal)
    updated_quat = proposal_quat * updated_quat
    updated_pose.orientation.w = updated_quat[0]
    updated_pose.orientation.z = updated_quat[3]
    return updated_pose

def pmmh_resampling(particles, weights, num_iterations=1000, proposal_std=0.1, sample_size=100):
    num_particles = len(particles)
    mcmc_samples = []

    while len(mcmc_samples) < sample_size:
        for _ in range(num_iterations):
            proposal = np.random.normal(0, proposal_std, num_particles)
            
            # Update the pose objects using the provided context and update_pose function
            proposed_particles = [update_pose(particles[i], proposal[i]) for i in range(num_particles)]

            diff_squared = ((proposed_particles - weights) ** 2).sum()
            proposal_squared = (proposal ** 2).sum()
            acceptance_prob = np.exp(-0.5 * diff_squared - 0.5 * proposal_squared)

            if np.random.uniform(0, 1) < acceptance_prob:
                # Update the particles based on the updated pose objects
                particles = proposed_particles

        mcmc_samples.append(particles.copy())

    return mcmc_samples


#This code takes pre-defined particles and weights as input and performs the resampling part of the PMMH algorithm using a Gaussian

#me
def auxiliary_particle_resampling(particles, weights):
    total_weight = sum(weights)
    normalized_weights = [weights / total_weight ]

        # Resample particles using the provided resampling method
    resampled_particles = residual_stratified_resampling(particles, normalized_weights)

    return resampled_particles


