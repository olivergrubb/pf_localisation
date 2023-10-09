import numpy as np 
import math
import scipy.integrate as spi
import copy

#me
def multinomial_resampling(particles, weights):
    num_particles = len(particles)
    sampled_particles = []

    for i in range(1, num_particles):
        sample = np.random.choice(particles, p=weights)
        np.append(sampled_particles, sample)
    return sampled_particles

#me
def residual_resampling(particles, weights):
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

    for i in range(num_particles):
        target_weight = r + i / num_particles
        for j in range(num_particles):
            if target_weight <= cdf[j]:
                sampled_particles.append(particles[j])
                break

    return sampled_particles

#me
def systematic_resampling(particles, weights):
    num_particles = len(particles)
    sampled_particles = []
    stochastic_ratio = 1

    sum_of_weights = sum(weights)
    normaliser = 1 / sum_of_weights
    cdf = [weights[0]* normaliser]
        
    for i in range(1, num_particles):
        cdf.append(cdf[i-1] + weights[i] * normaliser)

    # Select starting threshold
    threshold = np.random.uniform(0, 1/len(weights))
    i = 0
        
    # Resample select portion of posessudo apt-get install python3-scipyS
    for j in range(0, math.floor(num_particles * stochastic_ratio)):
        while threshold > cdf[i]:
            i += 1
        sampled_particles.append(particles[i])  # Corrected the line
        threshold += 1 / num_particles

    return sampled_particles

#me
def stratified_resampling(particles, weights):
    num_particles = len(particles)
    sampled_particles = []

    sum_of_weights = sum(weights)
    normalizer = 1 / sum_of_weights
    cdf = [weights[0] * normalizer]
    
    for i in range(1, num_particles):
        cdf.append(cdf[i-1] + weights[i] * normalizer)
    
    # Stratified resampling
    u = [np.random.uniform(0, 1) / num_particles + (i / num_particles) for i in range(num_particles)]
    
    for i in range(num_particles):
        for j in range(num_particles):
            if cdf[j] > u[i]:
                sampled_particles.append(particles[j])
                break

    return sampled_particles

#me
def adaptive_resampling(particles, weights, threshold):
    num_particles = len(particles)
    low_weight_particles = []

    for i in range(1,num_particles):
        if weights[i]<threshold:
            low_weight_particles.append(particles[i])
    
    if len(low_weight_particles) > threshold:
        sampled_particles = stratified_resampling(low_weight_particles, weights)
    
    else:
        sampled_particles= multinomial_resampling(particles, weights)
    
    return sampled_particles

#me
def compute_weighted_average(particles, weights):
    num_particles = len(particles)
    weighted_sum_x = 0
    weighted_sum_y = 0
    weighted_sum_theta = 0

    for i in range(1, num_particles):
        weighted_sum_x += weights[i] * particles[i].x
        weighted_sum_y += weights[i] * particles[i].y
        weighted_sum_theta += weights[i] * particles[i].theta
    
    weighted_avg_x = weighted_sum_x
    weighted_avg_y = weighted_sum_y
    weighted_avg_theta = weighted_sum_theta

    return (weighted_avg_x, weighted_avg_y, weighted_avg_theta)

#me
def reguralized_resampling(particles, weights, regularization_factor):
    num_particles = len(particles)
    sampled_particles = []

    weighted_average = compute_weighted_average(particles, weights)

    for i in range(1, num_particles):
        regularized_particle = (1 - regularization_factor) * particles[i] + regularization_factor * weighted_average
        sampled_particles.append(regularized_particle)

    return sampled_particles

#chatgpt
def compute_true_distribution_prob(true_distribution, bins):
    true_distribution_prob = {}

    for bin in bins:
        bin_start = bin.start
        bin_end = bin.end

        # Integrate the true distribution over the bin range (for 1D)
        true_prob_bin, _ = spi.quad(true_distribution, bin_start, bin_end)

        true_distribution_prob[bin] = true_prob_bin

    return true_distribution_prob

#chatgpt
def compute_particle_distribution_prob(particles, weights, bins):
    particle_distribution_prob = {}

    for bin in bins:
        bin_start = bin.start
        bin_end = bin.end
        particles_in_bin = []

        for i, particle in enumerate(particles):
            # Determine if the particle belongs to the current bin
            if bin_start <= particle.position <= bin_end:
                particles_in_bin.append(weights[i])

        # Calculate the particle distribution probability for the bin
        if particles_in_bin:
            particle_prob_bin = sum(particles_in_bin) / sum(weights)
        else:
            particle_prob_bin = 0.0

        particle_distribution_prob[bin] = particle_prob_bin

    return particle_distribution_prob

#chatgpt
def compute_kld_threshold(true_distribution, particles, weights, bins):
    num_particles = len(particles)

    true_distribution_prob = compute_true_distribution_prob(true_distribution, bins)
    particle_distribution_prob = compute_particle_distribution_prob(particles, weights, bins)

    kld = 0

    for bin in bins:
        kld += true_distribution_prob[bin] * np.log(true_distribution_prob[bin] / particle_distribution_prob[bin])

    return kld

#chatgpt
def KLD_sampling(particles, weights, true_distribution, max_particles):
    num_particles = len(particles)
    kld_threshold = compute_kld_threshold(true_distribution, particles, weights, bins)

    if num_particles > max_particles:
        low_weight_particles = []
        new_particles = []

        for i in range(num_particles):
            if weights[i] < kld_threshold:
                low_weight_particles.append(particles[i])

        if len(low_weight_particles) > max_particles:
            # Perform more sophisticated resampling (e.g., stratified or residual)
            new_particles = stratified_resampling(low_weight_particles, weights)
        else:
            new_particles = particles

    return new_particles

#chatgpt wrote this one
def residual_stratified_resampling(particles, weights):
    num_particles = len(weights)
    
    sum_of_weights = sum(weights)
    normalizer = 1 / sum_of_weights
    cdf = [weights[0] * normalizer]
    
    for i in range(1, num_particles):
        cdf.append(cdf[i-1] + weights[i] * normalizer)
    
    # Create the indices for the selected particles
    selected_indices = []
    
    # Step 1: Determine the number of copies for each particle
    num_copies = []
    for i in range(num_particles):
        num_copies[i] = int(num_particles * weights[i])
    
    # Step 2: Calculate the remaining fractional parts
    fractional_parts = num_particles * weights - num_copies
    
    # Step 3: Resample the whole number of copies
    j = 0
    for i in range(num_particles):
        while num_copies[i] > 0:
            selected_indices[j] = i
            num_copies[i] -= 1
            j += 1
    
    # Step 4: Resample the fractional parts
    remaining_fractional_parts = fractional_parts - np.floor(fractional_parts)
    cum_fractional_parts = np.cumsum(remaining_fractional_parts)
    u = np.random.rand() / num_particles  # Random number to select particles
    
    for i in range(num_particles):
        while u + i / num_particles > cum_fractional_parts[j]:
            j += 1
        selected_indices[j] = i
    
    # Return the selected indices
    return selected_indices
#me with help learning copy import
def smoothed_resampling(particles, weights):
    num_particles = len(particles)
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
def AdaptiveThresholdResampling(particles, resampling_threshold_fn):
    num_particles = len(particles)
    
    # Calculate the resampling threshold based on particle weights
    resampling_threshold = resampling_threshold_fn(particles)
    
    # Initialize resampled_particles as an empty list
    resampled_particles = []
    
    # Initialize variables for resampling
    current_weight = particles[0].weight
    j = 0
    
    for i in range(1, num_particles):
        # Calculate the number of times the current particle will be duplicated
        num_copies = math.floor(current_weight / resampling_threshold)
        
        # Add num_copies copies of the current particle to resampled_particles
        for k in range(1, num_copies + 1):
            resampled_particles.append(copy.deepcopy(particles[i - 1]))
        
        # Calculate the remaining weight for the current particle
        current_weight = current_weight - num_copies * resampling_threshold
        
        # Check if we should move to the next particle
        if current_weight < resampling_threshold:
            j = i
            current_weight += particles[j].weight
    
    # Return the resampled set of particles
    return resampled_particles

#chatgpt
#I wanted to import this to compare and to read into it but none of this was written by me 
#Particle Marginal Metropolis-Hastings (PMMH)
def pmmh_resampling(particles, weights, num_iterations=1000, proposal_std=0.1):
    # Get the number of particles
    num_particles = len(particles)

    # Initialize the MCMC samples list
    mcmc_samples = []

    for _ in range(num_iterations):
        # MCMC update (using a simple random walk proposal)
        proposal = np.random.normal(0, proposal_std, num_particles)
        proposed_particles = particles + proposal

        # Calculate acceptance probability
        acceptance_prob = np.exp(-0.5 * ((proposed_particles - weights) ** 2).sum() -
                                0.5 * (proposal ** 2).sum())

        # Accept or reject the proposed state with probability acceptance_prob
        if np.random.uniform(0, 1) < acceptance_prob:
            particles = proposed_particles

        mcmc_samples.append(particles)

    return mcmc_samples


#This code takes pre-defined particles and weights as input and performs the resampling part of the PMMH algorithm using a Gaussian

#me
def auxiliary_particle_resampling(particles, weights):
    total_weight = sum(weights)
    normalized_weights = [weights / total_weight ]

        # Resample particles using the provided resampling method
    resampled_particles = residual_stratified_resampling(particles, normalized_weights)

    return resampled_particles


