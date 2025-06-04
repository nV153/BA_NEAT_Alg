import logging
import os
import time
import gymnasium as gym
import neat
import matplotlib.pyplot as plt
import pandas as pd
from neat import parallel


def plot_mean_fitness(all_means: dict[str, list[float]]) -> None:
    """
    Plots mean fitness values over generations for various configurations.

    Args:
        all_means (dict[str, list[float]]): Keys are configuration names, 
                                            values are lists of mean fitness per generation.
    """
    # Temporarily suppress console output during plotting
    original_logging_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.WARNING)

    plt.figure(figsize=(10, 6))

    # Plot mean fitness for each configuration
    for name, mean_values in all_means.items():
        plt.plot(mean_values, label=name)

    plt.xlabel('Generation')
    plt.ylabel('Mean Fitness')
    plt.title('Mean Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Restore original logging level
    logging.getLogger().setLevel(original_logging_level)


def save_stats_to_excel(mean: list[float], stdex: list[float], minF: list[float], maxF: list[float], filename: str = "training_stats.xlsx") -> None:
    """
    Saves statistical data (mean, standard deviation, min and max fitness) for each generation to an Excel file.

    Args:
        mean (list[float]): A list of mean fitness values for each generation.
        stdex (list[float]): A list of standard deviations of fitness for each generation.
        minF (list[float]): A list of minimum fitness values for each generation.
        maxF (list[float]): A list of maximum fitness values for each generation.
        filename (str): The name of the Excel file to save the data to (default is "training_stats.xlsx").
    
    Returns:
        None
    """
    # Convert data into a DataFrame
    data = {
        "Generation": range(1, len(mean) + 1),
        "Mean Fitness": mean,
        "Standard Deviation": stdex,
        "Min Fitness": minF,
        "Max Fitness": maxF
    }
    
    df = pd.DataFrame(data)
    
    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False)
    #print(f"Statistiken wurden in {filename} gespeichert.")


def save_times_to_file(times: list[tuple[str, float]], file_name: str) -> None:
    """
    Saves the times of different configurations to a file.

    Args:
        times (list[tuple[str, float]]): A list of tuples where each tuple contains the configuration name (str)
                                          and the elapsed time (float).
        file_name (str): The name of the file to which the times will be saved.
    
    Returns:
        None
    """
    # Open the file in append mode ('a') to avoid overwriting existing data
    with open(file_name, 'a') as file:
        # Write each configuration name and its corresponding time to the file
        for config_name, time in times:
            file.write(f"{config_name}: {time}\n")


class NoCrossoverReproduction(neat.DefaultReproduction):
    """
    Custom reproduction class that disables crossover during reproduction.

    Inherits from neat.DefaultReproduction and overrides the 'reproduce' method.
    """
    def reproduce(self, config, species, pop_size, generation):
        return super().reproduce(config, species, pop_size, generation)

    

def play_game(net: neat.nn.FeedForwardNetwork, mode: str = "train") -> float:
    """
    Simulates a game using the provided neural network and environment, returning the total fitness.

    Args:
        net (neat.nn.FeedForwardNetwork): The neural network to control the agent in the game.
        mode (str, optional): The mode in which to run the game. 
                              'play' will render the environment, 'train' will run the game without rendering.
                              Defaults to "train".
    
    Returns:
        float: The fitness of the agent, calculated as the total reward accumulated over episodes.
    """
    if mode == "play":
        env = gym.make("Swimmer-v4", render_mode="human") 
        num_episodes = 5  
    else:
        env = gym.make("Swimmer-v4")  
        num_episodes = 1  

    total_reward = 0.0  # Initialize total reward

    # Loop through the episodes
    for _ in range(num_episodes):
        state, _ = env.reset(seed=42)  
        done = False 

        start_time = time.time()  
        max_duration = 5 

        episode_reward = 0.0 

        # Loop until the episode is done
        while not done:
            current_time = time.time() 
            elapsed_time = current_time - start_time  

            if elapsed_time > max_duration:
                print(f"Abbruch der Schleife nach {max_duration} Sekunden. Episode Reward: {episode_reward}")
                break

            # Get the neural network's output for the current state
            output = net.activate(state)

            # Take a step in the environment with the neural network's output as action
            next_state, reward, truncated, terminated, _ = env.step(output)

            # If the episode is terminated or truncated, end the loop
            if terminated or truncated:
                done = True

            # Accumulate the reward for this episode
            episode_reward += reward
            state = next_state  # Update the state

        if mode == "play":
            print(f"Episonden Reward:{episode_reward}")

        # Add the episode's reward to the total reward
        total_reward += episode_reward

    env.close()

    fitness = total_reward / num_episodes
    return fitness


def eval_genome(genome: neat.DefaultGenome, config: neat.Config) -> float:
    """
    Evaluates the fitness of a genome by running the associated neural network in the game environment.

    Args:
        genome (neat.DefaultGenome): The genome (individual) to evaluate, containing the neural network.
        config (neat.Config): The configuration object containing the settings for the NEAT algorithm.

    Returns:
        float: The fitness of the genome, based on the performance in the game.
    """

    genome.fitness = 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = play_game(net, mode="train")
    return genome.fitness


def run_test(name: str, config_path: str, mode: str = "train") -> tuple[list[float], list[float], float]:
    """
    Runs a test on the NEAT algorithm using a specified configuration file and mode. It trains a population of 
    genomes and returns the mean fitness, max fitness per generation, and the total elapsed time.

    Args:
        name (str): The name of the test, used for identifying the configuration and the output files.
        config_path (str): The file path to the NEAT configuration file.
        mode (str): The mode in which to run the test. 'train' for training or 'play' for evaluation. Default is "train".

    Returns:
        tuple[list[float], list[float], float]: A tuple containing the following:
            - list of mean fitness values per generation
            - list of max fitness values per generation
            - total elapsed time for the test in seconds
    """
    print("Starte Test:" + name)
    
    # Configure NEAT based on the test name
    if name == "confNonMating":
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        # Set a custom reproduction type (no crossover) for the specific test
        config.reproduction_type = NoCrossoverReproduction
    else:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    # Initialize the NEAT population using the configuration
    p = neat.Population(config)
    
    # Add a reporter to show output 
    if mode == "play":
        p.add_reporter(neat.StdOutReporter(True))

    # Set up the statistics reporter to gather fitness data
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    st = time.time()

    workers = 12
    pe = neat.ParallelEvaluator(workers, eval_genome)
    
    winner = p.run(pe.evaluate, 250)

    et = time.time()
    elapsed_time = et - st

    # Gather fitness statistics from the NEAT population
    mean = stats.get_fitness_mean()
    stdex = stats.get_fitness_stdev()
    minF = stats.get_fitness_stat(min)
    maxF = stats.get_fitness_stat(max)

    # Print the results of the test
    gen = len(mean)
    print(f"Der Test hat {gen} Generation und {elapsed_time} Sekunden gedauert")

    # If in "play" mode, show the best genome and run a simulation with it
    if mode == "play":
        print('\nBest genome:\n{!s}'.format(winner))
        play_game(net=winner, mode="play")
    else:
        filename = os.path.join(local_dir, f"Excel Results/Exp1_{name}_stats.xlsx")
        save_stats_to_excel(mean, stdex, minF, maxF, filename)

    return mean, maxF, elapsed_time

    

def run(mode: str = "train") -> None:
    """
    Runs the NEAT algorithm with multiple configurations and gathers statistics, 
    including mean and max fitness values per generation. Optionally, it can also 
    run in "play" mode to evaluate pre-trained models.

    Args:
        mode (str): The mode in which to run the test. Default is "train". 
                    In "train" mode, the algorithm will train and gather statistics, 
                    while in "play" mode, it will simply run pre-trained models for evaluation.
        
    Returns:
        None
    """
    # Initialize dictionaries to store the results of each run
    all_means = {}
    all_maxF = {}
    times = []

    # Select configurations and number of runs based on mode
    if mode == "play":
        configs = [
            os.path.join(local_dir, "confNormal.cfg"),
        ]
        num_runs = 1
    else:
        configs = [
            os.path.join(local_dir, "confNormal.cfg"),
            os.path.join(local_dir, "confInitRandom.cfg"),
            os.path.join(local_dir, "confNonMating.cfg"),
            os.path.join(local_dir, "confNoGrowth.cfg"),
            os.path.join(local_dir, "confNoSpecie.cfg")
        ]
        num_runs = 5

    # Iterate through each configuration
    for config_path in configs:
        # Extract the name of the configuration for labeling purposes
        config_name = config_path.split('\\')[-1].split('.')[0].replace('conf', '')
        
        # Run each configuration a specified number of times
        for i in range(num_runs):
            run_name = f"{config_name}_{i+1}"  # Append run number to the configuration name
            
            # Run the test for the current configuration and mode
            if config_name == "NonMating":  # Special handling for NonMating configuration, NonMating requires settings outside the configuration
                mean, maxF, time = run_test(run_name, os.path.join(local_dir, "confNonMating.cfg"), mode)
            else:
                mean, maxF, time = run_test(run_name, config_path, mode)
            
            # Store the results in the dictionaries
            if run_name not in all_means:
                all_means[run_name] = []
                all_maxF[run_name] = []
            
            all_means[run_name] = mean
            all_maxF[run_name] = maxF
            times.append((run_name, time))

    # Show Trainingdata
    if mode != "play":
        plot_mean_fitness(all_means)
        plot_mean_fitness(all_maxF)
        save_times_to_file(times, os.path.join(local_dir, "Excel Results/Exp1 times.txt"))

if __name__ == '__main__':   
    answer = input("Warning, if old Excel files exist, they will be overwritten. Do you want to continue? (yes/no): ").strip().lower()  
    local_dir = os.path.dirname(__file__)
    if answer == "no":
        print("Program will be terminated.")
        quit()
    else:
        run(mode="play")

