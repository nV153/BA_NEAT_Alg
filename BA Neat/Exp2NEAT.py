import os
import pickle
import shutil
import time
import pandas as pd  
from typing import List

import os
import time
import gymnasium as gym
import neat
import pandas as pd
from neat import parallel
import GeneticAlgUtil

import time
from multiprocessing import Process, Manager


local_dir = os.path.dirname(__file__)

def play_game(net: neat.nn.FeedForwardNetwork, human: bool = False) -> float:
    """
    Simulates a game (BipedalWalker) using a NEAT network.
    
    Parameters:
        net (neat.nn.FeedForwardNetwork): The trained neural network controlling the agent.
        human (bool): If True, enables human rendering for manual control.
        
    Returns:
        float: Total fitness (reward) accumulated during the episode.
    """
    fitness: float = 0.0
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human" if human else None)
    state, _ = env.reset()
    done = truncated = False

    while not (done or truncated):
        action = net.activate(state)  # Compute action from the network
        state, reward, done, truncated, _ = env.step(action)
        fitness += reward  # Accumulate reward as fitness

    env.close()
    return fitness



def eval_genome(genome: neat.DefaultGenome, config: neat.Config) -> float:
    """
    Evaluates a genome by using its corresponding neural network to play the game and calculate fitness.
    
    Parameters:
        genome (neat.DefaultGenome): The genome representing an individual in the population.
        config (neat.Config): The configuration object containing the parameters for the NEAT algorithm.
        
    Returns:
        float: The fitness of the genome, which is determined by the total reward accumulated during the game.
    """
    
    genome.fitness = 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = play_game(net, False)
    
    # Return the fitness score of the genome
    return genome.fitness


def play(run_num: int) -> None:
    """
    Plays a game using a pre-trained neural network (from a given run number) and evaluates its performance
    over a specified number of episodes.
    
    Parameters:
        run_num (int): The run number that corresponds to the saved neural network.
        
    Returns:
        None: The function doesn't return any value but prints the performance (fitness) of the network for each episode.
    """

    save_location = os.path.join(local_dir, "HallOfFame", f"NeatRun{run_num}")
    
    with open(save_location, 'rb') as f:
        net = pickle.load(f)

    while True:
        try:
            num_episodes = int(input("Please enter the number of episodes: "))
            if num_episodes > 0:
                break  
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    for episode in range(1, num_episodes + 1):
        score: float = play_game(net, human=True)  
        print(f"Episode {episode} for RunV1 {run_num}: Achieved Fitness: {score}")



    
def get_latest_checkpoint(checkpoint_dir: str, config) -> neat.Population:
    """
    Retrieves the latest checkpoint from the given checkpoint directory and restores the population from it.
    If no checkpoint is found, it creates a new population using the provided configuration.
    
    Parameters:
        checkpoint_dir (str): The directory where checkpoint files are stored.
        config: The configuration object needed to initialize a new NEAT population.
        
    Returns:
        neat.Population: A NEAT population object, either restored from a checkpoint or newly created.
    """
    
    # Check if the checkpoint directory exists, and create it if it doesn't
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Define the checkpoint file prefix for easy identification
    checkpoint_prefix = os.path.join(checkpoint_dir, 'neat-checkpoint-')
    
    # List all checkpoint files in the directory that match the prefix
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('neat-checkpoint-')]

    # If checkpoint files are found, restore the latest one
    if checkpoints:
        # Find the latest checkpoint by sorting and getting the one with the highest number
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Restoring from checkpoint: {checkpoint_path}")
        return neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        # If no checkpoint is found, create and return a new population
        return neat.Population(config)

     

def train(num_generations: int, run_num: int):
    """
    Trains a NEAT algorithm for the specified number of generations, 
    restoring from a checkpoint if available, or starting a new population.
    
    Parameters:
        num_generations (int): The number of generations to train the NEAT algorithm.
        run_num (int): The run number, used for checkpoint directories and file saving.
        
    Returns:
        None
    """
    
    # Define the path to the configuration file and checkpoint directory
    config_path = os.path.join(local_dir, 'confWalker.cfg')
    checkpoint_dir = os.path.join(local_dir, "NEAT CheckPoints", f'NEATPointsMultiwalker_Run{run_num}')
    
    # Create the NEAT configuration object
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Search for existing checkpoint files
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'neat-checkpoint-')
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('neat-checkpoint-')]

    # If checkpoints exist, restore from the latest one; otherwise, create a new population
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Restoring from checkpoint: {checkpoint_path}")
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        p = neat.Population(config)

    # Add reporters to track progress
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoint_prefix))

    # Start timer
    st = time.time()

    # Run the NEAT algorithm in parallel with 12 workers
    workers = 12
    pe = neat.ParallelEvaluator(workers, eval_genome)
    winner = p.run(pe.evaluate, num_generations)

    # Stop timer and calculate elapsed time
    et = time.time()
    elapsed_time = et - st
    print(f"Training took {elapsed_time:.2f} seconds.")

    # Gather fitness statistics
    mean = stats.get_fitness_mean()
    stdex = stats.get_fitness_stdev()
    minF = stats.get_fitness_stat(min)
    maxF = stats.get_fitness_stat(max)

    # Record number of generations and print info
    gen = len(mean)  # Number of generations completed
    print(f"Training ran for {gen} generations.")
    GeneticAlgUtil.log_training_time(run_num, num_generations, elapsed_time, 0)

    # Prepare data for saving in an Excel file
    filename = f"d:/Python Workspace/BA Neat/Excel Results/Exp2_NeatRun{run_num}_stats.xlsx"
    data = {
        'Generation': list(range(gen)),
        'Mean Fitness': mean,
        'Standard Deviation': stdex,
        'Min Fitness': minF,
        'Max Fitness': maxF,
    }

    # Create DataFrame and save the data to an Excel file
    df = pd.DataFrame(data)
    try:
        df.to_excel(filename, index=False)
        print(f"Statistics saved to {filename}.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    # Save the winning neural network to a file
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    save_location = os.path.join(local_dir, "HallOfFame", f"NeatRun{run_num}")

    with open(save_location, 'wb') as f:
        pickle.dump(winner_net, f)




def get_used_NEATrun_numbers() -> list[int]:
    """
    Retrieves a list of all run numbers that have existing checkpoints in the checkpoint directory.
    
    This function scans the "NEAT CheckPoints" directory for subdirectories that start with
    "NEATPointsMultiwalker_Run", extracts the run number from the subdirectory name, and 
    returns a list of those run numbers.
    
    Parameters:
        None
        
    Returns:
        list[int]: A list of used run numbers extracted from the directory names.
    """
    
    checkpoint_base_dir = os.path.join(local_dir, "NEAT CheckPoints")
    
    if not os.path.exists(checkpoint_base_dir):
        return []

    existing_run_nums = []

    # Iterate through all subdirectories in the checkpoint base directory
    for entry in os.listdir(checkpoint_base_dir):
        if entry.startswith('NEATPointsMultiwalker_Run'):
            try:
                # Extract the run number from the directory name
                run_num = int(entry.split('_')[-1].replace('Run', ''))
                existing_run_nums.append(run_num)
            except ValueError:
                # Skip if the run number cannot be converted to an integer
                continue

    # Return the list of existing run numbers
    return existing_run_nums


def delete_checkpoints(run_num: int) -> None:
    """
    Deletes the checkpoint directory for a given run number.
    
    This function checks if a directory corresponding to the provided run number exists in the 
    "NEAT CheckPoints" directory. If it does, the directory and its contents are deleted.
    If the directory does not exist, it prints an appropriate message.
    
    Parameters:
        run_num (int): The run number for which the checkpoint directory should be deleted.
        
    Returns:
        None
    """
    
    # Define the checkpoint directory for the given run number
    checkpoint_dir = os.path.join(local_dir, "NEAT CheckPoints", f'NEATPointsMultiwalker_Run{run_num}')

    # Check if the directory exists before deleting
    if os.path.exists(checkpoint_dir):
        # Remove the directory and its contents
        shutil.rmtree(checkpoint_dir)
        print(f"Checkpoint directory for Run {run_num} has been deleted.")
    else:
        print(f"Checkpoint directory for Run {run_num} does not exist.")


def main_menu() -> None:
    """
    Main menu function to allow the user to choose between training, playing, deleting runs, or both.
    
    The user is prompted to select an option, and the corresponding action is taken:
    - Training: Begins the training of a new population or continues an existing one.
    - Playing: Lets the user play with the trained model.
    - Both: Allows both training and playing.
    - Deleting: Deletes a selected run and its associated checkpoints.
    
    Parameters:
        None
        
    Returns:
        None
    """
    
    # Get the list of used run numbers from the checkpoint directory
    used_run_numbers = get_used_NEATrun_numbers()
    
    # Display the list of existing run numbers
    print("\n=== Existing Run Numbers ===")
    if used_run_numbers:
        print(", ".join(map(str, sorted(used_run_numbers))))  # Show sorted run numbers
    else:
        print("No existing runs found.")
    print("============================\n")
    
    while True:
        choice = input("What would you like to do? (t = Train, p = Play, b = Both, d = Delete): ").strip().lower()
        
        # Check if the input is valid
        if choice in ['t', 'p', 'b', 'd']:
            
            if choice == 'd':
                # Handle deletion of a run
                if used_run_numbers:
                    run_num = int(input("Please enter the run number to delete: "))
                    if run_num in used_run_numbers:
                        delete_checkpoints(run_num)  # Delete the checkpoints for the given run
                        used_run_numbers.remove(run_num)  # Remove the run number from the list
                        print(f"Run {run_num} successfully deleted.")
                    else:
                        print(f"Run {run_num} does not exist.")
                else:
                    print("No existing runs to delete.")
                continue  # Go back to the next iteration of the loop
            
            # Ask the user for the run number to train or play with
            run_num = GeneticAlgUtil.get_run_number()
            
            if choice in ['t', 'b']:
                if run_num in used_run_numbers:
                    overwrite = input(f"Run {run_num} already exists. Do you want to continue and overwrite the existing population (not recommended due to statistical errors)? (y/n): ").strip().lower()
                    if overwrite != 'y':
                        print("Training canceled.")
                        continue
                try:
                    num_generations = int(input("How many generations should be trained? "))
                    train(num_generations, run_num)  # Start training with the specified number of generations
                except ValueError:
                    print("Invalid number of generations. Please enter a valid integer.")
                    continue
                
            if choice in ['p', 'b']:
                if run_num not in used_run_numbers:
                    print(f"Run {run_num} does not exist. Please train with this run number first.")
                    continue
                play(run_num) 
            
            break  
        else:
            print("Invalid input. Please enter 't' for Train, 'p' for Play, 'b' for Both, or 'd' for Delete.")




if __name__ == "__main__":
    main_menu()
