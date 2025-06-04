import os
import pickle
import time
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


class CustomCallback(BaseCallback):
    """
    Custom callback for monitoring the agent's performance during training.

    The callback checks the agent's performance every `check_freq` timesteps and prints the average score of 5 game runs.

    Parameters:
        check_freq (int): Frequency (in timesteps) at which the performance is evaluated and printed. Default is 5000.
        verbose (int): Verbosity level. Default is 0 (no output).

    Returns:
        None
    """
    
    def __init__(self, check_freq: int = 5000, verbose: int = 0):
        """
        Initializes the callback with the specified parameters.
        
        Parameters:
            check_freq (int): Number of timesteps between each performance evaluation. Default is 5000.
            verbose (int): Verbosity level. Default is 0 (no output).
        
        Returns:
            None
        """
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq  # Set the frequency for checking performance

    def _on_step(self) -> bool:
        """
        Called after each training step. Evaluates the agent's performance if the timestep is a multiple of `check_freq`.
        
        Parameters:
            None
        
        Returns:
            bool: True to continue training, False to stop.
        """
        # Check if the current number of timesteps is a multiple of check_freq
        if self.num_timesteps % self.check_freq == 0:
            score = 0.0  # Initialize score variable
            for _ in range(5):
                score += play_game(self.model)  # Play the game 5 times and accumulate the score
            score = score / 5  # Calculate the average score over the 5 games
            print(f"Score at timestep {self.num_timesteps}: {score}")  # Print the score at the current timestep
            scores.append(score)  # Append the score to the list of scores
        
        return True  # Continue the training process


def train(timesteps: int) -> None:
    """
    Train a PPO model on the specified environment, using the provided number of timesteps.
    
    This function loads or creates a model and score statistics, performs training, and saves the model and score statistics.

    Parameters:
        timesteps (int): The total number of timesteps for training the model.

    Returns:
        None
    """
    global scores

    # Load or initialize score statistics
    if os.path.exists(f"{score_filename}.zip"):
        with open(score_filename, 'rb') as f:
            scores = pickle.load(f)
    else:
        scores = []  

    LOAD = False  

    # Load or create the PPO model
    if os.path.exists(f"{model_filename}.zip"):
        model = PPO.load(model_filename, env=env)  
        print("Loaded existing model.")
        LOAD = True
    else:
        model = PPO(
            policy="MlpPolicy",                 # Policy model to be used
            env=env,                            # The environment the model will interact with
            learning_rate=0.001,                # Learning rate for training
            n_steps=2048,                       # Number of steps per update
            batch_size=64,                      # Mini-batch size for training
            gamma=0.99,                         # Discount factor for reward calculation
            gae_lambda=0.95,                    # GAE Lambda for advantage estimation
            ent_coef=0.0,                       # Entropy coefficient (for exploration)
            vf_coef=0.5,                        # Coefficient for the value function
            max_grad_norm=0.5,                  # Maximum gradient norm for stability
            tensorboard_log=os.path.join(local_dir, "tb_logs"),  # Directory for tensorboard logs
            verbose=0,                          # Verbosity level (0 = no output, 1 = output stats)
            seed=153,                           # Random seed for reproducibility
            device='auto',                      # Automatically select device (CPU/GPU)
            _init_setup_model=True              # Initialize the model at creation
        )
        print("Created new model.")

    start_time = time.time()
    print("Starting training:")

    # Initialize custom callback for periodic performance evaluation
    callback = CustomCallback(check_freq=100000)

    # Train the model with the specified timesteps and callback
    model.learn(
        total_timesteps=timesteps,
        log_interval=10,                     # Interval for logging during training
        reset_num_timesteps=False,           # Do not reset the timestep counter
        progress_bar=True,                   # Show a progress bar during training
        callback=callback,                   # Pass the custom callback
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time:.2f} seconds")

    # Save the trained model to file
    model.save(model_filename)
    print("Saved new model.")

    # Save the updated score statistics to file
    with open(score_filename, 'wb') as f:
        pickle.dump(scores, f)

    # Save training information (including load status and time) to a log file
    with open(os.path.join(local_dir, "PPO_Training_Infos.txt"), "a") as f:
        f.write(f"Model loaded: {LOAD}, Training time: {elapsed_time:.2f} seconds\n")



def play_game(model: PPO, human: bool = False) -> float:
    """
    Play a game using the provided PPO model on the BipedalWalker-v3 environment.

    Parameters:
        model (PPO): Trained PPO model to predict actions.
        human (bool): If True, renders the environment for human interaction.

    Returns:
        float: Total score (fitness) achieved during the game.
    """
    test_env = gym.make("BipedalWalker-v3", render_mode="human" if human else None)
    score = 0.0
    obs, _ = test_env.reset()
    done = truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs)  # Get action from the model
        obs, reward, done, truncated, _ = test_env.step(action)
        score += reward  # Update total score

    return score


def play(num_episodes: int = 3) -> None:
    """
    Play multiple episodes using the trained PPO model in human mode (rendered for a human player).
    
    Parameters:
        num_episodes (int, optional): The number of episodes to play. Default is 3.
    
    Returns:
        None: This function prints the scores for each episode but does not return any value.
    """
    model = PPO.load(model_filename, env=env)

    for i in range(num_episodes):
        score = play_game(model, human=True)  
        print(f"Run {i} erziele einen Score von {score}") 



def showStrucure() -> None:
    """
    Display the architecture of the trained PPO model's policy network.
    
    Parameters:
        None
        
    Returns:
        None: This function prints the structure of the model's policy network but does not return any value.
    """
    # Load the PPO model
    model = PPO.load(model_filename, env=env)

    # Access and print the model's policy network architecture
    print("Policy Network Architecture:")
    print(model.policy)



def paintStats() -> None:
    """
    Plot the training progress (fitness over timesteps) based on stored scores.
    """
    with open(score_filename, 'rb') as f:
        scores = pickle.load(f)

    x = np.arange(len(scores)) * 25000  # Timesteps (scaled by 25,000)
    y = scores

    plt.plot(x, y, marker='o', label='Data Points')
    plt.xlabel("Timestep")
    plt.ylabel('Fitness')
    plt.title('PPO Stats')
    plt.legend()
    plt.show()




def main() -> None:
    """
    Provides a user interface for training, playing, visualizing stats, or displaying model information.
    """
    mode = input("Would you like to train (t), play (s), view training stats (paint), or see network info (info)? ").strip().lower()
    
    if mode == 't':
        try:
            step = int(input("How many timesteps would you like to train? "))   # (100,000 ≈ 4 minutes)
            print(f"Training for {step} timesteps (~{step / 100000 * 4:.2f} minutes)")
            train(timesteps=step)
        except ValueError:
            print("Please enter a valid number.")
    
    elif mode == 's':
        try:
            num_episodes = int(input("How many episodes would you like to play? "))
            play(num_episodes)
        except ValueError:
            print("Please enter a valid number.")
            
    elif mode == "paint":
        paintStats()  # Display the training stats plot
        
    elif mode == "info":
        showStrucure()  # Show model structure
        
    else:
        print("Invalid input. Please choose 't', 's', 'paint', or 'info'.")



def make_env() -> gym.Env:
    """
    Creates and returns a new instance of the "BipedalWalker-v3" environment.

    Parameters:
        None

    Returns:
        gym.Env: A new BipedalWalker-v3 environment instance with the render mode set to "rgb_array".
    """
    return gym.make("BipedalWalker-v3", render_mode="rgb_array")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    model_filename = os.path.join(local_dir, "ppo_bipedalWalker")
    score_filename = os.path.join(local_dir, "PPO_stats")
    # Create a vectorized environment with multiprocessing
    num_envs = 4    # For parallel execution with SubprocVecEnv
    env = SubprocVecEnv([make_env for _ in range(num_envs)]) 
    main()

