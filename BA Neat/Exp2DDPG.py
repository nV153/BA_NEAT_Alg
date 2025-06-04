import os
import pickle
import time
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


class CustomCallback(BaseCallback):
    """
    A custom callback class for monitoring training performance during each step.

    Parameters:
        check_freq (int): Frequency (in timesteps) to check and log the score. Default is 5000.
        verbose (int): Verbosity level for logging. Default is 0 (no logs).
    """

    def __init__(self, check_freq: int = 5000, verbose=0):
        super(CustomCallback, self).__init__(verbose)  # Initialize the base class
        self.check_freq = check_freq  # Set the frequency of score checks

    def _on_step(self) -> bool:
        """
        Callback method called at each training step.

        Checks if the number of timesteps is a multiple of `check_freq` and computes 
        the average score of the model over 5 game episodes, printing the result.
        
        Parameters:
            None
        
        Returns:
            bool: Always returns True to continue training.
        """
        if self.num_timesteps % self.check_freq == 0:
            score = 0.0
            for _ in range(5):  # Play 5 episodes to calculate the average score
                score += play_game(self.model)  
            score = score / 5  # Calculate average score
            print(f"Score at timestep {self.num_timesteps}: {score}")
            scores.append(score)  
        
        return True  # Continue training


def train(timesteps):
    """
    Train the DDPG model for a specified number of timesteps.

    Parameters:
        timesteps (int): Number of timesteps for the training process.

    This function:
    - Initializes action noise for exploration.
    - Loads or creates a score stats file.
    - Loads an existing model if it exists or creates a new one.
    - Trains the model using the DDPG algorithm.
    - Saves the trained model and score stats.
    """
    global scores  # Global variable to store scores during training

    # Define action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Load or initialize score stats
    if os.path.exists(f"{score_filename}.zip"):
        with open(score_filename, 'rb') as f:
            scores = pickle.load(f)
    else:
        scores = []  # Initialize an empty list for scores

    LOAD = False
    # Load an existing model if it exists
    if os.path.exists(f"{model_filename}.zip"):
        model = DDPG.load(model_filename, env=env)
        print("Loaded existing model.")
        LOAD = True
    else:
        # Create a new DDPG model if it doesn't exist
        model = DDPG(
            policy="MlpPolicy",  # Policy model
            env=env,  # Environment
            learning_rate=0.001,  # Learning rate
            buffer_size=1000000,  # Replay buffer size
            learning_starts=100,  # Steps before training begins
            batch_size=256,  # Mini-batch size
            tau=0.005,  # Soft update coefficient
            gamma=0.99,  # Discount factor
            train_freq=1,  # Frequency of model updates
            gradient_steps=1,  # Number of gradient steps per update
            action_noise=action_noise,  # Action noise for exploration
            replay_buffer_class=None,  # Use the default replay buffer
            replay_buffer_kwargs=None,  # No specific arguments
            optimize_memory_usage=False,  # Don't use memory-efficient version
            tensorboard_log="BA NEAT/tb_logs/",  # TensorBoard log directory
            policy_kwargs=None,  # No specific policy arguments
            verbose=0,  # No verbose output
            seed=153,  # Random seed
            device='auto',  # Automatically choose device (CPU or GPU)
            _init_setup_model=True  # Initialize model when created
        )
        print("Created new model.")

    # Start training timer
    start_time = time.time()
    print("Training started:")

    # Initialize custom callback
    callback = CustomCallback(check_freq=100000)

    # Train the model with the specified timesteps
    model.learn(
        total_timesteps=timesteps,
        log_interval=10,
        reset_num_timesteps=False,
        progress_bar=True,
        callback=callback,
    )

    # Calculate and display training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

    # Save the trained model
    model.save(model_filename)
    print("Saved new model.")

    # Save the score stats to a file
    with open(score_filename, 'wb') as f:
        pickle.dump(scores, f)

    # Log training time and model loading status
    with open("BA NEAT\\DDPG_Training_Infos.txt", "a") as f:
        f.write(f"Model loaded: {LOAD}, Training time: {elapsed_time:.2f} seconds\n")




def play_game(model, human=False):
    """
    Play a game using the trained model in the BipedalWalker-v3 environment.

    Parameters:
        model: The trained model to predict actions.
        human (bool): If True, the environment will be rendered for human interaction. Default is False.

    Returns:
        float: The total score achieved in the game.
    """
    # Choose the environment based on whether it's a human or AI play
    if human:
        test_env = gym.make("BipedalWalker-v3", render_mode="human")  # Render the environment for human interaction
    else:
        test_env = gym.make("BipedalWalker-v3")  # No rendering, for AI play

    score = 0.0  # Initialize the score
    obs, _ = test_env.reset()  # Reset the environment and get the initial observation
    done = truncated = False  # Initialize termination flags

    # Loop until the game is done or truncated
    while not (done or truncated):      
        action, _states = model.predict(obs)  # Get the action predicted by the model
        obs, reward, done, truncated, infos = test_env.step(action)  # Take a step in the environment
        score += reward  # Accumulate the reward

    return score  # Return the total score achieved


def play(num_episodes=3):
    """
    Play multiple episodes using the trained DDPG model in the BipedalWalker-v3 environment.

    Parameters:
        num_episodes (int): The number of episodes to play. Default is 3.

    """
    model = DDPG.load(model_filename, env=env)  # Load the pre-trained DDPG model

    # Play the specified number of episodes
    for i in range(num_episodes):
        score = play_game(model, human=True)  # Play one episode with human interaction
        print(f"Run {i} erziele einen Score von {score}")  # Print the score for this run



def showStrucure():
    """
    Display the architecture of the Actor and Critic networks of the trained DDPG model.
    This function prints out the structure of both networks to help understand their design.

    """
    model = DDPG.load(model_filename, env=env)  # Load the pre-trained DDPG model

    # Zugriff auf das Actor-Netzwerk (Policy Network)
    print("Actor Network Architecture:")
    print(model.actor)  # Print the architecture of the Actor network

    # Zugriff auf das Critic-Netzwerk (Value Network)
    print("\nCritic Network Architecture:")
    print(model.critic)  # Print the architecture of the Critic network


def paintStats():
    """
    Visualizes the training scores over timesteps by plotting them using Matplotlib.
    The scores are recorded every 25,000 timesteps during the DDPG training process.
    """
    with open(score_filename, 'rb') as f:
        scores = pickle.load(f)  # Load the training scores

    x = np.arange(len(scores)) * 25000  # Timesteps corresponding to the scores
    y = scores  # Fitness scores

    # Plotting
    plt.plot(x, y, marker='o', label='Scores')  # Fitness scores over timesteps
    plt.xlabel("Timestep")
    plt.ylabel('Fitness')
    plt.title('DDPG Stats')
    plt.legend()
    plt.show()




def main():
    """
    Main function that allows the user to choose between training a model, playing with a model, visualizing statistics, or inspecting the model's network architecture.
    """
    mode = input("Do you want to train (t), play (p), or view information about the used networks (info)? ").strip().lower()

    if mode == 't':
        step = int(input("How many timesteps should the model be trained for (100000 approx. 6min)? "))
        print(f"Training for {step} timesteps")
        print(f"Training will take approximately {step/100000 * 6} minutes") 
        train(timesteps=step)  # Train the model with the specified number of timesteps
    elif mode == 'p':
        try:
            num_episodes = int(input("How many times would you like to play? "))
            play(num_episodes)  # Play with the model for the specified number of episodes
        except ValueError:
            print("Please enter a valid number.")  # Error handling for invalid input
    elif mode == "paint":
        paintStats()  # Visualize training statistics
    elif mode == "info":
        showStrucure()  # Show network architecture information
    else:
        print("Invalid input. Please enter 't' to train or 's' to play.")

def make_env():
    """
    Creates and returns a gym environment for the BipedalWalker-v3 task with RGB array rendering mode.
    """
    return gym.make("BipedalWalker-v3", render_mode="rgb_array")




if __name__ == "__main__":
    model_filename = "BA NEAT\\ddpg_bipedalWalker"
    score_filename = "BA NEAT\\DDPG_stats"
    # Create a vectorized environment with multiprocessing
    num_envs = 4    # For parallel execution with SubprocVecEnv
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    main()