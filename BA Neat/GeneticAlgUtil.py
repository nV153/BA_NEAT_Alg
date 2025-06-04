import os
import pickle
from typing import List, Tuple, Any

def save_hall_of_fame(hof: Any, filename: str) -> None:
    """
    Saves the Hall of Fame to a file.

    Args:
        hof (Any): The Hall of Fame to be saved.
        filename (str): The file path where the Hall of Fame will be stored.
    """
    with open(filename, 'wb') as f:
        pickle.dump(hof, f)


def load_hall_of_fame(filename: str) -> Any:
    """
    Loads the Hall of Fame from a file.

    Args:
        filename (str): The file path from which the Hall of Fame will be loaded.

    Returns:
        Any: The loaded Hall of Fame.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_population(population_file: str) -> List[Any]:
    """
    Loads the population from a file.

    Args:
        population_file (str): The file path from which the population will be loaded.

    Returns:
        List[Any]: The loaded population.
    """
    with open(population_file, 'rb') as f:
        return pickle.load(f)


def save_population(population: List[Any], population_file: str) -> None:
    """
    Saves the population to a file.

    Args:
        population (List[Any]): The population to be saved.
        population_file (str): The file path where the population will be stored.
    """
    with open(population_file, 'wb') as f:
        pickle.dump(population, f)


def get_used_run_numbers(base_path: str, version: int) -> List[int]:
    """
    Returns a list of already used run numbers, based on the provided version.

    Args:
        base_path (str): The base path to the HallOfFame folder.
        version (int): The version of the algorithm.

    Returns:
        List[int]: A list of used run numbers.
    """
    used_runs: List[int] = []

    # Determine the prefix based on the version
    if version == 1:
        prefix: str = "Vers_GA_"
    elif version == 2:
        prefix: str = "Vers_PSO_"
    else:
        prefix: str = f"RunV{version}_"

    # Check existing files in the HallOfFame folder
    for file_name in os.listdir(base_path):
        if file_name.startswith(prefix) and file_name.endswith("_hof.pkl"):
            # Extract run number from the file name
            try:
                # The run number is after the prefix and before '_hof.pkl'
                run_number_str: str = file_name.split(prefix)[1].split('_hof.pkl')[0]
                run_number: int = int(run_number_str)
                used_runs.append(run_number)
            except (IndexError, ValueError):
                print(f"File '{file_name}' does not have a valid run number.")  # Debug output

    return used_runs


def get_run_number() -> int:
    """
    Asks the user for a run number and validates the input.

    Returns:
        int: The entered run number.
    """
    while True:
        try:
            run_num: int = int(input("Please enter the run number: "))
            return run_num
        except ValueError:
            print("Invalid number. Please enter an integer.")


def get_file_paths(version: int, run_num: int) -> Tuple[str, str, str]:
    """
    Generates file paths based on the version and run number.

    Args:
        version (int): The version number to determine the file path format.
        run_num (int): The run number to append to the file paths.

    Returns:
        Tuple[str, str, str]: A tuple containing the paths to the Hall of Fame file,
                              population file, and the Excel results file.
    """
    local_dir: str = os.path.dirname(os.path.abspath(__file__))
    hof_dir: str = os.path.join(local_dir, "HallOfFame")
    excel_results_dir: str = os.path.join(local_dir, "Excel Results")
    
    # Determine the version string based on the version
    if version == 1:
        version_str: str = f"Vers_GA_{run_num}"
    elif version == 2:
        version_str: str = f"Vers_PSO_{run_num}"
    else:
        version_str: str = f"RunV{version}_{run_num}"
    
    hall_of_fame_path: str = os.path.join(hof_dir, f"{version_str}_hof.pkl")
    population_path: str = os.path.join(hof_dir, f"{version_str}_pop.pkl")
    excel_file_path: str = os.path.join(excel_results_dir, f"{version_str}_results.xlsx")

    return hall_of_fame_path, population_path, excel_file_path


def delete_hof_and_excel(run_num: int, base_path: str, version: int) -> None:
    """
    Deletes the Hall of Fame, Excel, and Population files for the given run number and version.

    Args:
        run_num (int): The run number whose files are to be deleted.
        base_path (str): The base path to the HallOfFame and Excel Results folders.
        version (int): The version of the algorithm.
    """
    # Determine the version string based on the version
    if version == 1:
        version_str: str = f"Vers_GA_{run_num}"
    elif version == 2:
        version_str: str = f"Vers_PSO_{run_num}"
    else:
        version_str: str = f"RunV{version}_{run_num}"

    hof_file_path: str = os.path.join(base_path, "HallOfFame", f"{version_str}_hof.pkl")
    excel_file_path: str = os.path.join(base_path, "Excel Results", f"{version_str}_results.xlsx")
    pop_file_path: str = os.path.join(base_path, "HallOfFame", f"{version_str}_pop.pkl")

    # Delete the Hall of Fame file
    if os.path.exists(hof_file_path):
        os.remove(hof_file_path)
        print(f"Hall of Fame file for {version_str} deleted.")
    else:
        print(f"Hall of Fame file for {version_str} not found.")

    # Delete the Excel file
    if os.path.exists(excel_file_path):
        os.remove(excel_file_path)
        print(f"Excel file for {version_str} deleted.")
    else:
        print(f"Excel file for {version_str} not found.")

    # Delete the Population file
    if os.path.exists(pop_file_path):
        os.remove(pop_file_path)
        print(f"Population file for {version_str} deleted.")
    else:
        print(f"Population file for {version_str} not found.")


def log_training_time(run_num: int, num_generations: int, elapsed_time: float, version: int) -> None:
    """
    Logs the training time in a TXT file in the Excel Results folder, depending on the version.

    Args:
        run_num (int): The run number.
        num_generations (int): The number of generations trained.
        elapsed_time (float): The elapsed training time in seconds.
        version (int): The version of the algorithm.
    """
    local_dir: str = os.path.dirname(os.path.abspath(__file__))
    
    # Mapping the version to a string
    if version == 0:
        version_str: str = "NEAT"
    else:
        version_str: str = f"Version{version}"  # Fallback for unknown versions

    # Base path for the Excel Results folder
    results_dir: str = os.path.join(local_dir, "Excel Results")
    os.makedirs(results_dir, exist_ok=True)  # Create the folder if it doesn't exist
    time_file_path: str = os.path.join(results_dir, "Exp2Times.txt")

    # Write the time to the TXT file
    with open(time_file_path, 'a') as time_file:
        time_file.write(f"{version_str} Run{run_num} Generation {num_generations} Duration: {elapsed_time:.2f} seconds\n")


