import csv
import os

def collect_and_save_metrics(episode, reward, epsilon, file_path):
    """Collect reward and epsilon per episode and save to a CSV file."""
    file_exists = os.path.isfile(file_path)
    
    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["Episode", "Reward", "Epsilon"])
        
        # Write data
        writer.writerow([episode, reward, epsilon])

        