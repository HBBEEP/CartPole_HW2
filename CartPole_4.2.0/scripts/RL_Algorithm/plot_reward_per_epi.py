import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
# file_paths = ["/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/MC/metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/SARSA/metrics.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Q_Learning/metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics.csv", ]
# labels = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]  # Labels for legend

# file_paths = [#"/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/MC/metrics2.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/SARSA/metrics.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Q_Learning/metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics.csv", ]
# labels = ["SARSA", "Q_Learning", "Double_Q_Learning"]  # Labels for legend

# file_paths = ["/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics_act5.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics_act20.csv",
#                ]
# labels = ["Double Q_Learning (act = 5)", "Double Q_Learning (act = 10)", "Double Q_Learning (act = 20)"]  # Labels for legend


# file_paths = ["/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics_obs5.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics_obs20.csv",
#                ]
# labels = ["Double Q_Learning (obs = [5, 5, 5, 5])", "Double Q_Learning (obs = [10, 10, 10, 10])", "Double Q_Learning (obs = [20, 20, 20, 20])"]  # Labels for legend

# file_paths = ["/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/MC/metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/SARSA/metrics.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Q_Learning/metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/metrics.csv", ]
# labels = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]  # Labels for legend


# file_paths = ["/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/MC/play_metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/SARSA/play_metrics.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Q_Learning/play_metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics.csv", ]
# labels = ["MC", "SARSA", "Q_Learning", "Double_Q_Learning"]  # Labels for legend

# file_paths = [
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/SARSA/play_metrics.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Q_Learning/play_metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics.csv", ]
# labels = ["SARSA", "Q_Learning", "Double_Q_Learning"]  # Labels for legend

# file_paths = [
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics_obs5.csv",
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics.csv", 
#               "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics_obs20.csv", ]
# labels = ["Double Q_Learning (obs = [5, 5, 5, 5])", "Double Q_Learning (obs = [10, 10, 10, 10])", "Double Q_Learning (obs = [20, 20, 20, 20])"]  # Labels for legend


file_paths = [
              "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics_act5.csv",
              "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics.csv", 
              "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/rl_metrics/Stabilize/Double_Q_Learning/play_metrics_act20.csv", ]
labels = ["Double Q_Learning (act = 5)", "Double Q_Learning (act = 10)", "Double Q_Learning (act = 20)"]  # Labels for legend

# colors = ["blue", "green", "red", "purple"]  # Define colors for each plot
colors = ["blue", "green", "red"]  # Define colors for each plot

use_mean = True

plt.figure(figsize=(10, 6))

# Loop through each file and plot
for file_path, label, color in zip(file_paths, labels, colors):
    df = pd.read_csv(file_path)
    mean_reward = df["Reward"].mean()
    print(f"Mean reward for {label}: {mean_reward}")

    if (use_mean):

        df["Episode_Group"] = df["Episode"][:] // 5  # Group episodes into bins of 10
        avg_rewards = df.groupby("Episode_Group")["Reward"].mean()  # Compute mean for each bin
        
        # Plot with specified color
        plt.plot(avg_rewards.index * 5, avg_rewards, label=label, color=color)
    else:
        # Plot with specified color
        plt.plot(df["Episode"], df["Reward"], label=label, color=color)

# Plot settings
plt.xlabel("Episode")
plt.ylabel("Average Reward (per 100 episodes)")
plt.title("Comparison of Rewards Over Episodes")
plt.legend()
plt.grid()

# Show plot
plt.show()