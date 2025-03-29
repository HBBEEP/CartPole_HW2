import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load Q-values from a JSON file
# file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/MC/qvalue.json"
# file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/SARSA/qvalue.json"
# file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/Q_Learning/qvalue.json"
# file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/Double_Q_Learning/qvalue.json"
# file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/Double_Q_Learning/qvalue_act5.json"
# file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/Double_Q_Learning/qvalue_act20.json"
# file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/Double_Q_Learning/qvalue_obs5.json"
file_path = "/home/hbbeep/FRA503-Deep-Reinforcement-Learning-for-Robotics/CartPole_4.2.0/q_value/Stabilize/Double_Q_Learning/qvalue_obs20.json"


with open(file_path, "r") as file:
    q_data = json.load(file)["qa_values"]
    # q_data = json.load(file)["qb_values"]
    # q_data = json.load(file)["q_values"]

# Extract coordinates and values
X, Y, Z = [], [], []
# flag = True
# x = 0
for key, values in q_data.items():
    coords = tuple(map(int, key.strip("()").split(", ")))  # Convert key to tuple


    for i, q_value in enumerate(values):
        X.append(coords[1])  # Use first coordinate as X
        Y.append(i)           # Use index as Y
        Z.append(q_value)     # Q-value as Z
    #     if flag:
    #         x+=1
    #         print(x, coords[1])
    # flag = False
            
# Convert to numpy arrays
X, Y, Z = np.array(X), np.array(Y), np.array(Z)

# Create 3D surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('Pole Pose')
ax.set_ylabel('Action Index')
ax.set_zlabel('Q-Value')
# ax.set_title('3D Surface Plot of Q-Values (MC)')
# ax.set_title('3D Surface Plot of Q-Values (SARSA)')
# ax.set_title('3D Surface Plot of Q-Values (Q_Learning)')
ax.set_title('3D Surface Plot of Q-Values [A] (Double_Q_Learning)')
# ax.set_title('3D Surface Plot of Q-Values [B] (Double_Q_Learning)')

# Show plot
plt.show()
