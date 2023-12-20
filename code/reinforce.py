import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./environment')
#import Six_Axis_Robot_Arm as bot
import Three_Axis_Robot_Arm as bot

# suppress scientific notation
np.set_printoptions(suppress=True)


def get_action_epsilon_greedy(robot, epsilon: float = 0.1) -> int:
    """Explore / Exploit, choose action from Q.

    :param robot: Robot to do the action on
    :type robot: Six_Axis_Robot_Arm object

    :param epsilon: Probability to explore
    :type epsilon: float

    :return: tuple with action index from robot action dict and action
    :rtype: ((float, float, float, float, float, float), int)
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore: select a random action
        return robot.get_random_action()
    else:
        # Exploit: select the best action based on current Q-values
        action = np.argmax(robot.get_current_qs())
        return robot.get_action_from_dict(action), action

def sarsa(robot, num_episodes, alpha=0.1, gamma=1.0, epsilon=0.1):
    # Assume robot is initialized
    episode_lengths = []
    for episode in range(num_episodes):
        # Initialize the starting state S (choosing from the starting positions)
        robot.reset()

        # Choose the first action A from S using Q (epsilon-greedy)
        current_action, current_action_index = get_action_epsilon_greedy(robot, epsilon)

        # Loop for each step of the episode
        done = False
        i = 0
        while not done:
            # Save Q
            last_q = robot.get_current_q(current_action_index)

            # Take action A, observe R, S'
            new_pos, reward, done = robot.do_move(current_action_index)

            # Choose A' from S' using Q (epsilon-greedy)
            new_action, new_action_index = get_action_epsilon_greedy(robot, epsilon)

            # Update the Q-value for the current state and action pair
            new_Q = alpha * (reward + gamma * robot.get_current_q(new_action_index) - last_q)
            robot.set_last_q(current_action_index, new_Q)
            #Q[current_state[0][0]][current_state[0][1]][inv_actions_dict[current_action]] += alpha * (reward + gamma * Q[new_state[0][0]][new_state[0][1]][inv_actions_dict[new_action]] - Q[current_state[0][0]][current_state[0][1]][inv_actions_dict[current_action]])

            # S <- S'; A <- A'
            current_action_index = new_action_index
            current_action = new_action
            i+=1
            if (i % 10000 == 0):
                print(f"Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        episode_lengths.append(i)
        print(f"Episode {episode} ended with length {i}")
    return episode_lengths

#arm = bot.Six_Axis_Robot_Arm()
arm = bot.Three_Axis_Robot_Arm()
#arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

num_episodes = 1000
alpha = 1/10
gamma = 0.99
epsilon = 0.1

episode_lengths = sarsa(arm, num_episodes, alpha, gamma, epsilon)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(episode_lengths)
print(f"average length of the last 100 episodes: {np.average(episode_lengths[-100:len(episode_lengths)])}")
print(f"last 10 episode lengths: {episode_lengths[-10:len(episode_lengths)]}")
plt.show()
