import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('./environment')
#import Six_Axis_Robot_Arm as bot
import Three_Axis_Robot_Arm as bot

# suppress scientific notation
np.set_printoptions(suppress=True)

def debug_pause(string_to_print=None):
    if string_to_print is not None:
        print(string_to_print)
    input("Press Enter to continue...")

def get_action_epsilon_greedy(robot, epsilon: float = 0.1, verbosity_level=0) -> int:
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
        action = robot.get_random_action()
        if verbosity_level >= 2: print(f"  Explore with action: {action}")
        return action
    else:
        # Exploit: select the best action based on current Q-values
        current_qs = robot.get_current_qs()
        action = np.argmax(current_qs)
        action_tuple = (robot.get_action_from_dict(action), action)
        if verbosity_level >= 2: print(f"  Exploiting with action: {action_tuple}")
        return action_tuple
    


def sarsa(robot, num_episodes, alpha=0.1, gamma=1.0, epsilon=0.1, verbosity_level=0):
    # Assume robot is initialized
    episode_lengths = []
    for episode in range(num_episodes):
        start_time = time.time()
        # Initialize the starting state S (choosing from the starting positions)
        robot.reset()
        if verbosity_level >= 2: print(f"Robot initialized, starting SARSA, Starting angles: {robot.get_joint_angles_rad()}, Starting position: {robot.rob.end_effector_position()} Choosing action!")
        if verbosity_level >= 2: print(f"   Initial Qs for starting_voxel:\n{robot.Q[robot.voxels_index_dict[(-500, 0, 0)]]}")

        # Choose the first action A from S using Q (epsilon-greedy)
        current_action, current_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)

        # Loop for each step of the episode
        done = False
        i = 0

        # remember best reward
        best_reward = -1

        while not done:
            if verbosity_level >= 2: print(f"\nNew loop, iteration = {i}, current Voxel: {robot.current_voxel}")
            if verbosity_level >= 2: debug_pause(f" Qs for current Voxel:\n {robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")
            
            # Save Q
            last_q = robot.get_current_q(current_action_index)

            if verbosity_level >= 2: print(f"  Q for current action {current_action} with index {current_action_index} at voxel {robot.current_voxel}: {last_q}. Doing SAR")

            # Take action A, observe R, S'
            new_pos, reward, done = robot.do_move(current_action_index)
            
            if verbosity_level >= 2: print(f"  New position: {new_pos}, reward: {reward}, new Voxel: {robot.current_voxel}. Doing SA")
            if verbosity_level >= 2: print(f"  Qs for new position:\n{robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")
            if reward > best_reward:
                best_reward = reward


            # Choose A' from S' using Q (epsilon-greedy)
            new_action, new_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)

            new_q = robot.get_current_q(new_action_index)

            if verbosity_level >= 2: print(f"  Q for action {new_action} with index {new_action_index} at voxel {robot.current_voxel}: {new_q}. Updating Q values.")

            # Update the Q-value for the current state and action pair
            new_q = last_q + alpha * (reward + gamma * new_q - last_q)
            robot.set_last_q(current_action_index, new_q)


            if verbosity_level >= 2: print(f"  Updated Q value: {new_q}")

            if verbosity_level >= 2: print(f"  Updated Qs for last Voxel:\n{robot.Q[robot.voxels_index_dict[robot.last_voxel]]}")

            # S <- S' <- done in robot.do_move(); A <- A'
            current_action_index = new_action_index
            current_action = new_action
            i += 1
            if (i % 10000 == 0) and verbosity_level >= 1:
                print(f"Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                print(f"Move / out of bound ratio: {robot.out_of_bounds_counter/i}, best reward: {best_reward}")
                print(f"Percentage of track done in furthest move: {round((best_reward + 1)*100, 3)}%")
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        end_time = time.time()
        episode_lengths.append(i)
        print(f"Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s")
        algo = 'sarsa' 
    return episode_lengths, algo



def n_step_sarsa(robot, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1, verbosity_level=0):
    
    # Assume robot is initialized    
    n=5
    episode_lengths = []    
    for episode in range(num_episodes):
        start_time = time.time()
        # Initialize the starting state S (choosing from the starting positions)
        robot.reset()

        if verbosity_level >= 2: print(f"Robot initialized, starting SARSA, Starting angles: {robot.get_joint_angles_rad()}, Starting position: {robot.rob.end_effector_position()} Choosing action!")
        if verbosity_level >= 2: print(f"   Initial Qs for starting_voxel:\n{robot.Q[robot.voxels_index_dict[(-500, 0, 0)]]}")

        
        current_state = robot.get_tcp()
        
        # Choose the first action A from S using Q (epsilon-greedy)
        current_action, current_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)
        print('current_action: ', current_action)
        print('current_action_index: ', current_action_index)


        # Loop for each step of the episode
        done = False
        i = 0

        states = []
        actions = []        
        actions_index = []
        rewards = []

        states.append(current_state)
        actions.append(current_action)
        actions_index.append(current_action_index)


        G=0
        best_reward = -1

        while not done:   
            if verbosity_level >= 2: print(f"\nNew loop, iteration = {i}, current Voxel: {robot.current_voxel}")
            if verbosity_level >= 2: debug_pause(f" Qs for current Voxel:\n {robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")
            
            # Save Q
            last_q = robot.get_current_q(actions_index[0]) 
            if verbosity_level >= 2: print(f"  Q for current action {current_action} with index {current_action_index} at voxel {robot.current_voxel}: {last_q}. Doing SAR")
            
            
            # Take action A, observe R, S'
            new_pos, reward, done = robot.do_move(current_action_index)  


            if verbosity_level >= 2: print(f"  New position: {new_pos}, reward: {reward}, new Voxel: {robot.current_voxel}. Doing SA")
            if verbosity_level >= 2: print(f"  Qs for new position:\n{robot.Q[robot.voxels_index_dict[robot.current_voxel]]}")
            if reward > best_reward:
                best_reward = reward

            # Choose A' from S' using Q (epsilon-greedy)
            new_action, new_action_index = get_action_epsilon_greedy(robot, epsilon, verbosity_level=verbosity_level)
            
            rewards.append(reward)           
            states.append(new_pos) 
            actions.append(new_action)
            actions_index.append(new_action_index)


            if(len(rewards)==n): 
                # calculate n-step reward
                for rew in range(len(rewards)-1):
                    G += (gamma**rew)*rewards[rew]

                #last_q = robot.get_current_q(actions_index[0])
                last_n_q = robot.get_last_n_q(actions_index[0])            


                new_q = robot.get_current_q(new_action_index)
                #if verbosity_level >= 2: print(f"  Q for action {new_action} with index {new_action_index} at voxel {robot.current_voxel}: {new_q}. Updating Q values.")

                # Update the Q-value for the current state and action pair
                #new_q = last_n_q + alpha * (G+gamma**n*new_q - last_n_q)
                last_n_q = last_n_q + alpha * (G+gamma**n*new_q - last_n_q)

                #last_q = last_q + alpha * (G+gamma**n*new_q - last_q)
                
                
                #robot.set_last_q(actions_index[0], last_q)
                robot.set_last_n_q(actions_index[0], last_n_q)


                #if verbosity_level >= 2: print(f"  Updated Q value: {new_q}")
                # only store the last n iterations
                del(rewards[0])
                del(states[0])
                del(actions[0])
                del(actions_index[0])   


            #if verbosity_level >= 2: print(f"  Updated Qs for last Voxel:\n{robot.Q[robot.voxels_index_dict[robot.last_voxel]]}")
            # S <- S' <- done in robot.do_move(); A <- A'
            current_action_index = new_action_index
            current_action = new_action
            
            i += 1
            G=0

            if (i % 10000 == 0) and verbosity_level >= 1:
                print(f"Current_iteration {i} in episode {episode}, current_pos {new_pos}, Out of bounds: {robot.out_of_bounds_counter}")
                print(f"Move / out of bound ratio: {robot.out_of_bounds_counter/i}, best reward: {best_reward}")
                print(f"Percentage of track done in furthest move: {round((best_reward + 1)*100, 3)}%")
                print(f"alpha = {alpha} gamma = {gamma} epsilon = {epsilon}")
                #robot.show(draw_path=True, draw_voxels=True, zoom_path=True)

        end_time = time.time()
        episode_lengths.append(i)
        print(f"Episode {episode} ended with length {i}. Time for eposiode: {end_time-start_time} s")  
        algo = 'sarsa_n_steP'      

    return episode_lengths, algo



#arm = bot.Six_Axis_Robot_Arm()
arm = bot.Three_Axis_Robot_Arm()
arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

num_episodes = 10000
alpha = 1/25
gamma = 0.99
epsilon = 0.1

#episode_lengths, algo = sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1)

episode_lengths, algo = n_step_sarsa(arm, num_episodes, alpha, gamma, epsilon, verbosity_level=1)


fig, ax = plt.subplots(figsize=(10, 10))
ax.set_yscale('log')
ax.plot(episode_lengths)

ax.set_title(f"Algorithm: {algo}")
ax.set_xlabel('Episoden')
ax.set_ylabel('Episode length')

print(f"average length of the last 100 episodes: {np.average(episode_lengths[-100:len(episode_lengths)])}")
print(f"last 10 episode lengths: {episode_lengths[-10:len(episode_lengths)]}")
plt.savefig(f'{algo}_plot.png')
plt.show()
