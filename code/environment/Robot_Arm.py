"""
Robotic Arm Simulation and Learning Environment

This module provides a comprehensive environment for simulating and learning robotic arm movements.
It leverages the Maddux library for robot visualization and arm manipulation. The core class, Robot_Arm,
encompasses a wide range of functionalities including initializing the robot arm, setting and getting joint
angles, executing movements based on learned actions, and visualizing these movements.

Key Features:
- Initialization of a robotic arm with customizable parameters such as section length, number of axes, and
  starting position.
- Implementation of a mechanism where the arm can explore actions and update its strategies based
  on rewards.
- Support for loading and saving learned values, facilitating the continuation of learning processes.
- Visualization capabilities including drawing the robot's path, voxels, and animations of its movements.

Dependencies:
- numpy for numerical operations.
- matplotlib and Maddux library for visualization and animation of the robotic arm.
- Various other standard Python libraries for handling file operations and mathematical computations.

Note:
The module assumes the presence of the Maddux library for robot arm simulation. Ensure that Maddux and
its dependencies are correctly installed and configured in your environment.

Authors: F. M. Sokol, N. M. Hahn, M. Ubbelohde
"""

import numpy as np
import matplotlib.pyplot as plt
# maddux for robot visualization
from maddux.robots.link import Link
from maddux.robots.arm import Arm
from maddux.environment import Environment
import time
from path import Path
import itertools
import ujson
import math
from scipy.interpolate import interp1d

# suppress scientific notation
np.set_printoptions(suppress=True)


class Robot_Arm:
    """
    A simulation class for a robotic arm using direct / inverse kinematics and reinforcement learning.

    This class represents a robotic arm, capable of executing movements and learning optimal paths
    in a simulated environment. It utilizes direct kinematics for precise joint angle control and
    implements a reinforcement learning framework for the arm to learn movements towards a target path.
    The class provides functionalities for setting and getting joint angles, managing arm movements,
    visualizing the arm's path, and handling the learning process.

    Attributes:
        voxels_index_dict (dict): A dictionary mapping voxel positions to unique indices.
        winning_voxels (list): A list of voxel positions that represent winning states.
        Q (list): A list of Q-values used in the reinforcement learning algorithm.
        num_axis (int): Number of axes/joints in the robotic arm.
        helix_section (int): Current section of the helix being processed.
        section_length (float): Length of each section of the arm.
        path (list): The path that the arm is supposed to follow.
        rob (Arm object): Instance of the Arm class representing the robotic arm.
        env (Environment object): Instance of the Environment class representing the simulation environment.
        actions_dict (dict): A dictionary mapping action indices to arm movements.
        inv_actions_dict (dict): Inverse of the actions_dict.
        current_voxel (tuple): Current voxel position of the robotic arm's end effector.
        last_voxel (tuple): The last voxel position where the end effector was located.
        n (int): Number of steps to consider in the n-step reinforcement learning.
        last_n_voxel (list): List of the last 'n' voxels visited by the end effector.
        out_of_bounds_counter (int): Counter for the number of times the arm goes out of bounds.
        reward_out_of_bounds (int): Reward value for going out of bounds.
        reward_win (int): Reward value for reaching a winning voxel.
        move_along_q_in_section (int): Indicates the section of the arm movement while following Q-values.
        q_path (list): List storing the path taken by the end effector.
        starting_pos (tuple): Starting position of the robotic arm.

    Methods:
        get_joint_angles_degrees: Returns the current joint angles in degrees.
        get_joint_angles_rad: Returns the current joint angles in radians.
        set_joint_angles_degrees: Sets the arm's joint angles using degrees.
        set_joint_angles_rad: Sets the arm's joint angles using radians.
        reset: Resets the arm to its starting state.
        get_random_action: Returns a random action from the available actions.
        get_current_qs: Returns the Q-values for the current state.
        do_move: Executes a movement based on the specified action.
        animate_move_along_q_values: Animates the arm's movement along learned Q-values.
        show: Opens a window and draws the robotic arm with optional path and voxel visualization.
        animate: Animates the robotic arm's movement with optional path and voxel visualization.
        save_learned_to_file: Saves learned Q-values, voxels, and other data to a file.
        load_learned_from_file: Loads learned Q-values, voxels, and other data from a file.
        stitch_from_file: Stitches the next segment of voxels and Q-values from file to the robot's attributes.
        get_finishing_angles_rad: Determines the finishing angles of the arm based on learned movements.
        calc_mse: Calculates the Mean Squared Error between the actual path and learned path.
    """

    def __init__(self, starting_pos: (float, float, float) = (-500, 0, 0),
                 section_length=1, helix_section=0,
                 voxel_volume=1, num_axis=6) -> None:
        """
        Initialize a robotic arm with specified parameters.

        This constructor initializes the robotic arm for simulation and learning. It sets up the arm with a given 
        number of axes, a specified section of a helix to follow, and initializes the learning environment. 
        The arm is configured with a set of links (each link has one joint) and is capable of inverse kinematics 
        for precise control. The learning aspect involves creating a dictionary of possible actions and setting 
        up an environment for reinforcement learning.
        Args:
            starting_pos (tuple): The starting position (x, y, z) of the helix path for the arm.
            section_length (float): The length of the helix section that the arm will learn to navigate.
            helix_section (int): The specific section of the helix that the arm is currently working on.
            voxel_volume (int): The volume of the voxels used in the learning environment.
            num_axis (int): The number of axes or joints in the robotic arm.

        Returns:
            None
        """
        # Create array for voxels, q-values and winning voxels
        self.voxels_index_dict = [None]
        self.winning_voxels = [None]
        self.Q = [None]

        # save number of axis
        self.num_axis = num_axis

        self.helix_section = helix_section

        if(helix_section != int((1/section_length))-1):
            # make the section a little longer so each section overlaps a litle
            self.section_length = section_length*1.2
        else:
            # last section, don't make it longer
            self.section_length = section_length
        helix_section = helix_section * section_length

        # Create path for the robot
        path = Path(helix_start=starting_pos, max_distance=voxel_volume,
                    generate_percentage_of_helix=self.section_length, generate_start=helix_section)

        # save voxels, winning_voxels and rewards in lists
        self.voxels, self.winning_voxels[0], self.rewards = path.get_helix_voxels()

        # save path
        self.path = path.get_helix_data()

        # Create hashtable of voxels with a unique index for each voxel
        self.voxels_index_dict[0] = {value: index for index, value in enumerate(self.voxels)}

        # Create links for robot arm
        if num_axis == 6:
            # Create a series of links (each link has one joint)
            # (theta, offset, length, twist, q_lim=None)
            L1 = Link(0, 151.85, 0, 1.570796327, link_size=5)
            L2 = Link(0, 0, -243.55, 0, link_size=4)
            L3 = Link(0, 0, -213.2, 0, link_size=4)
            L4 = Link(0, 131.05, 0, 1.570796327, link_size=3)
            L5 = Link(0, 85.35, 0, -1.570796327, link_size=2)
            L6 = Link(0, 92.1, 0, 0, link_size=0.1)
            links = np.array([L1, L2, L3, L4, L5, L6])
            # Initial arm angles
            q0 = np.array((0, 0, 0, 0, 0, 0))
        elif num_axis == 3:
            # Create a series of links (each link has one joint)
            # (theta, offset, length, twist, q_lim=None)
            L1 = Link(0, 151.85, 0, 1.570796327, link_size=5)
            L2 = Link(0, 0, -300.00, 0, link_size=4)
            L3 = Link(0, 0, -300.00, 0, link_size=0.2)
            links = np.array([L1, L2, L3])
            # Initial arm angles
            q0 = np.array((0, 0, 0))
        else:
            raise ValueError(f"\033[91mNumber of robot axis num_axis must be either 3 or 6. It currently is: {num_axis}")
            return

        # Create arm
        self.rob = Arm(links, q0, '1-link')

        # Do inverse kinematics for the starting position and
        # create a new arm and set it to the start of the helix
        self.starting_angles = self.rob.ikine((self.path[0][0], self.path[1][0], self.path[2][0]))
        # do mod 2pi to starting angles to not get crazy large angles
        self.starting_angles = np.array([angle % (2*np.pi) for angle in self.starting_angles])
        # Create arm
        self.rob = Arm(links, self.starting_angles, '1-link')

        self.env = Environment(dimensions=[1500.0, 1500.0, 1500.0],
                               robot=self.rob)

        # Create all possible actions
        # Define possible actions for each joint in deg
        joint_actions_deg = [-0.1, 0, 0.1]
        # convert actionspace to radians
        joint_actions_rad = np.array([self.__deg_to_rad(action) for action in joint_actions_deg])

        # Generate all possible action combinations for the joints
        if num_axis == 6:
            action_combinations = list(itertools.product(joint_actions_rad, repeat=6))
            total_amount_actions = len(action_combinations)
        else:
            action_combinations = list(itertools.product(joint_actions_rad, repeat=3))
            total_amount_actions = len(action_combinations)

        # Create a dictionary to map each combination to a unique integer
        self.actions_dict = {i: action for i, action in enumerate(action_combinations)}
        # Create inverse actiuon dict, as a user moght want to know what index a specific action has
        self.inv_actions_dict = {v: k for k, v in self.actions_dict.items()}

        # Create variable for voxel of current TCP position, so it only needs to be calculated
        # when the TCP is changed
        self.current_voxel = self.__get_tcp_voxel_position()
        # Save voxel the TCP was in before the current voxel to be able to set last Q
        self.last_voxel = None

        # Save last n voxels to be able to set last Qs
        self.n = 5
        self.last_n_voxel = []

        # Init out of bounds counter for debugging
        self.out_of_bounds_counter = 0

        # Move robot to the start of the helix desired section
        section_length_path = len(self.path[0]) * section_length

        for i in range(0, self.helix_section+1):
            #print(f"Iteration: {i} of {self.helix_section}")
            current_place_in_path = int(i * section_length_path)
            self.starting_angles = self.rob.ikine((self.path[0][current_place_in_path], self.path[1][current_place_in_path], self.path[2][current_place_in_path]), set_robot=False)
            self.set_joint_angles_rad(self.starting_angles, save=True)

        # Set starting angles in robot
        # Overwrite Q0 of the robot arm
        self.rob.q0 = self.starting_angles

        # Remember which section to stitch
        self.section_to_stitch = 1

        # Reset robot arm to starting position
        self.reset()

        # Set reward for going out of bounds and winning.
        # Other rewards are calculated in Path class
        self.reward_out_of_bounds = -5
        self.reward_win = 0

        # Create Q
        amount_voxels = len(self.voxels)
        self.Q[0] = -np.random.rand(amount_voxels, total_amount_actions)

        # Set ending positions to zero in Q
        for winning_voxel in self.winning_voxels[0]:
            self.Q[0][self.voxels_index_dict[0][winning_voxel]] = np.zeros(total_amount_actions)

        # Remember which Q values to use to traverse helix
        self.move_along_q_in_section = 0

        # Array to save the path taken by the TCP. The starting position is the starting position of the helix
        self.q_path = [[starting_pos[0]], [starting_pos[1]], [starting_pos[2]]]
        self.starting_pos = starting_pos

    def __deg_to_rad(self, deg: float) -> float:
        """Convert degree to radians.

        Args:
            deg (float): Degrees to convert to radians.

        Returns:
            float: Radians equivalent of the input degrees.
        """
        return deg*np.pi/180

    def __rad_to_deg(self, rad: float) -> float:
        """Convert radians to degrees.

        Args:
            rad (float): Radians to convert to degrees.

        Returns:
            float: Degrees equivalent of the input radians.
        """
        return rad*180/np.pi

    def __limit_angle(self, angle: float) -> float:
        """Limit angle (in rad) to +-pi (+-180°).

        :param rad: Angle in radians
        :type rad: float

        :return: Limited angle in radians
        :rtype: float
        """
        if angle > np.pi/2:
            return np.pi
        if angle < -np.pi:
            return -np.pi
        return angle

    def __limit_angles(self, angles: (float, float, float, float, float, float)) -> (float, float, float, float, float,):
        """Limit all angles of the robot (in rad) to +-pi (+-180°).

        Args:
            angle (float): Angle in radians to be limited.

        Returns:
            float: Angle limited to the range of +-pi.
        """
        # no limit for now.
        return angles

    def __get_tcp_voxel_position(self) -> (int, int, int):
        """Get the current voxel position of the robotic arm's end effector (TCP).

        Returns:
            tuple: Coordinates of the voxel containing the end effector.
        """
        # Compute forward kinematics to receive current TCP
        tcp = self.rob.end_effector_position()
        x = tcp[0]
        y = tcp[1]
        z = tcp[2]
        return (int(round(x, 0)), int(round(y, 0)), int(round(z, 0)))

    def __check_win(self) -> bool:
        """Check if the current position is in a winning voxel.

        :return: Bool indicating if the TCP is in a winning voxel
        :rtype: bool
        """
        return self.current_voxel in self.winning_voxels[self.move_along_q_in_section]

    def __check_in_voxels(self) -> bool:
        """Check if the robotic arm's end effector (TCP) is currently within a defined voxel.

        Returns:
            bool: True if the TCP is within a voxel, False otherwise.
        """
        #print(f"Check in Voxels.\n  Current Voxel: {self.current_voxel}\n  Voxels index dict: {self.voxels_index_dict[0]}")
        return self.current_voxel in self.voxels_index_dict[self.move_along_q_in_section]

    def __get_reward(self) -> float:
        """Retrieve the reward for the robotic arm's current position.

        Returns:
            float: Reward value for the current position.
        """
        return self.rewards[self.voxels_index_dict[0][self.current_voxel]]

    def __interpolate_path(self, x, y, z, new_length):
        """Interpolate a 3D path to a new length using linear interpolation.

        Args:
            x (list): X-coordinates of the path.
            y (list): Y-coordinates of the path.
            z (list): Z-coordinates of the path.
            new_length (int): The desired length of the interpolated path.

        Returns:
            tuple: Interpolated path as three lists (x, y, z).
        """
        # Create an array of indices
        old_indices = np.linspace(0, 1, num=len(x))
        new_indices = np.linspace(0, 1, num=new_length)

        # Interpolate each dimension
        f_x = interp1d(old_indices, x, kind='linear')
        f_y = interp1d(old_indices, y, kind='linear')
        f_z = interp1d(old_indices, z, kind='linear')

        # Generate the new path
        new_x = f_x(new_indices)
        new_y = f_y(new_indices)
        new_z = f_z(new_indices)

        return new_x, new_y, new_z

    def __mean_squared_error(self, path_1, path_2):
        """
        Calculate the mean squared error (MSE) between two paths.

        This method computes the mean squared error, a common measure of the differences between
        two sets of values, between two provided paths. The paths are assumed to be sequences of
        points (usually representing coordinates) and the MSE is calculated over these points.

        Args:
            path_1 (list): The first path as a list of coordinates.
            path_2 (list): The second path as a list of coordinates to compare with the first path.

        Returns:
            float: The mean squared error between the two paths.
        """
        # Calculate squared errors
        squared_errors = (path_1 - path_2) ** 2

        # Calculate mean squared error
        mse = np.mean(squared_errors)
        return mse

    def get_joint_angles_degrees(self) -> (float, float, float, float, float,):
        """Return current joint angles in degrees.

        Returns:
            tuple: Current joint angles in degrees.
        """
        return np.array([self.__rad_to_deg(angle) for angle in self.rob.get_current_joint_config()])

    def get_joint_angles_rad(self) -> (float, float, float, float, float,):
        """Return current joint angles in radians.

        Returns:
            tuple: Current joint angles in radians.
        """
        return self.rob.get_current_joint_config()

    def set_joint_angles_degrees(self, angles: (float, float, float, float, float,), save=False) -> None:
        """Set joint angles in degrees.

        Args:
            angles (tuple): Joint angles in degrees.
            save (bool, optional): Whether to save the new angles to the robot's state.

        Returns:
            None
        """
        # Convert degrees of angles to radians
        angles_rad = np.array([self.__deg_to_rad(angle) for angle in angles])
        self.set_joint_angles_rad(angles_rad, save=save)

    def set_joint_angles_rad(self, angles: (float, float, float, float, float,), save=False, set_last_voxel=True) -> None:
        """Set joint angles in radians.

        Args:
            angles (tuple): Joint angles in radians.
            save (bool, optional): Whether to save the new angles to the robot's state.
            set_last_voxel (bool, optional): Whether to update the 'last voxel' position.

        Returns:
            None
        """
        # Limit angles to +-180°
        angles_rad = self.__limit_angles(angles)
        self.rob.update_angles(angles_rad, save=save)

        if set_last_voxel is True:
            self.last_voxel = self.current_voxel
            self.last_n_voxel.append(self.current_voxel)
            if len(self.last_n_voxel)>self.n:
                del(self.last_n_voxel[0])

        self.current_voxel = self.__get_tcp_voxel_position()

    def reset(self) -> None:
        """Reset robot to starting state.

        Returns:
            None
        """
        self.rob.reset(save=False)
        self.out_of_bounds_counter = 0
        self.current_voxel = self.__get_tcp_voxel_position()

    def get_random_action(self) -> ((float, float, float, float, float,), int):
        """Get a random action from all actions.

        Returns:
            tuple: A random action and its unique identifier.
        """
        x = np.random.randint(len(self.actions_dict))
        return self.actions_dict[x], x

    def get_current_qs(self) -> list[float]:
        """Get the q values of the current state.

        Returns:
            list: List of Q-values for the current state.
        """
        #print(f"self.current_voxel: {self.current_voxel}")
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.move_along_q_in_section][self.voxels_index_dict[self.move_along_q_in_section][self.current_voxel]]

    def get_current_q(self, action: int) -> float:
        """Get the Q value for the current state and a specific action.

        Args:
            action (int): Action index for the action dictionary.

        Returns:
            float: Q-value for the specified action at the current state.
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.move_along_q_in_section][self.voxels_index_dict[self.move_along_q_in_section][self.current_voxel]][action]

    def get_last_q(self, action: int) -> float:
        """Get the Q value for the current state and a specific action.

        Args:
            action (int): Action index for the action dictionary.

        Returns:
            float: Q-value for the specified action at the last state.
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[0][self.voxels_index_dict[0][self.last_voxel]][action]

    def get_last_n_q(self, action: int) -> float:
        """Get the Q values of n states ago, with n = 5.

        Args:
            action (int): Action index for the action dictionary.

        Returns:
            float: Q-value for the specified action 'n' steps ago.
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[0][self.voxels_index_dict[0][self.last_n_voxel[0]]][action]

    def set_current_q(self, action: int, q: float) -> None:
        """Set a Q value for the current state.

        Args:
            action (int): Action index for the action dictionary.
            q (float): New Q-value to be set.

        Returns:
            None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[0][self.voxels_index_dict[0][self.current_voxel]][action] = q

    def set_last_q(self, action: int, q: float) -> None:
        """Set a Q value for the state before the current state.

        Args:
            action (int): Action index for the action dictionary.
            q (float): New Q-value to be set.

        Returns:
            None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[0][self.voxels_index_dict[0][self.last_voxel]][action] = q


    def set_last_n_q(self, action: int, q: float) -> None:
        """Set a q value for n states before the current state, with n = 5.

        Args:
            action (int): Action index for the action dictionary.
            q (float): New Q-value to be set.

        Returns:
            None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[0][self.voxels_index_dict[0][self.last_n_voxel[0]]][action] = q


    def get_action_dict(self) -> dict:
        """Get the dict containing all actions (action_number : action).

        Returns:
            dict: Dictionary with all possible actions.
        """
        return self.actions_dict

    def get_action_from_dict(self, action: int) -> (float, float, float, float, float,):
        """Get action from the actions dict.

        Args:
            action (int): Action index for the action dictionary.

        Returns:
            tuple: The action corresponding to the given index.
        """
        return self.actions_dict[action]

    def get_inverse_action_dict(self) -> dict:
        """Get the inverse dict containing all actions (action : action_number).

        Returns:
            dict: Inverse dictionary containing all actions.
        """
        return self.inv_actions_dict

    def do_move(self, action: int) -> ((int, int, int), int, bool):
        """Move the robot based on the action.

        Args:
            action (int): Action index for the action dictionary.

        Returns:
            tuple: New TCP coordinates, reward, and a boolean indicating if it's a winning move.
        """
        # Assume no win for now, set to True when in winning voxels
        win = False
        # calculate the new joint angles
        new_angles = self.rob.get_current_joint_config() + self.actions_dict[action]
        # Move robot into new position
        self.set_joint_angles_rad(new_angles)

        # Check for boundaries and reset if neccessary
        if self.__check_in_voxels() is False:
            # Go back to starting position
            self.out_of_bounds_counter += 1
            self.set_joint_angles_rad(self.starting_angles, set_last_voxel=False)
            # High punishment for going out of bounds!
            reward = self.reward_out_of_bounds
        else:
            # Get the normal reward when in bounds
            reward = self.__get_reward()
        # Check for win
        if self.__check_win() is True:
            win = True
            reward = self.reward_win

        # Forward kinematics for TCP coordinate calculation
        tcp_matrix = self.rob.fkine()
        # TCP Coordinates as (x, y, z)
        tcp_coordinates = (tcp_matrix[0, 3], tcp_matrix[1, 3], tcp_matrix[2, 3])
        return tcp_coordinates, reward, win

    def get_tcp(self) -> (float, float, float):
        """Retrieve the current position of the robotic arm's end effector (Tool Center Point, TCP).

        This method calculates the current position of the arm's end effector using forward kinematics.
        The position is given in terms of Cartesian coordinates (x, y, z) in the arm's workspace.

        Returns:
            tuple: Coordinates of the end effector.
        """
        # Forward kinematics for TCP coordinate calculation
        tcp_matrix = self.rob.fkine()
        # TCP Coordinates as (x, y, z)
        tcp_coordinates = (tcp_matrix[0, 3], tcp_matrix[1, 3], tcp_matrix[2, 3])
        return tcp_coordinates

    def animate_move_along_q_values(self, draw_path=False, draw_voxels=False, zoom_path=False, fps=20, max_steps=2000):
        """Move the robot along the learned Q values and animate it.

        Animates the robot arm's movement based on the highest Q values from the learning process. The animation
        will stop if the robot runs out of bounds, completes its path, or reaches the maximum number of steps 
        to prevent infinite loops. The method provides options to visualize the path, voxels, and to zoom in on the path.

        Will stop when running out of bounds.
        Needs to be called last.

        Args:
            draw_path (bool): If True, the path the robot is supposed to learn is drawn.
            draw_voxels (bool): If True, the voxels are drawn.
            zoom_path (bool): If True, the drawing is zoomed in on the path.
            fps (int): Frames per second for the animation.
            max_steps (int): Maximum number of steps in the animation to prevent infinite loops.

        Returns:
            None
        """
        # Reset robot to starting position
        self.reset()

        # Do moves along largest Q values and save them
        done = False
        i = 0
        while not done:
            # Get the current Qs and search for the highest Q
            action = np.argmax(self.get_current_qs())
            # Move the direction with the highest Q
            new_angles = self.rob.get_current_joint_config() + self.actions_dict[action]
            # Move robot into new position
            self.set_joint_angles_rad(new_angles, save=True)
            # Check for boundaries, check for win, check if max steps are reached max steps
            in_voxels = self.__check_in_voxels()
            in_win = self.__check_win()
            if (not in_voxels) or (in_win) or (i > max_steps):
                if not in_voxels: print("Animation out of bounds!")
                if i > max_steps: print("Possible infinite loop!")
                if in_win is True:
                    # Check if we are in the last winning voxels, so at the end of the helix 
                    if len(self.winning_voxels)-1 == self.move_along_q_in_section:
                        done = True
                    else:
                        self.move_along_q_in_section += 1
            i += 1

        self.move_along_q_in_section = 0

        # Animate with static FPS of 20, which is not reached at all as it is too fast.
        self.animate(draw_path=draw_path, draw_voxels=draw_voxels, zoom_path=zoom_path, fps=20)

    def show(self, draw_path=False, draw_voxels=False, zoom_path=False, draw_q_path=False) -> None:
        """Open window and draw robot arm.

        Args:
            draw_path (bool): If True, the target path is drawn.
            draw_voxels (bool): If True, voxels are drawn.
            zoom_path (bool): If True, zooms in on the path in the visualization.
            draw_q_path (bool): If True, draws the path taken based on Q-values.

        Returns:
            None
        """
        if draw_q_path is True:
            self.get_finishing_angles_rad()

        if zoom_path is False:
            ax = self.env.plot(show=False)
        else:
            ax = self.env.plot(show=False,
                               xlim=[np.min(self.path[0])-10, np.max(self.path[0])+10],
                               ylim=[np.min(self.path[1])-10, np.max(self.path[1])+10],
                               zlim=[np.min(self.path[2])-10, np.max(self.path[2])+10],
                               )

        if draw_path is True:
            ax.plot(self.path[0], self.path[1], self.path[2], color='red')

        if draw_q_path is True:
            ax.plot(self.q_path[0], self.q_path[1], self.q_path[2], color='green')

        if draw_voxels is True:
            x = []
            y = []
            z = []
            for voxel in self.voxels:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2, c='aqua')

            x = []
            y = []
            z = []
            for voxel in self.winning_voxels[0]:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2.5, color='red')

        plt.show()

    def animate(self, draw_path=False, draw_voxels=False, zoom_path=False, fps=20, save_path=None) -> None:
        """Animate robot.

        Needs to be called last.

        Args:
            draw_path (bool): If True, the path the robot is supposed to learn is drawn.
            draw_voxels (bool): If True, voxels are drawn.
            zoom_path (bool): If True, zooms in on the path in the visualization.
            fps (int): Frames per second for the animation.
            save_path (str, optional): (Currently BROKEN!) Path to save the animation video. None means no saving.

        Returns:
            None
        """
        if zoom_path is False:
            if draw_path is False:
                if draw_voxels is False:
                    self.env.animate(fps=fps, save_path=save_path)
                else:
                    self.env.animate(voxels=self.voxels, winning_voxels=self.winning_voxels[self.section_to_stitch-1],
                                     fps=fps, save_path=save_path)
            else:
                if draw_voxels is False:
                    self.env.animate(path=self.path)
                else:
                    self.env.animate(path=self.path, voxels=self.voxels,
                                     winning_voxels=self.winning_voxels[self.section_to_stitch-1],
                                     fps=fps, save_path=save_path)
        else:
            if draw_path is False:
                if draw_voxels is False:
                    self.env.animate(xlim=[np.min(self.path[0])-10, np.max(self.path[0])+10],
                                     ylim=[np.min(self.path[1])-10, np.max(self.path[1])+10],
                                     zlim=[np.min(self.path[2])-10, np.max(self.path[2])+10],
                                     fps=fps, save_path=save_path)
                else:
                    self.env.animate(xlim=[np.min(self.path[0])-10, np.max(self.path[0])+10],
                                     ylim=[np.min(self.path[1])-10, np.max(self.path[1])+10],
                                     zlim=[np.min(self.path[2])-10, np.max(self.path[2])+10],
                                     voxels=self.voxels, winning_voxels=self.winning_voxels[self.section_to_stitch-1],
                                     fps=fps, save_path=save_path)
            else:
                if draw_voxels is False:
                    self.env.animate(xlim=[np.min(self.path[0])-10, np.max(self.path[0])+10],
                                     ylim=[np.min(self.path[1])-10, np.max(self.path[1])+10],
                                     zlim=[np.min(self.path[2])-10, np.max(self.path[2])+10],
                                     path=self.path, fps=fps, save_path=save_path)
                else:
                    self.env.animate(xlim=[np.min(self.path[0])-10, np.max(self.path[0])+10],
                                     ylim=[np.min(self.path[1])-10, np.max(self.path[1])+10],
                                     zlim=[np.min(self.path[2])-10, np.max(self.path[2])+10],
                                     path=self.path, voxels=self.voxels,
                                     winning_voxels=self.winning_voxels[self.section_to_stitch-1],
                                     fps=fps, save_path=save_path)

    def save_learned_to_file(self):
        """Save the learned Q-values, winning voxels, index dictionary, and rewards to files.

        This method serializes and writes the Q-values, winning voxels, voxel index dictionary, and
        rewards used by the robotic arm in the learning process to separate files. The files are
        named based on the number of axes of the robotic arm and the specific helix section the arm
        is learning. This enables the persistence of learning data for later use or analysis.

        Returns:
            None

        Note: The files are saved in the directory 'learned_values_[num_axis]_axis', with the helix section
        number appended to the filenames. Ensure that this directory exists or handle exceptions
        for potential 'FileNotFoundError'.
        """
        # Write Qs to file
        np.save(f"learned_values_{self.num_axis}_axis/Q_values_section_{self.helix_section}.npy", self.Q[0])
        # Write Winning Voxels to file
        np.save(f"learned_values_{self.num_axis}_axis/Winning_voxels_{self.helix_section}.npy", self.winning_voxels[0])
        # Write index dict to file
        with open(f"learned_values_{self.num_axis}_axis/Index_dict_{self.helix_section}.json", 'w') as json_file:
            json_file.write(ujson.dumps(self.voxels_index_dict[0]))
        # Write rewards used to file
        np.save(f"learned_values_{self.num_axis}_axis/Rewards_{self.helix_section}.npy", self.rewards)

    def load_learned_from_file(self):
        """Load previously saved Q-values, winning voxels, index dictionary, and rewards from files.

        This method attempts to load the learning data of the robotic arm from files saved earlier by
        the `save_learned_to_file` method. It reads the Q-values, winning voxels, voxel index dictionary,
        and rewards and sets them to the respective attributes of the class. The files are expected to be
        named based on the number of axes of the robotic arm and the specific helix section the arm has
        learned.

        Returns:
            None

        Note: The method handles the absence of files by catching exceptions and returning early.
        This means that if any of the expected files are not found, the method will not update the
        corresponding attributes and will exit without error. Ensure that the files are located in the
        'learned_values_[num_axis]_axis' directory.
        """
        # Load Qs from file
        try:
            self.Q[0] = np.load(f"learned_values_{self.num_axis}_axis/Q_values_section_{self.helix_section}.npy")
        except:
            #print("No file, not loading")
            return
        # Load Winning Voxels to file
        self.winning_voxels[0] = []
        try:
            loaded_winning_voxels = np.load(f"learned_values_{self.num_axis}_axis/Winning_voxels_{self.helix_section}.npy")
        except:
            #print("No file, not loading")
            return
        # Convert arrays to tuples
        for i, winning_voxel_arr in enumerate(loaded_winning_voxels):
            self.winning_voxels[0].append(tuple(winning_voxel_arr))
        # Load index dict to file
        try:
            with open(f"learned_values_{self.num_axis}_axis/Index_dict_{self.helix_section}.json", 'r') as json_file:
                loaded_dict = ujson.load(json_file)
        except:
            #print("No file, not loading")
            return
        # Convert strings to tuples
        self.voxels_index_dict[0] = {eval(key): value for key, value in loaded_dict.items()}
        # Load Rewards from file
        try:
            self.rewards = np.load(f"learned_values_{self.num_axis}_axis/Rewards_{self.helix_section}.npy")
        except:
            #print("No file, not loading")
            return

    def stitch_from_file(self):
        """Stitch the next segment of learned data from files into the robot's learning attributes.

        This method is used to load additional segments of learned data (Q-values, winning voxels, and
        voxel index dictionaries) from files and append them to the existing data structures of the robot.
        This is useful for constructing a comprehensive learning model from multiple segments of learned data.
        The method handles the loading of data and ensures that the robotic arm's learning attributes are
        updated accordingly.

        Returns:
            None

        Note: The method expects files to be located in the 'learned_values_[num_axis]_axis' directory
        and named based on the current section to stitch. It handles the absence of files
        by catching exceptions and not updating the attributes in such cases. Also, ensure that the
        method is called in the correct sequence with the appropriate section to be stitched.
        """
        print(f"stitching section: {self.section_to_stitch}")
        #print("Loading Qs and Voxels from file and stitching them to the robots Qs and voxels")
        # Load Qs from file
        try:
            additional_Qs = np.load(f"learned_values_{self.num_axis}_axis/Q_values_section_{self.section_to_stitch}.npy")
        except:
            print("No file, not loading")
            return
        # Load Winning Voxels from file and overwrite the current winning voxels
        new_winning_voxels = []
        try:
            loaded_winning_voxels = np.load(f"learned_values_{self.num_axis}_axis/Winning_voxels_{self.section_to_stitch}.npy")
        except:
            print("No file, not loading")
            return
        # Convert arrays to tuples
        for i, winning_voxel_arr in enumerate(loaded_winning_voxels):
            new_winning_voxels.append(tuple(winning_voxel_arr))
        # Load index dict to file
        try:
            with open(f"learned_values_{self.num_axis}_axis/Index_dict_{self.section_to_stitch}.json", 'r') as json_file:
                loaded_dict = ujson.load(json_file)
        except:
            print("No file, not loading")
            return
        # Convert strings to tuples
        additional_voxels_index_dict = {eval(key): value for key, value in loaded_dict.items()}
        # Load Rewards from file
        try:
            additional_rewards = np.load(f"learned_values_{self.num_axis}_axis/Rewards_{self.section_to_stitch}.npy")
        except:
            print("No file, not loading")
            return

        self.voxels_index_dict.append(additional_voxels_index_dict)
        self.winning_voxels.append(new_winning_voxels)
        self.Q.append(additional_Qs)

        for voxel in self.voxels_index_dict[self.section_to_stitch]:
            self.voxels.append(voxel)

        # Reset robot arm to starting position
        self.reset()

        self.section_to_stitch += 1

    def get_finishing_angles_rad(self, max_steps=2000) -> (str, (float, float, float, float, float,)):
        """Determine the finishing angles of the robot arm based on the learned movements.

        This method navigates the robot arm through its environment based on the highest Q-values obtained
        from the learning process. It stops navigating either when it goes out of bounds, completes the task,
        or reaches a predefined maximum number of steps. The method provides the final status and the joint angles
        of the robot arm in radians.

        Args:
            max_steps (int): Maximum number of steps to perform for determining finishing angles.

        Returns:
            tuple: A string indicating the final status ('Success', 'Out of bounds', 'Infinite Loop')
                   and the final joint angles in radians.

        Note: The method resets the robot arm to its starting position before beginning the navigation process.
        """
        # Reset robot to starting position
        self.reset()

        # Do moves along largest Q values
        done = False
        return_string = "Success"
        i = 0
        self.q_path = [[self.starting_pos[0]], [self.starting_pos[1]], [self.starting_pos[2]]]
        while not done:
            # Get the current Qs and search for the highest Q
            action = np.argmax(self.get_current_qs())
            # Move the direction with the highest Q
            new_angles = self.rob.get_current_joint_config() + self.actions_dict[action]
            # Move robot into new position
            self.set_joint_angles_rad(new_angles)
            tcp = self.get_tcp()
            self.q_path[0].append(tcp[0])
            self.q_path[1].append(tcp[1])
            self.q_path[2].append(tcp[2])
            # Check for boundaries, check for win, check if max steps are reached max steps
            in_voxels = self.__check_in_voxels()
            in_win = self.__check_win()
            if (not in_voxels) or (in_win) or (i > max_steps):
                if not in_voxels: return_string = "Out of bounds"
                if i > max_steps: return_string = "Infinite Loop"
                done = True
                if in_win is True:
                    # Check if we are in the last winning voxels, so at the end of the helix
                    if len(self.winning_voxels)-1 != self.move_along_q_in_section:
                        done = False
                        self.move_along_q_in_section += 1
            i += 1

        self.move_along_q_in_section = 0

        return return_string, tuple(self.get_joint_angles_rad())

    def calc_mse(self, support_points=1000):
        """Calculate the Mean Squared Error (MSE) between the desired path and the path taken by the robot.

        This method first interpolates both the desired path and the path taken by the robot (q_path) to a
        specified number of support points. It then calculates the Mean Squared Error (MSE) separately for
        each dimension (x, y, z) and combines these to provide a single MSE value representing the average
        deviation of the robot's path from the desired path.

        Args:
            support_points (int): Number of points for path interpolation.

        Returns:
            float: The Mean Squared Error between the interpolated desired path and the robot's path.
        """
        self.get_finishing_angles_rad()
        path_x, path_y, path_z = self.__interpolate_path(self.path[0], self.path[1], self.path[2], support_points)
        q_path_x, q_path_y, q_path_z = self.__interpolate_path(self.q_path[0], self.q_path[1], self.q_path[2], support_points)

        # Calculate MSE after converting paths to arrays
        mse_x = self.__mean_squared_error(path_x, q_path_x)
        mse_y = self.__mean_squared_error(path_y, q_path_y)
        mse_z = self.__mean_squared_error(path_z, q_path_z)

        # Combine the MSE for each dimension
        total_mse = (mse_x + mse_y + mse_z) / 3

        return total_mse

    def set_starting_angles_rad(self, angles=(float, float, float, float, float,)):
        """Set the starting joint angles of the robotic arm in radians.

        This method configures the starting joint angles of the robotic arm. The angles are specified in radians
        and are used to set the initial configuration (q0) of the robot. After setting the angles, the robot arm
        is reset to this starting position.

        Args:
            angles (tuple): A tuple of joint angles in radians.

        Returns:
            None
        """
        # Convert angles to numpy array
        angles_array = np.asarray(angles)

        # Overwrite Q0 in the robot arm
        self.rob.q0 = angles_array

        # Reset robot arm to starting position
        self.reset()
