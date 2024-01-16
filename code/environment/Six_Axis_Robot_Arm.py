"""Robot environment for the task."""

import numpy as np
import matplotlib.pyplot as plt
# import mplot3d as plt3d
# maddux for robot visualization
from maddux.robots.link import Link
from maddux.robots.arm import Arm
from maddux.environment import Environment
import time
from path import Path
from path_short import Path_Short
import itertools
import ujson
import math

# suppress scientific notation
np.set_printoptions(suppress=True)


class Six_Axis_Robot_Arm:
    """Class simulating the robot arm."""

    def __init__(self, starting_pos: (float, float, float, float, float, float) = (-500, 0, 0),
                 section_length=1, helix_section=0,
                 voxels=None, winning_voxels=None,
                 voxel_volume=1, stitch_section=1,
                 checkpoints=False, num_checkpoints=64) -> None:
        """Initialize robot arm.

        :param initial_angles: Tuple with the initial angles of the robot joints in degrees
        :type initial_angles: (float, float, float)

        :param path: Coordinates of path to draw
        :type path: list of touples of ints

        :return: None
        """
        # Create path for the robot
        #path = Path(helix_start=starting_pos, max_distance=2)
        self.helix_section = helix_section
        self.section_length = section_length*1.018
        helix_section = helix_section * section_length
        # make the section a little longer so each section overlaps a litle
        path = Path(helix_start=starting_pos, max_distance=voxel_volume,
                    generate_percentage_of_helix=self.section_length, generate_start=helix_section)
        self.voxels, self.winning_voxels, self.rewards = path.get_helix_voxels()
        self.voxel_size = 1
        self.path = path.get_helix_data()
        amount_voxels = len(self.voxels)

        # Create hashtable of voxels with a unique index for each voxel
        self.voxels_index_dict = {value: index for index, value in enumerate(self.voxels)}

        # Create a series of links (each link has one joint)
        # (theta, offset, length, twist, q_lim=None)
        L1 = Link(0, 151.85, 0, 1.570796327, link_size=5)
        L2 = Link(0, 0, -243.55, 0, link_size=4)
        L3 = Link(0, 0, -213.2, 0, link_size=4)
        L4 = Link(0, 131.05, 0, 1.570796327, link_size=3)
        L5 = Link(0, 85.35, 0, -1.570796327, link_size=2)
        L6 = Link(0, 92.1, 0, 0, link_size=0.1)
        links = np.array([L1, L2, L3, L4, L5, L6])

        # Caculate starting angles from starting position and
        # convert degrees of starting position to radiants

        # Initial arm angles
        q0 = np.array((0, 0, 0, 0, 0, 0))

        # Create arm
        self.rob = Arm(links, q0, '1-link')

        # Do inverse kinematics for the starting position and
        # create a new arm and set it to the start of the helix
        self.starting_angles = self.rob.ikine((self.path[0][0], self.path[1][0], self.path[2][0]))
        # do mod 2pi to starting angles to not get crazy large angles
        self.starting_angles = np.array([angle % (2*np.pi) for angle in self.starting_angles])
        self.initial_angles = self.starting_angles
        # Create arm
        self.rob = Arm(links, self.starting_angles, '1-link')

        self.env = Environment(dimensions=[1500.0, 1500.0, 1500.0],
                               robot=self.rob)

        # Create all possible actions
        # Define possible actions for each joint in deg
        # For now 1 degree per action, as the robot will take forever otherwise
        joint_actions_deg = [-0.1, 0, 0.1]
        joint_actions_rad = np.array([self.__deg_to_rad(action) for action in joint_actions_deg])

        # Generate all possible action combinations for the 3 joints
        action_combinations = list(itertools.product(joint_actions_rad, repeat=6))
        total_amount_actions = len(action_combinations)

        # Create a dictionary to map each combination to a unique integer
        self.actions_dict = {i: action for i, action in enumerate(action_combinations)}
        self.inv_actions_dict = {v: k for k, v in self.actions_dict.items()}

        # Create Q
        self.Q = -np.random.rand(amount_voxels, total_amount_actions)

        # Set ending positions to zero in Q
        for winning_voxel in self.winning_voxels:
            self.Q[self.voxels_index_dict[winning_voxel]] = np.zeros(total_amount_actions)

        # Create variable for voxel of current TCP position, so it only needs to be calculated
        # when the TCP is changed
        self.current_voxel = self.__get_tcp_voxel_position()
        # Save last voxel to be able to set last Q
        self.last_voxel = None

        # Save last voxel to be able to set last Q
        self.n = 5
        self.last_n_voxel = []

        # Init out of bounds counter
        self.out_of_bounds_counter = 0

        # Reset robot arm to starting position
        self.reset()

        # rewards
        self.reward_out_of_bounds = -500
        self.reward_going_backwards = -250
        self.reward_step = -1
        self.reward_win = 100
        self.reward_section_win = 0

        # Create Q
        self.Q = -np.random.rand(amount_voxels, total_amount_actions)

        # Set ending positions to zero in Q
        for winning_voxel in self.winning_voxels:
            self.Q[self.voxels_index_dict[winning_voxel]] = np.zeros(total_amount_actions)

        self.current_checkpoint = 0
        self.checkpoints = checkpoints
        self.num_checkpoints = num_checkpoints
        self.reward_needed_for_next_checkpoint = -100+((100)/self.num_checkpoints)

        # Remember last reward to punish going backwards
        self.last_reward = -1000000


    def __deg_to_rad(self, deg: float) -> float:
        """Convert degree to radiants.

        :param deg: Degrees to convert to radiants
        :type deg: float

        :return: Radiants
        :rtype: float
        """
        return deg*np.pi/180

    def __rad_to_deg(self, rad: float) -> float:
        """Convert degree to radiants.

        :param rad: Ratiants to convert to radiants
        :type rad: float

        :return: Degrees
        :rtype: float
        """
        return rad*180/np.pi

    def __limit_angle(self, angle: float) -> float:
        """Limit angle (in rad) to +-pi (+-180°).

        :param rad: Angle in radiants
        :type rad: float

        :return: Limited angle in radiants
        :rtype: float
        """
        if angle > np.pi/2:
            return np.pi
        if angle < -np.pi:
            return -np.pi
        return angle

    def __limit_angles(self, angles: (float, float, float, float, float, float)
                       ) -> (float, float, float, float, float, float):
        """Limit angles of the robot (in rad) to +-pi (+-180°).

        :param rad: Angles in radiants
        :type rad: float

        :return: Limited angles in radiants
        :rtype: float
        """
        ## Substract starting angles to start in the middle
        #angles_relation_to_start = angles - self.starting_angles
        ## Limit angles
        #limited_angles = np.array([self.__limit_angle(angle) for angle in angles_relation_to_start])
        ## Add to starting angles
        #return limited_angles + self.starting_angles

        # no limit for now.
        return angles

    def __get_tcp_voxel_position(self) -> (int, int, int):
        """Get voxel the TCP is in.

        :return: Tuple containing the x, y and z coordinates of the voxel
        :rtype: (int, int, int)
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
        return self.current_voxel in self.winning_voxels

    def __check_in_voxels(self) -> bool:
        """Check if the current position is in a voxel.

        :return: Bool indicating if the TCP is in a voxel
        :rtype: bool
        """
        return self.current_voxel in self.voxels_index_dict

    def __get_reward(self) -> int:
        """Get the reward for the current position.

        :return: Reward
        :rtype: int
        """
        # rewards go from -100 to 0.
        reward = self.rewards[self.voxels_index_dict[self.current_voxel]]

        # Punish moving backwards
        if self.last_reward > reward:
            reward = self.reward_going_backwards

        # In case a chekpoint is passed
        if self.checkpoints == True:
            if reward >= self.reward_needed_for_next_checkpoint:
                self.current_checkpoint += 1
                self.reward_needed_for_next_checkpoint = -100+((100*self.current_checkpoint)/self.num_checkpoints)
                self.starting_angles = self.get_joint_angles_rad()
                if self.current_checkpoint != 1:
                    reward = reward + 2

        return reward

    def get_joint_angles(self) -> (float, float, float, float, float, float):
        """Return current joint angles.

        :return: Tuple with the current angles of the robot joints in degrees
        :rtype: (float, float, float, float, float, float)

        :return: None
        """
        return np.array([self.__rad_to_deg(angle) for angle in self.rob.get_current_joint_config()])

    def get_joint_angles_rad(self) -> (float, float, float, float, float, float):
        """Return current joint angles.

        :return: Tuple with the current angles of the robot joints in degrees
        :rtype: (float, float, float, float, float, float)

        :return: None
        """
        return self.rob.get_current_joint_config()

    def set_joint_angles_degrees(self, angles: (float, float, float, float, float, float), save=False) -> None:
        """Set joint angles.

        :param angles: Tuple with the angles for the robot joints in degrees
        :type angles: (float, float, float, float, float, float)

        :return: None
        """
        # Convert degrees of angles to radiants
        angles_rad = np.array([self.__deg_to_rad(angle) for angle in angles])
        self.set_joint_angles_rad(angles_rad, save=save)

    def set_joint_angles_rad(self, angles: (float, float, float, float, float, float), save=False, set_last_voxel=True) -> None:
        """Set joint angles.

        :param angles: Tuple with the angles for the robot joints in radiants
        :type angles: (float, float, float, float, float, float)

        :return: None
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

        :return: None
        """
        self.rob.reset(save=False)
        self.out_of_bounds_counter = 0
        self.current_voxel = self.__get_tcp_voxel_position()
        self.current_checkpoint = 0
        self.starting_angles = self.initial_angles

    def get_random_action(self) -> ((float, float, float, float, float, float), int):
        """Get a random action from all actions.

        :return: Tuple containing the action and the unique integer representing the action in the
                 actions dict
        :rtype: ((float, float, float, float, float, float), int)
        """
        x = np.random.randint(len(self.actions_dict))
        return self.actions_dict[x], x

    def get_current_qs(self) -> list[float]:
        """Get the q values for the current state.

        :return: List of q values
        :rtype: list of float
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.voxels_index_dict[self.current_voxel]]

    def get_current_q(self, action: int) -> float:
        """Get the q value for the current state and a specific action.

        :param action: Action index for the action dict
        :type action: int

        :return: Q value
        :rtype: float
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.voxels_index_dict[self.current_voxel]][action]

    def get_last_q(self, action: int) -> float:
        """Get the q value for the current state and a specific action.

        :param action: Action index for the action dict
        :type action: int

        :return: Q value
        :rtype: float
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.voxels_index_dict[self.last_voxel]][action]

    def get_last_n_q(self, action: int) -> float:
        """Get the q value for the current state and a specific action.

        :param action: Action index for the action dict
        :type action: int

        :return: Q value
        :rtype: float
        """
        # Return the value of Q at the index of the current voxel in the index dict
        return self.Q[self.voxels_index_dict[self.last_n_voxel[0]]][action]


    def set_current_q(self, action: int, q: float) -> None:
        """Set a q value for the current state.

        :param action: Action index for the action dict
        :type action: int

        :param new_q: new Q value to set
        :type new_q: float

        :return: None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[self.voxels_index_dict[self.current_voxel]][action] = q

    def set_last_q(self, action: int, q: float) -> None:
        """Set a q value for the state before the current state.

        :param action: Action index for the action dict
        :type action: int

        :param new_q: new Q value to set
        :type new_q: float

        :return: None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[self.voxels_index_dict[self.last_voxel]][action] = q


    def set_last_n_q(self, action: int, q: float) -> None:
        """Set a q value for the state before the current state.

        :param action: Action index for the action dict
        :type action: int

        :param new_q: new Q value to set
        :type new_q: float

        :return: None
        """
        # Set the value of Q at the index of the current voxel in the index dict
        self.Q[self.voxels_index_dict[self.last_n_voxel[0]]][action] = q


    def get_action_dict(self) -> dict:
        """Get the dict containing all actions (action_number : action).

        :return: Dict with all actions
        :rtype: dict
        """
        return self.actions_dict

    def get_action_from_dict(self, action: int) -> (float, float, float, float, float, float):
        """Get action from the actions dict.

        :param action: Action index for the action dict
        :type action: int

        :return: Tuple with action
        :rtype: (float, float, float, float, float)
        """
        return self.actions_dict[action]

    def get_inverse_action_dict(self) -> dict:
        """Get the inverse dict containing all actions (action : action_number).

        :return: Dict with all actions
        :rtype: dict
        """
        return self.inv_actions_dict

    def do_move(self, action: int) -> ((int, int, int), int, bool):
        """Move the robot based on the action.

        :param action: Action index for the action dict
                       Action dict can be obtained with get_action_dict()
                       Inverse action dict can be obtained with get_inverse_action_dict()
        :type action: int

        :return: tuple containing the new TCP (x, y, z), reward and win indication
        :rtype: ((int, int, int), int, bool)
        """
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
            #self.last_voxel = self.current_voxel
            # High punishment for going out of bounds!
            reward = self.reward_out_of_bounds
        else:
            reward = self.__get_reward()
        # Check for win
        if self.__check_win() is True:
            win = True
            reward = 0

        # Forward kinematics for TCP coordinate calculation
        tcp_matrix = self.rob.fkine()
        # TCP Coordinates as (x, y, z)
        tcp_coordinates = (tcp_matrix[0, 3], tcp_matrix[1, 3], tcp_matrix[2, 3])
        return tcp_coordinates, reward, win

    def get_tcp(self) -> ((int, int, int), int, bool):
        # Forward kinematics for TCP coordinate calculation
        tcp_matrix = self.rob.fkine()
        # TCP Coordinates as (x, y, z)
        tcp_coordinates = (tcp_matrix[0, 3], tcp_matrix[1, 3], tcp_matrix[2, 3])
        return tcp_coordinates

    def animate_move_along_q_values(self, draw_path=False, draw_voxels=False, zoom_path=False, fps=20, max_steps=2000, inverse_zoom_factor=1):
        """Move the robot along the learned Q values and animate it.

        Will stop when running out of bounds.
        Needs to be called last.

        :param draw_path: Draw the path the robot is supposed to learn
        :type draw_path: bool

        :param draw_voxels: Draw the voxels
        :type draw_voxels: bool

        :param zoom_path: Fit drawing to the path
        :type zoom_path: bool

        :param fps: Fps of the animation
        :type fps: int

        :param max_steps: Maximum numbers of steps to animate
        :type max_steps: int
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
                done = True
            i += 1

        # Animate
        self.animate(draw_path=draw_path, draw_voxels=draw_voxels, zoom_path=zoom_path, fps=20, inverse_zoom_factor=inverse_zoom_factor)

    def show(self, draw_path=False, draw_voxels=False, zoom_path=False) -> None:
        """Open window and draw robot arm.

        :param draw_path: Draw the path the robot is supposed to learn
        :type draw_path: bool

        :param draw_voxels: Draw the voxels
        :type draw_voxels: bool

        :param zoom_path: Fit drawing to the path
        :type zoom_path: bool

        :return: None
        """
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

        if draw_voxels is True:
            x = []
            y = []
            z = []
            for voxel in self.voxels:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2, cmap=plt.get_cmap('hot'), c=self.rewards)

            x = []
            y = []
            z = []
            for voxel in self.winning_voxels:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2.5, color='red')

        plt.show()

    def animate(self, draw_path=False, draw_voxels=False, zoom_path=False, fps=20, save_path=None, inverse_zoom_factor=1) -> None:
        """Animate robot.

        Needs to be called last.

        :param draw_path: Draw the path the robot is supposed to learn
        :type draw_path: bool

        :param draw_voxels: Draw the voxels
        :type draw_voxels: bool

        :param zoom_path: Fit drawing to the path
        :type zoom_path: bool

        :param fps: Fps of the animation
        :type fps: int

        :param save_path: BROKEN ATM!
                          Save animation as video in path if not None
                          FFMPEG needed (sudo apt install ffmpeg)
                          name of the animation and format must be specified
                          (e.g.: animation.mp4)
        :type save_path: str

        :return: None
        """
        if zoom_path is False:
            if draw_path is False:
                if draw_voxels is False:
                    self.env.animate(fps=fps, save_path=save_path)
                else:
                    self.env.animate(voxels=self.voxels, winning_voxels=self.winning_voxels,
                                     fps=fps, save_path=save_path)
            else:
                if draw_voxels is False:
                    self.env.animate(path=self.path)
                else:
                    self.env.animate(path=self.path, voxels=self.voxels,
                                     winning_voxels=self.winning_voxels,
                                     fps=fps, save_path=save_path)
        else:
            if draw_path is False:
                if draw_voxels is False:
                    self.env.animate(xlim=[np.min(self.path[0])-10*inverse_zoom_factor, np.max(self.path[0])+10*inverse_zoom_factor],
                                     ylim=[np.min(self.path[1])-10*inverse_zoom_factor, np.max(self.path[1])+10],
                                     zlim=[np.min(self.path[2])-10*inverse_zoom_factor, np.max(self.path[2])+10*inverse_zoom_factor],
                                     fps=fps, save_path=save_path)
                else:
                    self.env.animate(xlim=[np.min(self.path[0])-10*inverse_zoom_factor, np.max(self.path[0])+10*inverse_zoom_factor],
                                     ylim=[np.min(self.path[1])-10*inverse_zoom_factor, np.max(self.path[1])+10*inverse_zoom_factor],
                                     zlim=[np.min(self.path[2])-10*inverse_zoom_factor, np.max(self.path[2])+10*inverse_zoom_factor],
                                     voxels=self.voxels, winning_voxels=self.winning_voxels,
                                     fps=fps, save_path=save_path)
            else:
                if draw_voxels is False:
                    self.env.animate(xlim=[np.min(self.path[0])-10*inverse_zoom_factor, np.max(self.path[0])+10*inverse_zoom_factor],
                                     ylim=[np.min(self.path[1])-10*inverse_zoom_factor, np.max(self.path[1])+10*inverse_zoom_factor],
                                     zlim=[np.min(self.path[2])-10*inverse_zoom_factor, np.max(self.path[2])+10*inverse_zoom_factor],
                                     path=self.path, fps=fps, save_path=save_path)
                else:
                    self.env.animate(xlim=[np.min(self.path[0])-10*inverse_zoom_factor, np.max(self.path[0])+10*inverse_zoom_factor],
                                     ylim=[np.min(self.path[1])-10*inverse_zoom_factor, np.max(self.path[1])+10*inverse_zoom_factor],
                                     zlim=[np.min(self.path[2])-10*inverse_zoom_factor, np.max(self.path[2])+10*inverse_zoom_factor],
                                     path=self.path, voxels=self.voxels,
                                     winning_voxels=self.winning_voxels,
                                     fps=fps, save_path=save_path)

    def save_learned_to_file(self):
        # Write Qs to file
        np.save(f"Q_values_section_{self.helix_section}.npy", self.Q)
        # Write Winning Voxels to file
        np.save(f"Winning_voxels_{self.helix_section}.npy", self.winning_voxels)
        # Write index dict to file
        with open(f"Index_dict_{self.helix_section}.json", 'w') as json_file:
            json_file.write(ujson.dumps(self.voxels_index_dict))

    def load_learned_from_file(self):
        # Load Qs from file
        try:
            self.Q = np.load(f"Q_values_section_{self.helix_section}.npy")
        except:
            print("No file, not loading")
            return
        # Load Winning Voxels to file
        self.winning_voxels = []
        try:
            loaded_winning_voxels = np.load(f"Winning_voxels_{self.helix_section}.npy")
        except:
            print("No file, not loading")
            return
        # Convert arrays to tuples
        for i, winning_voxel_arr in enumerate(loaded_winning_voxels):
            self.winning_voxels.append(tuple(winning_voxel_arr))
        # Load index dict to file
        try:
            with open(f"Index_dict_{self.helix_section}.json", 'r') as json_file:
                loaded_dict = ujson.load(json_file)
        except:
            print("No file, not loading")
            return
        # Convert strings to tuples
        self.voxels_index_dict = {eval(key): value for key, value in loaded_dict.items()}

    def get_finishing_angles_rad(self, max_steps=2000) -> (str, (float, float, float)):
        # Reset robot to starting position
        self.reset()

        # Do moves along largest Q values
        done = False
        return_string = "Success"
        i = 0
        while not done:
            # Get the current Qs and search for the highest Q
            action = np.argmax(self.get_current_qs())
            # Move the direction with the highest Q
            new_angles = self.rob.get_current_joint_config() + self.actions_dict[action]
            # Move robot into new position
            self.set_joint_angles_rad(new_angles)
            # Check for boundaries, check for win, check if max steps are reached max steps
            in_voxels = self.__check_in_voxels() is True
            in_win = self.__check_win() is True
            if (not in_voxels) or (in_win) or (i > max_steps):
                if not in_voxels: return_string = "Out of bounds"
                if i > max_steps: return_string = "Infinite Loop"
                done = True
            i += 1

        return return_string, tuple(self.get_joint_angles_rad())

    def set_starting_angles_rad(self, angles=(float, float, float)):

        # Convert angles to numpy array
        angles_array = np.asarray(angles)

        # Overwrite Q0 in the robot arm
        self.rob.q0 = angles_array

        # Reset robot arm to starting position
        self.reset()
