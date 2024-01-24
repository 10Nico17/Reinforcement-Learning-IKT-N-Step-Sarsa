"""Helix Path Generation for Robotic Navigation.

This module provides the functionality to create and manage a helix-shaped path, typically used
in robotic navigation and trajectory planning. The primary class, Path, generates a three-dimensional
helix path, calculates the coordinates of voxels along this path, and determines winning voxels based
on their proximity to the path. It supports various configurations of the helix, such as the start
position, voxel size, and the maximum distance for voxel generation from the path.

Classes:
    Path: Manages the generation and handling of a helix-shaped path.

The Path class offers methods to calculate specific points on the helix given a parameter, generate
voxels on or near the helix trajectory, and obtain the raw helix data. It is equipped to handle
different scales of the helix and provides functionalities for calculating the nearest distance from
a point to the helix and managing rewards associated with the voxels.

This module is particularly useful in robotic applications where path following and navigation in
three-dimensional space are essential, such as in robotic arms or autonomous drones navigating through
prescribed paths.
"""
import numpy as np
from math import cos, sin, pi, sqrt

class Path:
    """A class to represent and calculate the trajectory of a double helix-shaped path.

    This class is designed to generate and manage a helix-shaped path, considering the helix's start point,
    voxel size, and other helical parameters. It includes methods for calculating both the raw helix data
    and the voxels that lie along the helix within a specified distance.

    Attributes:
        helix_start (tuple): Starting coordinates of the helix.
        voxel_size (int): Size of each voxel in millimeters.
        max_distance (int): Maximum distance from the helix path to consider for voxel generation.
        par_space (int): Defines the parameter space of the helix.
        percentage_of_helix (float): Percentage of the helix to generate voxels on.
        start_of_voxels (float): Starting point on the helix for voxel generation.
        resolution (int): Resolution of helix generation.
        helix_scale (str): Scale of the helix ('dmm', 'mm', or 'cm').
        helix_factor (int): Factor to adjust the helix scale.

    Methods:
        x(t): Calculates the x-coordinate of the helix at a given parameter t.
        y(t): Calculates the y-coordinate of the helix at a given parameter t.
        z(t): Calculates the z-coordinate of the helix at a given parameter t.
        __calculate_nearest_distance(voxel, path): Calculates the nearest distance from a voxel to a path.
        get_helix_voxels(): Generates voxels on or near the helix trajectory.
        get_helix_data(): Returns raw helix data (coordinates) without voxel processing.
    """
    def __init__(self, helix_start: (int, int, int) = (0, 0, 0), voxel_size: int = 1,
                 max_distance: int = 1, generate_percentage_of_helix=1, generate_start=0):
        """Initialize the path with helix parameters.

        Args:
            helix_start (tuple of int): Starting coordinates of the helix (x, y, z).
            voxel_size (int): Size of each voxel in millimeters.
            max_distance (int): Maximum distance from the helix path for voxel generation.
            generate_percentage_of_helix (float): Percentage of the helix used for voxel generation.
            generate_start (float): Starting point on the helix for voxel generation.

        Returns:
            None
        """
        # the helix expands in positive x, y and z direction
        # the y-expansion has negative values after half a turn ...
        self.helix_start = helix_start
        # in mm
        self.voxel_size = voxel_size
        # in voxels
        self.max_distance = max_distance
        # defines the parameter space of the helix
        self.par_space = 1
        # defines the percentage of the helix the voxels get generated on
        self.percentage_of_helix = generate_percentage_of_helix
        # defines the start of the helix the voxels get generated on
        self.start_of_voxels = generate_start

        # defines resolution of helix generation
        self.resolution = 300

        self.helix_scale = "dff"

        if self.helix_scale == "dmm":
            self.helix_factor = 100
        elif self.helix_scale == "mm":
            self.helix_factor = 10
        elif self.helix_scale == "cm":
            self.helix_factor = 1
        else:
            self.helix_factor = 10

    def x(self, t: float):
        """Calculate the x-coordinate of a point on the helix path at a given parameter.

        Args:
            t (float): Parameter defining a point along the helix.

        Returns:
            float: x-coordinate of the helix at parameter t.
        """
        return - 3 * self.helix_factor * cos((4 / self.par_space)*pi*t) + 3 * self.helix_factor + self.helix_start[0]

    def y(self, t: float):
        """Calculate the y-coordinate of a point on the helix path at a given parameter.

        Args:
            t (float): Parameter defining a point along the helix.

        Returns:
            float: y-coordinate of the helix at parameter t.
        """
        return 3 * self.helix_factor * sin((4 / self.par_space)*pi*t) + self.helix_start[1]

    def z(self, t: float):
        """Calculate the z-coordinate of a point on the helix path at a given parameter.

        Args:
            t (float): Parameter defining a point along the helix.

        Returns:
            float: z-coordinate of the helix at parameter t.
        """
        return t * 2 * self.helix_factor / self.par_space + self.helix_start[2]

    def __calculate_nearest_distance(self, voxel, path):
        """Calculate the nearest distance from a given voxel to a specified path.

        Args:
            voxel (tuple of int): Coordinates of the voxel.
            path (tuple of lists): Points representing the path.

        Returns:
            float: Minimum distance between the voxel and the path.
        """
        min_distance = float('inf')
        voxel_x, voxel_y, voxel_z = voxel

        for x, y, z in zip(path[0], path[1], path[2]):
            distance = sqrt((x - voxel_x)**2 + (y - voxel_y)**2 + (z - voxel_z)**2)
            min_distance = min(min_distance, distance)

        return min_distance

    def get_helix_voxels(self):
        """Generate and return the voxels located on or near the helix path.

        This method calculates the coordinates of the center of all voxels that are on the trajectory of the helix
        or within a specified maximum distance, starting with the helix start voxel.

        Returns:
            tuple: List of voxel coordinates, winning voxels, and their associated rewards.
        """
        #print(f"Calculating Helix Voxels")
        elements = []
        winning_voxels = []
        rewards = []
        current_reward = -1
        elements.append((self.helix_start, current_reward))
        # generate a reward system that gets lower the closer we get to the finish
        # starting reward
        reward_win = 0.0

        for i in range(self.resolution):
            t = self.par_space / self.resolution * (i / (1/self.percentage_of_helix)) + self.start_of_voxels
            # calculate the helix voxels
            x = self.x(t)
            y = self.y(t)
            z = self.z(t)
            # round to the next voxel (convert to int)
            x = int(round(x / self.voxel_size))
            y = int(round(y / self.voxel_size))
            z = int(round(z / self.voxel_size))

            path = self.get_helix_data()

            # add adjacent voxels to the elements list
            for k in range(-self.max_distance, self.max_distance + 1):
                for j in range(-self.max_distance, self.max_distance + 1):
                    for l in range(-self.max_distance, self.max_distance + 1):
                        if (i >= self.resolution-((self.max_distance+4))):
                            # Winning voxel
                            element = (x + k, y + j, z + l)
                            winning_voxels.append(element)
                            # Add Voxel with reward 0
                            elements.append((element, 0))
                        else:
                            # Non winning voxel
                            voxel = (x + k, y + j, z + l)
                            # Add Voxel with reward
                            elements.append(((x + k, y + j, z + l), current_reward))

            current_reward += 1 / self.resolution

        # Sort out dual winning voxels
        winning_voxels = list(dict.fromkeys(winning_voxels))

        helix = []
        rewards = []

        seen = set()

        for i, element in enumerate(elements):
            coords, reward = element
            if coords not in seen:
                # Dont add if we did not generate voxels at start of helix and the voxel is at the start of the helix
                # This mitigates a bug, where it will always generate a voxel at the start of the helix
                if self.start_of_voxels is not 0 and coords is not self.helix_start:
                    seen.add(coords)
                    helix.append(coords)
                    if coords in winning_voxels:
                        rewards.append(0)
                    else:
                        rewards.append(reward)

        return helix, winning_voxels, rewards

    def get_helix_data(self):
        """Retrieve the raw coordinate data of the helix path.

        This method returns the raw data of the helix path in terms of x, y, and z coordinates without any voxel processing.

        Returns:
            tuple of lists: x, y, and z coordinates of the helix.
        """
        x = []
        y = []
        z = []
        for i in range(self.resolution):
            t = self.par_space / self.resolution * i
            # calculate the helix voxels
            x.append(self.x(t))
            y.append(self.y(t))
            z.append(self.z(t))


        return (x, y, z)
