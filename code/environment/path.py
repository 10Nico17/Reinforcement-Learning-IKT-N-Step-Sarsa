import numpy as np
from math import cos, sin, pi, sqrt


class Path:
    def __init__(self, helix_start: (int, int, int) = (0, 0, 0), voxel_size: int = 1, max_distance: int = 1, generate_percentage_of_helix=1, generate_start=0):
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

        self.helix_scale = "mm"

        if self.helix_scale == "dmm":
            self.helix_factor = 100
        elif self.helix_scale == "mm":
            self.helix_factor = 10
        elif self.helix_scale == "cm":
            self.helix_factor = 1
        else:
            self.helix_factor = 15

    def x(self, t: float):
        return - 3 * self.helix_factor * cos((4 / self.par_space)*pi*t) + 3 * self.helix_factor + self.helix_start[0]

    def y(self, t: float):
        return 3 * self.helix_factor * sin((4 / self.par_space)*pi*t) + self.helix_start[1]

    def z(self, t: float):
        return t * 2 * self.helix_factor / self.par_space + self.helix_start[2]

    def __calculate_nearest_distance(self, voxel, path):
        min_distance = float('inf')
        voxel_x, voxel_y, voxel_z = voxel

        for x, y, z in zip(path[0], path[1], path[2]):
            distance = sqrt((x - voxel_x)**2 + (y - voxel_y)**2 + (z - voxel_z)**2)
            min_distance = min(min_distance, distance)

        return min_distance

    # returns the coordinates of the center of all the voxels that are on the trajectory or
    # within the max_distance starting with the helix_start voxel returns a list of tuples (x, y, z)
    def get_helix_voxels(self):
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
                            # Calculate reward based on distance along helix and closeness to helix
                            distance = round(self.__calculate_nearest_distance(voxel, path)) / 3
                            #input(f"{voxel} distance: {distance}")
                            # Add Voxel with reward
                            #elements.append(((x + k, y + j, z + l), current_reward-distance))
                            elements.append(((x + k, y + j, z + l), current_reward))

            current_reward += 1 / self.resolution
            #current_reward = -1
            #if ((((i/self.resolution) + 0.01) * 100) % 5) == 0:
                #print(f"Process: {int((i/self.resolution)*100)}%\r", end='')

        #print(f"Process: 100%")

        # Sort out dual winning voxels
        #print("Removing double elements ")
        winning_voxels = list(dict.fromkeys(winning_voxels))

        helix = []
        rewards = []

        seen = set()

        for i, element in enumerate(elements):
            coords, reward = element
            #print(f"\rProcess: {int((i/len(elements))*100)}%\r", end='')
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

        #print(f"Process: 100%")

        #for coords, reward in zip(helix, rewards):
        #    print(f"Reward for {coords}, {reward}")
        #    if(coords in winning_voxels):
        #        print(f"In winning voxels!")

        #print(f"winning voxels: {winning_voxels}")

        return helix, winning_voxels, rewards

    # returns raw helix data
    def get_helix_data(self):
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


# example usage with plot
"""
helix = Path(helix_start=(0, 0, 0), voxel_size=1, max_distance=2)

helix_voxels = helix.get_helix_voxels()

# print the helix voxels as scatter plot with small dots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = []
y = []
z = []

for voxel in helix_voxels:
    x.append(voxel[0])
    y.append(voxel[1])
    z.append(voxel[2])

ax.scatter(x, y, z, marker=".", s=1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
"""
