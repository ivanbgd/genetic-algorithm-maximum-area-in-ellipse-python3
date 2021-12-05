"""
An Application of Genetic Algorithms

Task:
Inscribe a triangle of the maximum area in a given ellipse.
Ellipse is defined as: (x/a)^2  + (y/b)^2 = 1

Python 2.7 & 3.10
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

start_time = timer()

# Definition of parameters of the GA
NUM_UNITS = 100  # N - number of units (chromosomes)
CHROMOSOME_LENGTH = 36  # L - length of a chromosome (12 bits per vertex)
CROSSOVER_PROB = 0.8  # pc - crossover probability
MUTATION_PROB = 0.001  # pm - mutation probability
GENERATION_GAP = 0.8  # G - generation gap

# Parameters of the ellipse
a = 5
b = 3


#################
# The Algorithm #
#################

# Maximum number with L bits
MAX_NUM = 2**CHROMOSOME_LENGTH - 1

# We'll use these a lot
a2 = float(a * a)
b2 = float(b * b)
Lthird = CHROMOSOME_LENGTH // 3
twoLthirds = 2 * Lthird
maxl3 = 2**Lthird - 1
pi_half = np.pi / 2
three_pi_halves = 3. * pi_half
a2rec = 1. / a2

# Array of N chromosomes, each consisting of L bits
new_gen = np.array([np.binary_repr(0, CHROMOSOME_LENGTH)] * NUM_UNITS)
old_gen = np.array([np.binary_repr(0, CHROMOSOME_LENGTH)] * NUM_UNITS)

# Vertices of the triangles; a vertex is defined by its angle to the positive x-axis in radians
V1 = np.empty(NUM_UNITS)
V2 = np.empty(NUM_UNITS)
V3 = np.empty(NUM_UNITS)

# Coordinates of the vertices
x1 = np.empty(NUM_UNITS)
y1 = np.empty(NUM_UNITS)
x2 = np.empty(NUM_UNITS)
y2 = np.empty(NUM_UNITS)
x3 = np.empty(NUM_UNITS)
y3 = np.empty(NUM_UNITS)

# Fitness function
fitness = np.empty(NUM_UNITS)

# Array that holds the maximum value of fitness function in every generation
fitness_maximums_per_gen = []

# The first generation
for i in range(NUM_UNITS):
    tmp = np.random.random()   # a random number in [0.0, 1.0)
    new_gen[i] = np.binary_repr(int(round(tmp * MAX_NUM)), CHROMOSOME_LENGTH)

# Generation number counter
generation_counter = 0

# Condition for staying in the loop
condition = True

# The main loop
while condition:

    # Evaluation of the newly formed generation
    for i in range(NUM_UNITS):
        V1[i] = (float(int(new_gen[i][:Lthird], 2)) / maxl3) * 2.0 * np.pi
        V2[i] = (float(int(new_gen[i][Lthird:twoLthirds], 2)) / maxl3) * 2.0 * np.pi
        V3[i] = (float(int(new_gen[i][twoLthirds:], 2)) / maxl3) * 2.0 * np.pi

        # Coordinates of vertex V1
        if (V1[i] < pi_half) or (V1[i] > three_pi_halves):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(V1[i])**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(V1[i])**2)/b2)))
        y = x * math.tan(V1[i])
        x1[i] = x
        y1[i] = y

        # Coordinates of vertex V2
        if (V2[i] < pi_half) or (V2[i] > three_pi_halves):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(V2[i])**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(V2[i])**2)/b2)))
        y = x * math.tan(V2[i])
        x2[i] = x
        y2[i] = y

        # Coordinates of vertex V3
        if (V3[i] < pi_half) or (V3[i] > three_pi_halves):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(V3[i])**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(V3[i])**2)/b2)))
        y = x * math.tan(V3[i])
        x3[i] = x
        y3[i] = y

        # Lengths of the triangle's edges
        la = math.sqrt((x2[i] - x1[i])**2 + (y2[i] - y1[i])**2)
        lb = math.sqrt((x3[i] - x1[i])**2 + (y3[i] - y1[i])**2)
        lc = math.sqrt((x3[i] - x2[i])**2 + (y3[i] - y2[i])**2)

        # Semi-perimeter of the triangle
        s = (la + lb + lc) / 2.

        # Fitness function (Heron's formula)
        fitness[i] = math.sqrt(s * (s - la) * (s - lb) * (s - lc))

    # The highest (best) value of fitness
    maxf = np.amax(fitness)

    # Index of the highest value of fitness
    maxfindex = np.argmax(fitness)

    fitness_maximums_per_gen.append(maxf)

    # Plotting the result
    plt.figure("An Application of Genetic Algorithms")
    plt.title("Generation number {}\nThe best result: {:.4f}".format(generation_counter + 1, maxf))
    plt.xlim(-a - 1, a + 1)
    plt.ylim(-b - 1, b + 1)

    # Drawing the ellipse
    ellipse = np.array([[0.] * 361, [0.] * 361], dtype=float)
    for i in range(361):
        theta = 2.*np.pi*i/360.
        if (theta <= pi_half) or (theta > three_pi_halves):
            x = math.fabs(math.sqrt(1./(a2rec + (math.tan(theta)**2)/b2)))
        else:
            x = -math.fabs(math.sqrt(1./(a2rec + (math.tan(theta)**2)/b2)))
        y = x * math.tan(theta)
        ellipse[0][i] = x
        ellipse[1][i] = y
    plt.plot(ellipse[0], ellipse[1], 'g', linewidth=4.0)  # thick green line

    # Drawing the triangles that we got
    for i in range(NUM_UNITS):
        if fitness[i] == maxf:
            # The best chromosome - the triangle with the largest area
            plt.plot([x1[i], x2[i], x3[i], x1[i]], [y1[i], y2[i], y3[i], y1[i]], 'r', linewidth = 4.0)  # thick red line
        else:
            # The other chromosomes (triangles); they are all inscribed in the ellipse, but they don't have the largest area
            plt.plot([x1[i], x2[i], x3[i], x1[i]], [y1[i], y2[i], y3[i], y1[i]], 'b', linewidth = 0.5)  # thin blue line

    # Hold the graph for a given amount of time in seconds
    plt.pause(0.1)
    plt.plot()

    ### Natural selection by the roulette wheel method ###

    old_gen = np.copy(new_gen)

    # Cumulative function
    cumf = fitness.cumsum()

    # We let the best chromosome pass to the next generation directly.
    new_gen[0] = old_gen[maxfindex]

    # We also let another randomly chosen (1-G)*N-1 chromosomes pass. Probability of their selection depends on f(i).
    for i in range(1, int(round((1 - GENERATION_GAP) * NUM_UNITS))):
        tmp = np.random.random() * cumf[-1]
        firstPositive, firstPositiveIndex = np.amax(np.sign(cumf - tmp)), np.argmax(np.sign(cumf - tmp))
        new_gen[i] = old_gen[firstPositiveIndex]

    ### The rest of the new generation is formed by crossover (crossbreeding) ###

    # There are two parents, and two offsprings
    for i in range((NUM_UNITS - int(round((1 - GENERATION_GAP) * NUM_UNITS))) // 2):
        tmp = np.random.random() * cumf[-1]
        firstPositive, firstPositiveIndex = np.amax(np.sign(cumf - tmp)), np.argmax(np.sign(cumf - tmp))
        parent1 = old_gen[firstPositiveIndex]

        tmp = np.random.random() * cumf[-1]
        firstPositive, firstPositiveIndex = np.amax(np.sign(cumf - tmp)), np.argmax(np.sign(cumf - tmp))
        parent2 = old_gen[firstPositiveIndex]

        if np.random.random() < CROSSOVER_PROB:
            # crossover
            crossPoint = int(round(np.random.random() * (CHROMOSOME_LENGTH - 2))) + 1   # the crossover point can be after MSB and before LSB, thus L-2
            new_gen[int(round((1 - GENERATION_GAP) * NUM_UNITS)) + 2 * i] = parent1[:crossPoint] + parent2[crossPoint:]         # offspring 1
            new_gen[int(round((1 - GENERATION_GAP) * NUM_UNITS)) + 2 * i + 1] = parent2[:crossPoint] + parent1[crossPoint:]     # offspring 2
        else:
            # no crossover
            new_gen[int(round((1 - GENERATION_GAP) * NUM_UNITS)) + 2 * i] = parent1         # offspring 1
            new_gen[int(round((1 - GENERATION_GAP) * NUM_UNITS)) + 2 * i + 1] = parent2     # offspring 2

    ### Mutation ###

    for i in range(int(CHROMOSOME_LENGTH * NUM_UNITS * MUTATION_PROB)):
        chromosomeIndex = int(round(np.random.random() * (NUM_UNITS - 1)))
        bitPosition = int(round(np.random.random() * (CHROMOSOME_LENGTH - 1)))
        new_gen[chromosomeIndex] = new_gen[chromosomeIndex][:bitPosition] + ('0' if new_gen[chromosomeIndex][bitPosition] == '1' else '1') + new_gen[chromosomeIndex][bitPosition + 1:]

    generation_counter += 1

    # Exit condition - We want fitness functions of the first numchrom chromosomes to be inside of a given difference.
    numchrom = 10
    difference = 0.001
    fitness.sort()
    if abs(fitness[-1] - fitness[-numchrom]) < difference:
        condition = False

end_time = timer()

print("The result is: {}".format(max(fitness_maximums_per_gen)))
print("The algorithm took {} generations, and {} seconds to complete.".format(generation_counter, round(end_time - start_time, 3)))
print("The maximum value of fitness function through generations:\n{}".format(fitness_maximums_per_gen))

plt.figure("The maximum value of fitness function through generations")
plt.title("The maximum value of fitness function through generations")
plt.xlim(0, generation_counter - 1)
plt.plot(fitness_maximums_per_gen)
plt.show()
