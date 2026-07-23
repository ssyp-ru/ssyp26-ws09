import random

#ГЕНЕРАТОР ЛАБИРИНТА ПО ХРОМОСОМЕ
def generate_with_genome(genome, size):
    list_genome = genome.split('.')
    pr_x = int(list_genome[0])
    pr_y = int(list_genome[1])
    gen1 = list_genome[2]
    gen2 = list_genome[3]

    maze_matrix = list()
    for i in range(size):
        maze_matrix.append([])
        for j in range(size):
            maze_matrix[i].append(1)

    rot = 0
    x = pr_x
    y = pr_y

    for i in gen1:
        maze_matrix[x][y] = 0
        if i == 'F':
            if rot % 4 == 0:
                x += 1
            elif rot % 4 == 1:
                y += 1
            elif rot % 4 == 2:
                x -= 1
            else:
                y -= 1

        if i == '+':
            rot += 1

        if i == '-':
            rot -= 1

        if x >= size:
            x -= 2
            rot += 2

        if y >= size:
            y -= 2
            rot += 2

        if x < 0:
            x += 2
            rot += 2

        if y < 0:
            y += 2
            rot += 2

    start = [x, y]
    maze_matrix[x][y] = 0

    rot = 0
    x = pr_x
    y = pr_y

    for i in gen2:
        maze_matrix[x][y] = 0
        if i == 'F':
            if rot % 4 == 0:
                x += 1
            elif rot % 4 == 1:
                y += 1
            elif rot % 4 == 2:
                x -= 1
            else:
                y -= 1

        if i == '+':
            rot += 1

        if i == '-':
            rot -= 1

        if x >= size:
            x -= 2
            rot += 2

        if y >= size:
            y -= 2
            rot += 2

        if x < 0:
            x += 2
            rot += 2

        if y < 0:
            y += 2
            rot += 2

    finish = [x, y]
    maze_matrix[x][y] = 2.

    return {
        "mat": maze_matrix,
        "start": start,
        "finish": finish
    }

#КРОССИНГОВЕР
def crossover(genome1, genome2):
    list_genome_1 = genome1.split('.')
    pr_x_1 = int(list_genome_1[0])
    pr_y_1 = int(list_genome_1[1])
    gen1_1 = list_genome_1[2]
    gen2_1 = list_genome_1[3]

    list_genome_2 = genome2.split('.')
    pr_x_2 = int(list_genome_2[0])
    pr_y_2 = int(list_genome_2[1])
    gen1_2 = list_genome_2[2]
    gen2_2 = list_genome_2[3]

    new_pr_x = (pr_x_1 + pr_x_2) // 2
    new_pr_y = (pr_y_1 + pr_y_2) // 2

    n_cut1 = random.randint(1, min(len(gen1_1), len(gen1_2)) // 2)
    for i in range(n_cut1):
        cut1 = random.randint(0, min(len(gen1_1), len(gen1_2)))
        kostil1 = gen1_1
        gen1_1 = gen1_2[:cut1] + gen1_1[cut1:len(gen1_1)]
        gen1_2 = kostil1[:cut1] + gen1_2[cut1:len(kostil1)]

    n_cut2 = random.randint(1, min(len(gen2_1), len(gen2_2)) // 2)
    for i in range(n_cut2):
        cut2 = random.randint(0, min(len(gen2_1), len(gen2_2)))
        kostil2 = gen2_1
        gen2_1 = gen2_2[:cut2] + gen2_1[cut2:len(gen2_1)]
        gen2_2 = kostil2[:cut2] + gen2_2[cut2:len(kostil2)]

    l1 = [gen1_1, gen1_2]
    l2 = [gen2_1, gen2_2]
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    new_gen1 = l1[a]
    new_gen2 = l2[b]

    new_genome = str(new_pr_x) + '.' + str(new_pr_y) + '.' + new_gen1 + '.' + new_gen2
    return new_genome

#МУТАЦИЯ
def mutation(genome):
    list_genome = genome.split('.')
    gen1 = list_genome[2]
    gen2 = list_genome[3]
    dna = ['F', 'F', '+', '-']

    m1 = random.randint(0, len(gen1) - 1)
    n1 = random.randint(1, len(gen1) // 10 + 1)
    for i in range(n1):
        gen1 = gen1[:m1] + random.choice(dna) + gen1[m1:len(gen1)]

    m2 = random.randint(0, len(gen2) - 1)
    n2 = random.randint(1, len(gen2) // 10 + 1)
    for i in range(n2):
        gen2 = gen2[:m2] + random.choice(dna) + gen2[m2:len(gen2)]

    new_genome = list_genome[0] + '.' + list_genome[1] + '.' + gen1 + '.' + gen2
    return new_genome

#ГЕНЕРАЦИЯ СЛУЧАЙНОГО ЛАБИРИНТА ПО РАЗМЕРУ
def random_generate(size):
    x = random.randint(0, size - 1)
    y = random.randint(0, size - 1)
    dna = ['F', 'F', '+', '-']

    n1 = random.randint(size * size // 8, size * size // 2)
    gen1 = str()
    for i in range(n1):
        index1 = random.randint(0, 3)
        gen1 += dna[index1]

    n2 = random.randint(size * size // 8, size * size // 2)
    gen2 = str()
    for i in range(n2):
        index2 = random.randint(0, 3)
        gen2 += dna[index2]

    genome = str(x) + '.' + str(y) + '.' + gen1 + '.' + gen2
    return genome

#ОТБОР СЛЕДУЮЩЕГО ПОКОЛЕНИЯ
def select_and_crossover(population: list, fitness_scores: list) -> list:
    new_population = list()
    for i in range(len(population)):
        new_genome = crossover(population[tournament(population, fitness_scores)],
                               population[tournament(population, fitness_scores)])
        mutant_chance = random.randint(0, 9)
        if mutant_chance == 0:
            new_genome = mutation(new_genome)
        new_population.append(new_genome)

    return new_population

#ТУРНИР ДЛЯ ОТБОРА СЛЕДУЮЩЕГО ПОКОЛЕНИЯ
def tournament(population: list, fitness_scores: list):
    a = random.randint(0, len(population) - 1)
    b = random.randint(0, len(population) - 1)
    c = random.randint(0, len(population) - 1)
    index = max(fitness_scores[a], fitness_scores[b], fitness_scores[c])

    if index == fitness_scores[a]:
        return a
    if index == fitness_scores[b]:
        return b
    if index == fitness_scores[c]:
        return c

    return
