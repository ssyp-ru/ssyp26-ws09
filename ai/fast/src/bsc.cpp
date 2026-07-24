#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

const float C_WALL = -1.0f;
const float C_EMPTY = 0.0f;
const float C_FINISH = 1.0f;

inline std::mt19937& get_rng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

inline int get_random_int(int min_val, int max_val) {
    if (min_val > max_val) std::swap(min_val, max_val);
    std::uniform_int_distribution<int> dist(min_val, max_val);
    return dist(get_rng());
}

struct ParsedGenome {
    int pr_x;
    int pr_y;
    std::string gen1;
    std::string gen2;
};

ParsedGenome parse_genome(const std::string& genome) {
    ParsedGenome pg;
    size_t first_dot = genome.find('.');
    size_t second_dot = genome.find('.', first_dot + 1);
    size_t third_dot = genome.find('.', second_dot + 1);

    pg.pr_x = std::stoi(genome.substr(0, first_dot));
    pg.pr_y = std::stoi(genome.substr(first_dot + 1, second_dot - first_dot - 1));
    pg.gen1 = genome.substr(second_dot + 1, third_dot - second_dot - 1);
    pg.gen2 = genome.substr(third_dot + 1);
    return pg;
}

py::dict generate_with_genome(const std::string& genome, int size) {
    ParsedGenome pg = parse_genome(genome);

    py::array_t<float> maze({size, size});
    auto maze_view = maze.mutable_unchecked<2>();
    std::fill_n(maze.mutable_data(), maze.size(), C_WALL);

    int rot = 0;
    int x = pg.pr_x;
    int y = pg.pr_y;

    for (char i : pg.gen1) {
        if (x >= 0 && x < size && y >= 0 && y < size) {
            maze_view(x, y) = C_EMPTY;
        }

        if (i == 'F') {
            if (rot % 4 == 0)      x += 1;
            else if (rot % 4 == 1) y += 1;
            else if (rot % 4 == 2) x -= 1;
            else                   y -= 1;
        }
        else if (i == '+') rot += 1;
        else if (i == '-') rot -= 1;

        if (x >= size) { x -= 2; rot += 2; }
        if (y >= size) { y -= 2; rot += 2; }
        if (x < 0)     { x += 2; rot += 2; }
        if (y < 0)     { y += 2; rot += 2; }
    }

    std::vector<int> start_pos = {x, y};
    if (x >= 0 && x < size && y >= 0 && y < size) {
        maze_view(x, y) = C_EMPTY;
    }

    rot = 0;
    x = pg.pr_x;
    y = pg.pr_y;

    for (char i : pg.gen2) {
        if (x >= 0 && x < size && y >= 0 && y < size) {
            maze_view(x, y) = C_EMPTY;
        }

        if (i == 'F') {
            if (rot % 4 == 0)      x += 1;
            else if (rot % 4 == 1) y += 1;
            else if (rot % 4 == 2) x -= 1;
            else                   y -= 1;
        }
        else if (i == '+') rot += 1;
        else if (i == '-') rot -= 1;

        if (x >= size) { x -= 2; rot += 2; }
        if (y >= size) { y -= 2; rot += 2; }
        if (x < 0)     { x += 2; rot += 2; }
        if (y < 0)     { y += 2; rot += 2; }
    }

    std::vector<int> finish_pos = {x, y};
    if (x >= 0 && x < size && y >= 0 && y < size) {
        maze_view(x, y) = C_FINISH;
    }

    py::dict result;
    result["mat"] = maze;
    result["start"] = start_pos;
    result["finish"] = finish_pos;
    return result;
}

std::string crossover(const std::string& genome1, const std::string& genome2) {
    ParsedGenome p1 = parse_genome(genome1);
    ParsedGenome p2 = parse_genome(genome2);

    int new_pr_x = (p1.pr_x + p2.pr_x) / 2;
    int new_pr_y = (p1.pr_y + p2.pr_y) / 2;

    int min_len1 = std::min(p1.gen1.length(), p2.gen1.length());
    if (min_len1 >= 2) {
        int n_cut1 = get_random_int(1, min_len1 / 2);
        for (int i = 0; i < n_cut1; ++i) {
            int cut1 = get_random_int(0, min_len1);
            std::string kostil1 = p1.gen1;
            p1.gen1 = p2.gen1.substr(0, cut1) + p1.gen1.substr(cut1);
            p2.gen1 = kostil1.substr(0, cut1) + p2.gen1.substr(cut1);
        }
    }

    int min_len2 = std::min(p1.gen2.length(), p2.gen2.length());
    if (min_len2 >= 2) {
        int n_cut2 = get_random_int(1, min_len2 / 2);
        for (int i = 0; i < n_cut2; ++i) {
            int cut2 = get_random_int(0, min_len2);
            std::string kostil2 = p1.gen2;
            p1.gen2 = p2.gen2.substr(0, cut2) + p1.gen2.substr(cut2);
            p2.gen2 = kostil2.substr(0, cut2) + p2.gen2.substr(cut2);
        }
    }

    std::string new_gen1 = (get_random_int(0, 1) == 0) ? p1.gen1 : p2.gen1;
    std::string new_gen2 = (get_random_int(0, 1) == 0) ? p1.gen2 : p2.gen2;

    return std::to_string(new_pr_x) + "." + std::to_string(new_pr_y) + "." + new_gen1 + "." + new_gen2;
}

std::string mutation(const std::string& genome) {
    ParsedGenome pg = parse_genome(genome);
    std::vector<char> dna = {'F', 'F', '+', '-'};

    if (!pg.gen1.empty()) {
        int m1 = get_random_int(0, pg.gen1.length() - 1);
        int n1 = get_random_int(1, pg.gen1.length() / 10 + 1);
        for (int i = 0; i < n1; ++i) {
            char choice = dna[get_random_int(0, 3)];
            pg.gen1.insert(pg.gen1.begin() + m1, choice);
        }
    }

    if (!pg.gen2.empty()) {
        int m2 = get_random_int(0, pg.gen2.length() - 1);
        int n2 = get_random_int(1, pg.gen2.length() / 10 + 1);
        for (int i = 0; i < n2; ++i) {
            char choice = dna[get_random_int(0, 3)];
            pg.gen2.insert(pg.gen2.begin() + m2, choice);
        }
    }

    return std::to_string(pg.pr_x) + "." + std::to_string(pg.pr_y) + "." + pg.gen1 + "." + pg.gen2;
}

std::string random_generate(int size) {
    int x = get_random_int(0, size - 1);
    int y = get_random_int(0, size - 1);
    std::vector<char> dna = {'F', 'F', '+', '-'};

    int n1 = get_random_int(size * size / 8, size * size / 2);
    std::string gen1 = "";
    for (int i = 0; i < n1; ++i) {
        gen1 += dna[get_random_int(0, 3)];
    }

    int n2 = get_random_int(size * size / 8, size * size / 2);
    std::string gen2 = "";
    for (int i = 0; i < n2; ++i) {
        gen2 += dna[get_random_int(0, 3)];
    }

    return std::to_string(x) + "." + std::to_string(y) + "." + gen1 + "." + gen2;
}

int tournament(const std::vector<std::string>& population, const std::vector<float>& fitness_scores) {
    int p_size = population.size();

    if (fitness_scores.empty() || std::all_of(fitness_scores.begin(), fitness_scores.end(), [](float f){ return f == 0.0f; })) {
        return get_random_int(0, p_size - 1);
    }

    int a = get_random_int(0, p_size - 1);
    int b = get_random_int(0, p_size - 1);
    int c = get_random_int(0, p_size - 1);

    float f_a = fitness_scores[a];
    float f_b = fitness_scores[b];
    float f_c = fitness_scores[c];

    float max_fit = std::max({f_a, f_b, f_c});

    if (max_fit == f_a) return a;
    if (max_fit == f_b) return b;
    return c;
}

std::vector<std::string> select_and_crossover(const std::vector<std::string>& population, const std::vector<float>& fitness_scores) {
    std::vector<std::string> new_population;
    new_population.reserve(population.size());

    for (size_t i = 0; i < population.size(); ++i) {
        int parent1_idx = tournament(population, fitness_scores);
        int parent2_idx = tournament(population, fitness_scores);

        std::string new_genome = crossover(population[parent1_idx], population[parent2_idx]);

        if (get_random_int(0, 9) == 0) {
            new_genome = mutation(new_genome);
        }
        new_population.push_back(new_genome);
    }
    return new_population;
}

PYBIND11_MODULE(bsc_core, m) {
    m.def("generate_with_genome", &generate_with_genome, "Генерация лабиринта, старта и финиша по строке генома.");
    m.def("crossover", &crossover, "Кроссинговер двух родительских геномов.");
    m.def("mutation", &mutation, "Мутация генома.");
    m.def("random_generate", &random_generate, "Генерация случайного базового генома по размеру.");
    m.def("tournament", &tournament, "Турнирный отбор лучшего родителя.");
    m.def("select_and_crossover", &select_and_crossover, "Формирование следующего поколения популяции.");
}
