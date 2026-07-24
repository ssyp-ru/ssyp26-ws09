#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <immintrin.h>
#include <iostream>
#include <cstdint>

namespace py = pybind11;

using gen8 = __m256i;

struct SimpleRNG {
    uint32_t state;

    void seed(const uint32_t s) {
        state = s ? s: 1;
    }

    uint32_t next(void) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        return state;
    }
};

class GeneticAlgorithm {
protected:
    SimpleRNG fast_rng;

    gen8 gen_crossover_mask(void);
    gen8 gen_mutation_mask(void);

public:
    gen8 crossover(const gen8 parentA, const gen8 parentB);
    gen8 mutation(const gen8 last_value);
};


gen8 GeneticAlgorithm::gen_crossover_mask(void) {
    uint8_t crossover_pattern = static_cast<uint8_t>(fast_rng.next());

    gen8 v_byte = _mm256_set1_epi8(static_cast<char>(crossover_pattern));
    
    gen8 bit_mask = _mm256_setr_epi8(
        0x01, 0x01, 0x01, 0x01,
        0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04,
        0x08, 0x08, 0x08, 0x08,
        0x10, 0x10, 0x10, 0x10,
        0x20, 0x20, 0x20, 0x20,
        0x40, 0x40, 0x40, 0x40,
        0x80, 0x80, 0x80, 0x80
    );

    gen8 clean_bites = _mm256_and_si256(v_byte, bit_mask);

    return _mm256_cmpeq_epi8(clean_bites, bit_mask);
}

gen8 GeneticAlgorithm::gen_mutation_mask(void) {
    uint32_t A = fast_rng.next();
    uint32_t B = fast_rng.next();
    uint32_t C = fast_rng.next();
    uint32_t D = fast_rng.next();
    uint32_t E = fast_rng.next();

    gen8 reg1 = _mm256_setr_epi32(A, B, C, D, E, A, A, B);
    gen8 reg2 = _mm256_setr_epi32(B, C, D, E, A, C, D, D);

    gen8 mask1 = _mm256_set1_epi32(0x00FF0000);
    gen8 mask2 = _mm256_set1_epi32(0x0000FF00);
    gen8 mask3 = _mm256_set1_epi32(0x000000FF);

    gen8 reg_and = _mm256_and_si256(_mm256_and_si256(reg1, reg2), mask1);
    gen8 reg_xor = _mm256_and_si256(_mm256_xor_si256(reg1, reg2), mask2);
    gen8 reg_or = _mm256_and_si256(_mm256_or_si256(reg1, reg2), mask3);

    // Если [1 байт - команда, 2 и 3 - номер стены, 4 - сид]
    // 1 байт может указывать например, что сид также содержит информацию о модификации типов стен.
    // Это очень старая и, скорее всего, бесполезная попытка реализации. Её надо пересматривать с учётом
    // структуры генов в maze.py
    return _mm256_or_si256(_mm256_or_si256(reg_and, reg_xor), reg_or);
}

gen8 GeneticAlgorithm::crossover(const gen8 parentA, const gen8 parentB) {
    gen8 reg_mask = gen_crossover_mask();

    gen8 reg_child = _mm256_blendv_epi8(parentA, parentB, reg_mask);

    return reg_child;
}

gen8 GeneticAlgorithm::mutation(const gen8 last_value) {
    gen8 mut_mask = gen_mutation_mask();

    return _mm256_xor_si256(last_value, mut_mask);
}


class ProcedureGeneration {
public:
    py::array_t
};
