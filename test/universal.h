#include <ctime>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "../source/space.h"

inline std::vector<std::vector<float>> get_reference_answer(const std::vector<std::vector<float>> &train,
                                                            const std::vector<std::vector<float>> &test,
                                                            const std::vector<std::vector<uint64_t>> &neighbors)
{
    auto reference_answer = std::vector<std::vector<float>>(test.size(), std::vector<float>(neighbors[0].size(), 0));

    for (auto i = 0; i < test.size(); ++i)
    {
        for (auto j = 0; j < neighbors[i].size(); ++j)
        {
            reference_answer[i][j] =
                Space::Euclidean2::distance(test[i].data(), train[neighbors[i][j]].data(), train[0].size());
        }
    }

    return reference_answer;
}

inline void write_reference_answer(const char *file_path, const std::vector<std::vector<float>> &reference_answer)
{
    std::ofstream file;
    file.open(file_path, std::ios::out | std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "open file failed. " << std::endl;
        exit(0);
    }

    uint64_t number = reference_answer.size();
    uint64_t k = reference_answer[0].size();
    file.write((char *)&number, sizeof(uint64_t));
    file.write((char *)&k, sizeof(uint64_t));

    for (auto i = 0; i < number; ++i)
    {
        file.write((char *)reference_answer[i].data(), sizeof(float) * k);
    }

    file.close();
}

inline void load_reference_answer(const char *file_path, std::vector<std::vector<float>> &reference_answer)
{
    std::ifstream file;
    file.open(file_path, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "open file failed. " << std::endl;
        exit(0);
    }

    uint64_t number = 0;
    uint64_t k = 0;
    file.read((char *)&number, sizeof(uint64_t));
    file.read((char *)&k, sizeof(uint64_t));

    reference_answer.resize(number, std::vector<float>(k));

    for (auto i = 0; i < number; ++i)
    {
        file.read((char *)reference_answer[i].data(), sizeof(float) * k);
    }

    file.close();
}

inline uint64_t verify(const std::vector<std::vector<float>> &train, const std::vector<float> &test,
                       const std::vector<float> &reference_answer,
                       std::priority_queue<std::pair<float, uint64_t>> &query_result)
{
    auto result = std::vector<float>(query_result.size(), 0);

    while (100 < query_result.size())
    {
        query_result.pop();
    }

    for (uint64_t hit = 100; !query_result.empty(); --hit)
    {
        auto distance =
            Space::Euclidean2::distance(test.data(), train[query_result.top().second].data(), train[0].size());

        if (distance <= reference_answer[99])
        {
            return hit;
        }

        query_result.pop();
    }

    return 0;
}

inline std::vector<std::vector<float>> load_vector(const char *file_path)
{
    std::ifstream vectors_file;
    vectors_file.open(file_path, std::ios::in | std::ios::binary);

    if (!vectors_file.is_open())
    {
        std::cout << "open file failed. " << std::endl;
        exit(0);
    }

    uint64_t count = 0;
    uint64_t dimension = 0;
    vectors_file.read((char *)&count, sizeof(uint64_t));
    vectors_file.read((char *)&dimension, sizeof(uint64_t));
    auto vectors = std::vector<std::vector<float>>(count, std::vector<float>(dimension, 0));

    for (auto i = 0; i < count; ++i)
    {
        for (auto j = 0; j < dimension; ++j)
        {
            vectors_file.read((char *)&vectors[i][j], sizeof(float));
        }
    }

    vectors_file.close();
    return vectors;
}

inline std::vector<std::vector<uint64_t>> load_neighbors(const char *file_path)
{
    std::ifstream neighbors_file;
    neighbors_file.open(file_path, std::ios::in | std::ios::binary);

    if (!neighbors_file.is_open())
    {
        std::cout << "open file failed. " << std::endl;
        exit(0);
    }

    uint64_t count = 0;
    uint64_t neighbor_count = 0;
    neighbors_file.read((char *)&count, sizeof(uint64_t));
    neighbors_file.read((char *)&neighbor_count, sizeof(uint64_t));
    auto neighbors = std::vector<std::vector<uint64_t>>(count, std::vector<uint64_t>(neighbor_count, 0));

    for (auto i = 0; i < count; ++i)
    {
        for (auto j = 0; j < neighbor_count; ++j)
        {
            neighbors_file.read((char *)&neighbors[i][j], sizeof(uint64_t));
        }
    }

    neighbors_file.close();
    return neighbors;
}

inline void bvecs_vectors(const char *file_path, std::vector<std::vector<float>> &vectors, uint64_t number = 0)
{
    auto file = std::ifstream(file_path, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "open file failed. " << std::endl;
        exit(0);
    }

    int dimension = 0;
    file.read((char *)&dimension, 4);

    file.seekg(0, std::ios::end);
    auto end = file.tellg();
    auto all = ((uint64_t)end / (dimension + 4));

    if (number == 0 || all < number)
    {
        number = all;
    }

    vectors.resize(number);

    file.seekg(0, std::ios::beg);

    for (auto i = 0; i < number; ++i)
    {
        file.seekg(4, std::ios::cur);

        for (auto j = 0; j < dimension; ++j)
        {
            unsigned char temporary = 0;
            file.read((char *)&temporary, sizeof(unsigned char));
            vectors[i].push_back(temporary);
        }
    }

    file.close();
}

inline void ivecs(const char *file_path, std::vector<std::vector<uint64_t>> &neighbors, uint64_t number = 0)
{
    auto file = std::ifstream(file_path, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "open file failed. " << std::endl;
        exit(0);
    }

    int k = 0;
    file.read((char *)&k, 4);

    file.seekg(0, std::ios::end);
    auto end = file.tellg();
    auto all = ((uint64_t)end / ((k + 1) * 4));

    if (number == 0 || all < number)
    {
        number = all;
    }

    neighbors.resize(number);

    file.seekg(0, std::ios::beg);

    for (auto i = 0; i < number; ++i)
    {
        file.seekg(4, std::ios::cur);

        for (auto j = 0; j < k; ++j)
        {
            int temporary = 0;
            file.read((char *)&temporary, sizeof(int));
            neighbors[i].push_back(temporary);
        }
    }

    file.close();
}

inline void load_deleted(const char *file_path, std::vector<uint64_t> &result)
{
    auto file = std::ifstream(file_path, std::ios::in | std::ios::binary);

    uint64_t number = 0;
    file.read((char *)&number, sizeof(uint64_t));

    for (auto i = 0; i < number; ++i)
    {
        uint64_t temporary = 0;
        file.read((char *)&temporary, sizeof(uint64_t));
        result.push_back(temporary);
    }
}

inline uint64_t verify_with_delete(const std::vector<std::vector<float>> &train, const std::vector<float> &test,
                                   const std::vector<uint64_t> &neighbors, const std::vector<float> &reference_answer,
                                   std::priority_queue<std::pair<float, uint64_t>> &query_result,
                                   std::unordered_set<uint64_t> &relevant, uint64_t k)
{
    auto result = std::vector<float>(query_result.size(), 0);

    while (!query_result.empty())
    {
        if (relevant.contains(query_result.top().second))
        {
            std::cout << "wrong!" << std::endl;
            std::exit(0);
        }

        result[query_result.size() - 1] =
            Space::Euclidean2::distance(test.data(), train[query_result.top().second].data(), train[0].size());
        query_result.pop();
    }

    uint64_t hit = 0;

    for (auto i = 0; hit < k && i < reference_answer.size(); ++i)
    {
        if (!relevant.contains(neighbors[i]))
        {
            if (result[hit] <= reference_answer[i])
            {
                ++hit;
            }
        }
    }

    return hit;
}
