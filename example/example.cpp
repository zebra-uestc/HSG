#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "bruteforce.h"
#include "nnhnsw.h"

uint64_t verify(const std::vector<uint64_t> &neighbors, const std::map<float, uint64_t> &query_result)
{
    uint64_t hit = 0;
    auto query_result_iterator = query_result.begin();
    for (auto &neighbor : neighbors)
    {
        if (neighbor == query_result_iterator->second)
        {
            ++hit;
            ++query_result_iterator;
        }
    }
    return hit;
}

std::vector<std::vector<float>> load_vector(const char *file_path)
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

std::vector<std::vector<uint64_t>> load_neighbors(const char *file_path)
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

int main(int argc, char **argv)
{
#if defined(__AVX512F__)
    std::cout << "AVX512 supported. " << std::endl;
#elif defined(__AVX__)
    std::cout << "AVX supported. " << std::endl;
#elif defined(__SSE__)
    std::cout << "SSE supported. " << std::endl;
#else
    std::cout << "no SIMD supported. " << std::endl;
#endif
    auto train = load_vector(argv[1]);
    auto test = load_vector(argv[2]);
    auto neighbors = load_neighbors(argv[3]);
    uint64_t max_connect = 10;
    uint64_t relaxed_monotonicity = 5;
    uint64_t step = 3;
    uint64_t query_relaxed_monotonicity = 30;
    if (argc > 4)
    {
        max_connect = std::stoull(argv[4]);
    }
    if (argc > 5)
    {
        relaxed_monotonicity = std::stoull(argv[5]);
    }
    if (argc > 6)
    {
        step = std::stoull(argv[6]);
    }
    if (argc > 7)
    {
        query_relaxed_monotonicity = std::stoull(argv[7]);
    }
    std::cout << "max connect: " << max_connect << std::endl;
    std::cout << "relaxed monotonicity: " << relaxed_monotonicity << std::endl;
    std::cout << "step: " << step << std::endl;
    std::cout << "query relaxed monotonicity: " << query_relaxed_monotonicity << std::endl;
    nnhnsw::Index<float> index(train, Distance_Type::Euclidean2, max_connect, relaxed_monotonicity, step);
    uint64_t total_hit = 0;
    uint64_t total_time = 0;
    for (auto i = 0; i < test.size(); ++i)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        auto query_result = nnhnsw::query<float>(index, test[i], neighbors[i].size(), query_relaxed_monotonicity);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "one query costs(us): "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        auto hit = verify(neighbors[i], query_result);
        total_hit += hit;
        std::cout << "hit: " << hit << std::endl;
    }
    std::cout << "average time: " << total_time / test.size() << std::endl;
    std::cout << "total hit: " << total_hit << std::endl;
    return 0;
}
