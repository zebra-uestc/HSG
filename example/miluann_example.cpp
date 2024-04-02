#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <vector>

#include "distance.h"
#include "miluann.h"

std::vector<std::vector<float>> get_reference_answer(const std::vector<std::vector<float>> &train,
                                                     const std::vector<std::vector<float>> &test,
                                                     const std::vector<std::vector<uint64_t>> &neighbors)
{
    auto reference_answer = std::vector<std::vector<float>>(test.size(), std::vector<float>(neighbors[0].size(), 0));
    for (auto i = 0; i < test.size(); ++i)
    {
        for (auto j = 0; j < neighbors[i].size(); ++j)
        {
            reference_answer[i][j] =
                euclidean2::distance(test[i].data(), train[neighbors[i][j]].data(), train[0].size());
        }
    }
    return reference_answer;
}

uint64_t verify(const std::vector<std::vector<float>> &reference_answer, const uint64_t test,
                std::priority_queue<std::pair<float, uint64_t>> &query_result)
{
    auto result = std::vector<float>(query_result.size(), 0);
    while (!query_result.empty())
    {
        result[query_result.size() - 1] = query_result.top().first;
        query_result.pop();
    }
    uint64_t hit = 0;
    for (auto distance : reference_answer[test])
    {
        if (result[hit] <= distance)
        {
            ++hit;
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

void performence_test(const std::vector<std::vector<float>> &train, const std::vector<std::vector<float>> &test,
                      const std::vector<std::vector<uint64_t>> &neighbors,
                      const std::vector<std::vector<float>> &reference_answer)
{
    auto performence_test_result = std::ofstream("performence_test_result.txt", std::ios::app);
    performence_test_result << "===" << std::endl;
    auto short_edge_bounds = std::vector<uint64_t>{4, 8, 16, 32, 64};
    auto magnifications = std::vector<uint64_t>{30, 50, 100};
    for (auto &short_edge_bound : short_edge_bounds)
    {
        for (auto &magnification : magnifications)
        {
            miluann::Index index(Distance_Type::Euclidean2, train[0].size(), short_edge_bound, 10);
            for (auto i = 0; i < train.size(); ++i)
            {
                miluann::add(index, i, train[i]);
            }
            for (auto &magnification : magnifications)
            {
                uint64_t total_hit = 0;
                uint64_t total_time = 0;
                for (auto i = 0; i < test.size(); ++i)
                {
                    auto begin = std::chrono::high_resolution_clock::now();
                    auto query_result = miluann::search(index, test[i], neighbors[i].size(), magnification);
                    auto end = std::chrono::high_resolution_clock::now();
                    total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                    auto hit = verify(reference_answer, i, query_result);
                    total_hit += hit;
                }
                performence_test_result << std::format("total hit: {0:<7} average time: {1:<5}us", total_hit,
                                                       total_time / test.size())
                                        << std::endl;
                std::cout << std::format("total hit: {0:<7} average time: {1:<5}us", total_hit,
                                         total_time / test.size())
                          << std::endl;
            }
        }
    }
    performence_test_result << "===" << std::endl;
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
    auto reference_answer = get_reference_answer(train, test, neighbors);
    performence_test(train, test, neighbors, reference_answer);
    return 0;
}
