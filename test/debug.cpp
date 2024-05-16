#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

#include "HSG.h"
#include "space.h"

std::vector<std::vector<float>> train;
std::vector<std::vector<float>> test;
std::vector<std::vector<uint64_t>> neighbors;
std::vector<std::vector<float>> reference_answer;

std::vector<std::vector<float>> get_reference_answer()
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

uint64_t verify(const uint64_t t, std::priority_queue<std::pair<float, uint64_t>> &query_result)
{
    auto result = std::vector<float>(query_result.size(), 0);

    while (!query_result.empty())
    {
        result[query_result.size() - 1] =
            Space::Euclidean2::distance(test[t].data(), train[query_result.top().second].data(), train[0].size());
        query_result.pop();
    }

    uint64_t hit = 0;

    for (auto distance : reference_answer[t])
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

void base_test(uint64_t short_edge_lower_limit, uint64_t short_edge_upper_limit, uint64_t cover_range,
               uint64_t build_magnification, uint64_t search_magnification)
{
    HSG::Index index(Space::Metric::Euclidean2, train[0].size(), short_edge_lower_limit, short_edge_upper_limit,
                     cover_range, build_magnification);

    for (auto i = 0; i < train.size(); ++i)
    {
        HSG::add(index, i, train[i].data());
    }

    uint64_t total_hit = 0;
    uint64_t total_time = 0;

    for (auto i = 0; i < test.size(); ++i)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        auto query_result = HSG::search(index, test[i].data(), neighbors[i].size(), search_magnification);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        auto hit = verify(i, query_result);
        total_hit += hit;
    }

    std::cout << std::format("total hit: {0:<13} average time: {1:<13}us", total_hit, total_time / test.size())
              << std::endl;
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

    std::cout << "CPU physical units: " << std::thread::hardware_concurrency() << std::endl;

    train = load_vector(argv[1]);
    test = load_vector(argv[2]);
    neighbors = load_neighbors(argv[3]);
    reference_answer = get_reference_answer();

    auto short_edge_lower_limit = std::stoull(argv[4]);
    auto short_edge_upper_limit = std::stoull(argv[5]);
    auto cover_range = std::stoull(argv[6]);
    auto build_magnification = std::stoull(argv[7]);
    auto search_magnification = std::stoull(argv[8]);

    base_test(short_edge_lower_limit, short_edge_upper_limit, cover_range, build_magnification, search_magnification);

    return 0;
}
