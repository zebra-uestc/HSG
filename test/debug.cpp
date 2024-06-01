#include <chrono>
#include <ctime>
#include <format>
#include <iostream>
#include <thread>
#include <vector>

#include "HSG.h"
#include "universal.h"

std::vector<std::vector<float>> train;
std::vector<std::vector<float>> test;
std::vector<std::vector<uint64_t>> neighbors;
std::vector<std::vector<float>> reference_answer;

void base_test(uint64_t short_edge_lower_limit, uint64_t short_edge_upper_limit, uint64_t cover_range,
               uint64_t build_magnification, uint64_t search_magnification)
{
    HSG::Index index(Space::Metric::Euclidean2, train[0].size(), short_edge_lower_limit, short_edge_upper_limit,
                     cover_range, build_magnification);

    for (auto i = 0; i < train.size(); ++i)
    {
        HSG::Add(index, i, train[i].data());
    }

    uint64_t total_hit = 0;
    uint64_t total_time = 0;

    for (auto i = 0; i < test.size(); ++i)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        auto query_result = HSG::Search(index, test[i].data(), neighbors[i].size(), search_magnification);
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        auto hit = verify(train, test[i], reference_answer[i], query_result);
        total_hit += hit;
    }

    std::cout << std::format("total hit: {0:<13} average time(us): {1:<13}", total_hit, total_time / test.size())
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
    load_reference_answer(argv[4], reference_answer);
    // reference_answer = get_reference_answer(train, test, neighbors);

    auto short_edge_lower_limit = std::stoull(argv[5]);
    auto short_edge_upper_limit = std::stoull(argv[6]);
    auto cover_range = std::stoull(argv[7]);
    auto build_magnification = std::stoull(argv[8]);
    auto search_magnification = std::stoull(argv[9]);

    base_test(short_edge_lower_limit, short_edge_upper_limit, cover_range, build_magnification, search_magnification);

    return 0;
}
