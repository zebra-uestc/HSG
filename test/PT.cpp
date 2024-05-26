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

auto available_thread = std::counting_semaphore<>(12);
auto done_semaphore = std::counting_semaphore<>(1);
uint64_t done_thread = 0;
uint64_t done_number = 0;
auto done = std::counting_semaphore<>(0);

void base_test(uint64_t short_edge_lower_limit, uint64_t short_edge_upper_limit, uint64_t cover_range,
               uint64_t build_magnification)
{
    auto test_result = std::ofstream(std::format("{0}-{1}-{2}-{3}.txt", short_edge_lower_limit, short_edge_upper_limit,
                                                 cover_range, build_magnification),
                                     std::ios::app | std::ios::out);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4} short edge upper limit: {1:<4} cover range: {2:<4} "
                               "build magnification: {3:<4}",
                               short_edge_lower_limit, short_edge_upper_limit, cover_range, build_magnification)
                << std::endl;

    auto search_magnifications = std::vector<uint64_t>{5, 10, 30, 50};
    test_result << "search magnifications: [" << search_magnifications[0];

    for (auto i = 1; i < search_magnifications.size(); ++i)
    {
        test_result << ", " << search_magnifications[i];
    }

    test_result << "]" << std::endl;

    HSG::Index index(Space::Metric::Euclidean2, train[0].size(), short_edge_lower_limit, short_edge_upper_limit,
                     cover_range, build_magnification);

    for (auto i = 0; i < train.size(); ++i)
    {
        HSG::Add(index, i, train[i].data());
    }

    for (auto i = 0; i < search_magnifications.size(); ++i)
    {
        auto search_magnification = search_magnifications[i];
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

        test_result << std::format("total hit: {0:<7} average time: {1:<7}us", total_hit, total_time / test.size())
                    << std::endl;
    }

    test_result.close();
    done_semaphore.acquire();
    ++done_thread;

    if (done_thread == done_number)
    {
        done.release();
    }

    done_semaphore.release();
    available_thread.release();
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

    auto short_edge_lower_limits = std::vector<uint64_t>();
    auto short_edge_upper_limits = std::vector<uint64_t>();
    auto cover_ranges = std::vector<uint64_t>();
    auto build_magnifications = std::vector<uint64_t>();

    auto SELL = std::stringstream(argv[4]);
    auto SEUL = std::stringstream(argv[5]);
    auto CR = std::stringstream(argv[6]);
    auto BM = std::stringstream(argv[7]);

    auto temporary = (uint64_t)0;

    while (SELL >> temporary)
    {
        short_edge_lower_limits.push_back(temporary);
    }

    while (SEUL >> temporary)
    {
        short_edge_upper_limits.push_back(temporary);
    }

    while (CR >> temporary)
    {
        cover_ranges.push_back(temporary);
    }

    while (BM >> temporary)
    {
        build_magnifications.push_back(temporary);
    }

    done_number += short_edge_lower_limits.size() * short_edge_upper_limits.size() * cover_ranges.size() *
                   build_magnifications.size();

    for (auto a = 0; a < short_edge_lower_limits.size(); ++a)
    {
        auto &short_edge_lower_limit = short_edge_lower_limits[a];

        for (auto b = 0; b < short_edge_upper_limits.size(); ++b)
        {
            auto &short_edge_upper_limit = short_edge_upper_limits[b];

            for (auto c = 0; c < cover_ranges.size(); ++c)
            {
                auto &cover_range = cover_ranges[c];

                for (auto d = 0; d < build_magnifications.size(); ++d)
                {
                    auto &build_magnification = build_magnifications[d];
                    available_thread.acquire();
                    auto one_thread = std::thread(base_test, short_edge_lower_limit, short_edge_upper_limit,
                                                  cover_range, build_magnification);
                    one_thread.detach();
                }
            }
        }
    }

    done.acquire();
    return 0;
}
