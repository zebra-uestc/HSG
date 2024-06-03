#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

#include "HSG.h"
#include "universal.h"

std::vector<std::vector<float>> train;
std::vector<std::vector<float>> test;
std::vector<std::vector<uint64_t>> neighbors;
std::vector<std::vector<float>> reference_answer;
std::string name;

auto available_thread = std::counting_semaphore<>(12);
auto done_semaphore = std::counting_semaphore<>(1);
uint64_t done_thread = 0;
uint64_t done_number = 0;
auto done = std::counting_semaphore<>(0);

void base_test(uint64_t short_edge_lower_limit, uint64_t short_edge_upper_limit, uint64_t cover_range,
               uint64_t build_magnification)
{
    auto test_result = std::ofstream(std::format("result/HSG/{0}-{1}-{2}-{3}-{4}.txt", name, short_edge_lower_limit,
                                                 short_edge_upper_limit, cover_range, build_magnification),
                                     std::ios::app | std::ios::out);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4} short edge upper limit: {1:<4} cover range: {2:<4} "
                               "build magnification: {3:<4}",
                               short_edge_lower_limit, short_edge_upper_limit, cover_range, build_magnification)
                << std::endl;

    auto search_magnifications = std::vector<uint64_t>{10, 30, 50, 100};
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

    auto cover_rate = HSG::Calculate_Coverage(index);
    test_result << std::format("cover rate: {0:<6.4}", cover_rate) << std::endl;

    for (auto i = 0; i < search_magnifications.size(); ++i)
    {
        auto search_magnification = search_magnifications[i];
        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), 100, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result);
            total_hit += hit;
        }

        test_result << std::format("search magnification: {0:<4} total hit: {1:<10} average time: {2:<10}us",
                                   search_magnification, total_hit, total_time / test.size())
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

    name = std::string(argv[5]);

    if (name == "sift1B")
    {
        for (auto i = 0; i < 8; ++i)
        {
            available_thread.acquire();
        }

        bvecs_vectors(argv[1], train, 10000000);
        bvecs_vectors(argv[2], test);
        ivecs(argv[3], neighbors);
    }
    else
    {
        if (name == "gist")
        {
            for (auto i = 0; i < 4; ++i)
            {
                available_thread.acquire();
            }
        }

        train = load_vector(argv[1]);
        test = load_vector(argv[2]);
        neighbors = load_neighbors(argv[3]);
    }

    load_reference_answer(argv[4], reference_answer);

    auto short_edge_lower_limits = std::vector<uint64_t>();
    auto short_edge_upper_limits = std::vector<uint64_t>();
    auto cover_ranges = std::vector<uint64_t>();
    auto build_magnifications = std::vector<uint64_t>();

    auto SELL = std::stringstream(argv[6]);
    auto SEUL = std::stringstream(argv[7]);
    auto CR = std::stringstream(argv[8]);
    auto BM = std::stringstream(argv[9]);

    uint64_t temporary = 0;

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

    done_number += short_edge_lower_limits.size() * cover_ranges.size() * build_magnifications.size();

    for (auto a = 0; a < short_edge_lower_limits.size(); ++a)
    {
        auto &short_edge_lower_limit = short_edge_lower_limits[a];
        auto &short_edge_upper_limit = short_edge_upper_limits[a];

        for (auto b = 0; b < cover_ranges.size(); ++b)
        {
            auto &cover_range = cover_ranges[b];

            for (auto c = 0; c < build_magnifications.size(); ++c)
            {
                auto &build_magnification = build_magnifications[c];
                available_thread.acquire();
                auto one_thread = std::thread(base_test, short_edge_lower_limit, short_edge_upper_limit, cover_range,
                                              build_magnification);
                one_thread.detach();
            }
        }
    }

    done.acquire();
    return 0;
}
