#include <chrono>
#include <ctime>
#include <format>
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
std::vector<uint64_t> irrelevant;
std::vector<uint64_t> relevant;

auto done_semaphore = std::counting_semaphore<>(1);
uint64_t done_thread = 0;
const uint64_t done_number = 3;
auto done = std::counting_semaphore<>(0);

uint64_t short_edge_lower_limit;
uint64_t short_edge_upper_limit;
uint64_t cover_range;
uint64_t build_magnification;
uint64_t k;
uint64_t search_magnification;

void delete_relevant(HSG::Index index)
{
    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);

    auto test_result =
        std::ofstream(std::format("result/HSG/DR-{0}-{1}-{2}-{3}-{4}-{5}.txt", name, UTC_time->tm_year + 1900,
                                  UTC_time->tm_mon + 1, UTC_time->tm_mday, UTC_time->tm_hour + 8, UTC_time->tm_min),
                      std::ios::app | std::ios::out);

    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4}", short_edge_lower_limit) << std::endl;
    test_result << std::format("cover range: {0:<4}", cover_range) << std::endl;
    test_result << std::format("build magnification: {0:<4}", build_magnification) << std::endl;
    test_result << std::format("top k: {0:<4}", k) << std::endl;
    test_result << std::format("search magnification: {0:<4}", search_magnification) << std::endl;
    test_result << std::format("relevant number: {0:<9}", relevant.size()) << std::endl;

    uint64_t relevant_number = 0;
    auto iterator = relevant.begin();
    auto deleted_relevant = std::unordered_set<uint64_t>();

    // 删去1/3
    {
        while (relevant_number < relevant.size() / 3)
        {
            HSG::Erase(index, *iterator);
            deleted_relevant.insert(*iterator);
            ++relevant_number;
            ++iterator;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result,
                                          deleted_relevant, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 删去2/3
    {
        while (relevant_number < (relevant.size() / 3) * 2)
        {
            HSG::Erase(index, *iterator);
            deleted_relevant.insert(*iterator);
            ++relevant_number;
            ++iterator;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result,
                                          deleted_relevant, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 全删
    {
        while (relevant_number < relevant.size())
        {
            HSG::Erase(index, *iterator);
            deleted_relevant.insert(*iterator);
            ++relevant_number;
            ++iterator;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result,
                                          deleted_relevant, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    relevant_number = 0;
    iterator = relevant.begin();

    // 加回1/3
    {
        while (relevant_number < relevant.size() / 3)
        {
            HSG::Add(index, *iterator, train[*iterator].data());
            deleted_relevant.erase(*iterator);
            ++relevant_number;
            ++iterator;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result,
                                          deleted_relevant, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 加回2/3
    {
        while (relevant_number < (relevant.size() / 3) * 2)
        {
            HSG::Add(index, *iterator, train[*iterator].data());
            deleted_relevant.erase(*iterator);
            ++relevant_number;
            ++iterator;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result,
                                          deleted_relevant, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 加回全部
    {
        while (relevant_number < relevant.size())
        {
            HSG::Add(index, *iterator, train[*iterator].data());
            deleted_relevant.erase(*iterator);
            ++relevant_number;
            ++iterator;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result,
                                          deleted_relevant, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
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
}

void delete_irrelevant(HSG::Index index)
{
    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);

    auto test_result =
        std::ofstream(std::format("result/HSG/DI-{0}-{1}-{2}-{3}-{4}-{5}.txt", name, UTC_time->tm_year + 1900,
                                  UTC_time->tm_mon + 1, UTC_time->tm_mday, UTC_time->tm_hour + 8, UTC_time->tm_min),
                      std::ios::app | std::ios::out);

    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4}", short_edge_lower_limit) << std::endl;
    test_result << std::format("cover range: {0:<4}", cover_range) << std::endl;
    test_result << std::format("build magnification: {0:<4}", build_magnification) << std::endl;
    test_result << std::format("top k: {0:<4}", k) << std::endl;
    test_result << std::format("search magnification: {0:<4}", search_magnification) << std::endl;
    test_result << std::format("irrelevant number: {0:<9}", irrelevant.size()) << std::endl;

    uint64_t irrelevant_number = 0;

    // 删去1/3
    {
        while (irrelevant_number < irrelevant.size() / 3)
        {
            HSG::Erase(index, irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 删去2/3
    {
        while (irrelevant_number < (irrelevant.size() / 3) * 2)
        {
            HSG::Erase(index, irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 全删
    {
        while (irrelevant_number < irrelevant.size())
        {
            HSG::Erase(index, irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    irrelevant_number = 0;

    // 加回1/3
    {
        while (irrelevant_number < irrelevant.size() / 3)
        {
            HSG::Add(index, irrelevant[irrelevant_number], train[irrelevant[irrelevant_number]].data());
            ++irrelevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 加回2/3
    {
        while (irrelevant_number < (irrelevant.size() / 3) * 2)
        {
            HSG::Add(index, irrelevant[irrelevant_number], train[irrelevant[irrelevant_number]].data());
            ++irrelevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 加回全部
    {
        while (irrelevant_number < irrelevant.size())
        {
            HSG::Add(index, irrelevant[irrelevant_number], train[irrelevant[irrelevant_number]].data());
            ++irrelevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
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
}

void delete_both(HSG::Index index)
{
    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);

    auto test_result =
        std::ofstream(std::format("result/HSG/DB-{0}-{1}-{2}-{3}-{4}-{5}.txt", name, UTC_time->tm_year + 1900,
                                  UTC_time->tm_mon + 1, UTC_time->tm_mday, UTC_time->tm_hour + 8, UTC_time->tm_min),
                      std::ios::app | std::ios::out);

    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4}", short_edge_lower_limit) << std::endl;
    test_result << std::format("cover range: {0:<4}", cover_range) << std::endl;
    test_result << std::format("build magnification: {0:<4}", build_magnification) << std::endl;
    test_result << std::format("top k: {0:<4}", k) << std::endl;
    test_result << std::format("search magnification: {0:<4}", search_magnification) << std::endl;
    test_result << std::format("irrelevant number: {0:<9}", irrelevant.size()) << std::endl;
    test_result << std::format("relevant number: {0:<9}", relevant.size()) << std::endl;

    uint64_t irrelevant_number = 0;
    uint64_t relevant_number = 0;
    auto deleted = std::unordered_set<uint64_t>();

    // 删去1/3
    {
        while (irrelevant_number < irrelevant.size() / 3)
        {
            HSG::Erase(index, irrelevant[irrelevant_number]);
            deleted.insert(irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        while (relevant_number < relevant.size() / 3)
        {
            HSG::Erase(index, relevant[relevant_number]);
            deleted.insert(relevant[relevant_number]);
            ++relevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 删去2/3
    {
        while (irrelevant_number < (irrelevant.size() / 3) * 2)
        {
            HSG::Erase(index, irrelevant[irrelevant_number]);
            deleted.insert(irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        while (relevant_number < 31347)
        {
            HSG::Erase(index, relevant[relevant_number]);
            deleted.insert(relevant[relevant_number]);
            ++relevant_number;
        }

        while (relevant_number < (relevant.size() / 3) * 2)
        {
            HSG::Erase(index, relevant[relevant_number]);
            deleted.insert(relevant[relevant_number]);
            ++relevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 全删
    {
        while (irrelevant_number < irrelevant.size())
        {
            HSG::Erase(index, irrelevant[irrelevant_number]);
            deleted.insert(irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        while (relevant_number < relevant.size())
        {
            HSG::Erase(index, relevant[relevant_number]);
            deleted.insert(relevant[relevant_number]);
            ++relevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    irrelevant_number = 0;
    relevant_number = 0;

    // 加回1/3
    {
        while (irrelevant_number < irrelevant.size() / 3)
        {
            HSG::Add(index, irrelevant[irrelevant_number], train[irrelevant[irrelevant_number]].data());
            deleted.erase(irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        while (relevant_number < relevant.size() / 3)
        {
            HSG::Add(index, relevant[relevant_number], train[relevant[relevant_number]].data());
            deleted.erase(relevant[relevant_number]);
            ++relevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 加回2/3
    {
        while (irrelevant_number < (irrelevant.size() / 3) * 2)
        {
            HSG::Add(index, irrelevant[irrelevant_number], train[irrelevant[irrelevant_number]].data());
            deleted.erase(irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        while (relevant_number < (relevant.size() / 3) * 2)
        {
            HSG::Add(index, relevant[relevant_number], train[relevant[relevant_number]].data());
            deleted.erase(relevant[relevant_number]);
            ++relevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    // 加回全部
    {
        while (irrelevant_number < irrelevant.size())
        {
            HSG::Add(index, irrelevant[irrelevant_number], train[irrelevant[irrelevant_number]].data());
            deleted.erase(irrelevant[irrelevant_number]);
            ++irrelevant_number;
        }

        while (relevant_number < relevant.size())
        {
            HSG::Add(index, relevant[relevant_number], train[relevant[relevant_number]].data());
            deleted.erase(relevant[relevant_number]);
            ++relevant_number;
        }

        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify_with_delete(train, test[i], neighbors[i], reference_answer[i], query_result, deleted, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
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

    if (name == "sift10M")
    {
        bvecs_vectors(argv[1], train, 10000000);
        bvecs_vectors(argv[2], test);
        ivecs(argv[3], neighbors);
    }
    else
    {
        train = load_vector(argv[1]);
        test = load_vector(argv[2]);
        neighbors = load_neighbors(argv[3]);
    }

    load_reference_answer(argv[4], reference_answer);
    load_deleted(argv[12], irrelevant);
    load_deleted(argv[13], relevant);

    std::cout << "delete irrelevant number: " << irrelevant.size() << std::endl;
    std::cout << "delete relevant number: " << relevant.size() << std::endl;

    short_edge_lower_limit = std::stoull(argv[6]);
    short_edge_upper_limit = std::stoull(argv[7]);
    cover_range = std::stoull(argv[8]);
    build_magnification = std::stoull(argv[9]);
    k = std::stoull(argv[10]);
    search_magnification = std::stoull(argv[11]);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);

    auto test_result =
        std::ofstream(std::format("result/HSG/DN-{0}-{1}-{2}-{3}-{4}-{5}.txt", name, UTC_time->tm_year + 1900,
                                  UTC_time->tm_mon + 1, UTC_time->tm_mday, UTC_time->tm_hour + 8, UTC_time->tm_min),
                      std::ios::app | std::ios::out);

    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("short edge lower limit: {0:<4}", short_edge_lower_limit) << std::endl;
    test_result << std::format("cover range: {0:<4}", cover_range) << std::endl;
    test_result << std::format("build magnification: {0:<4}", build_magnification) << std::endl;
    test_result << std::format("top k: {0:<4}", k) << std::endl;
    test_result << std::format("search magnification: {0:<4}", search_magnification) << std::endl;
    test_result << std::format("irrelevant number: {0:<9}", irrelevant.size()) << std::endl;
    test_result << std::format("relevant number: {0:<9}", relevant.size()) << std::endl;

    HSG::Index index(Space::Metric::Euclidean2, train[0].size(), short_edge_lower_limit, cover_range,
                     build_magnification);

    for (auto i = 0; i < train.size(); ++i)
    {
        HSG::Add(index, i, train[i].data());
    }

    {
        uint64_t total_hit = 0;
        uint64_t total_time = 0;

        for (auto i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            auto query_result = HSG::Search(index, test[i].data(), k, search_magnification);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            auto hit = verify(train, test[i], reference_answer[i], query_result, k);
            total_hit += hit;
        }

        auto cover_rate = HSG::Calculate_Coverage(index);
        test_result << std::format("cover rate: {0:<6.4} total hit: {1:<10} average time: {2:<10}us", cover_rate,
                                   total_hit, total_time / test.size())
                    << std::endl;
    }

    test_result.close();

    auto DB = std::thread(delete_both, index);

    DB.detach();

    auto DI = std::thread(delete_irrelevant, index);

    DI.detach();

    auto DR = std::thread(delete_relevant, index);

    DR.detach();
    done.acquire();
    return 0;
}
