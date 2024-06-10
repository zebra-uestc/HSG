#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

#include "../../hnswlib/hnswlib/hnswlib.h"
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

void test_hnsw(uint64_t M, uint64_t ef_construction)
{
    auto test_result = std::ofstream(std::format("result/hnsw/{0}-{1}-{2}.txt", name, M, ef_construction),
                                     std::ios::app | std::ios::out);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("M: {0:<4} ef: {1:<4}", M, ef_construction) << std::endl;
    std::vector<uint64_t> efs{10, 20, 40, 80, 120, 200, 400, 600, 800};
    test_result << "ef: [" << efs[0];

    for (auto i = 1; i < efs.size(); ++i)
    {
        test_result << ", " << efs[i];
    }

    test_result << "]" << std::endl;

    int dim = train[0].size();
    int max_elements = train.size();
    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Add data to index
    for (auto i = 0; i < max_elements; ++i)
    {
        alg_hnsw->addPoint(train[i].data(), i);
    }

    for (auto &ef : efs)
    {
        uint64_t total_hit = 0;
        uint64_t total_time = 0;
        alg_hnsw->setEf(ef);

        // Query the elements for themselves and measure recall
        for (int i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(test[i].data(), 100);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            total_hit += verify(train, test[i], reference_answer[i], result, 100);
        }

        test_result << std::format("ef: {0:<4} total hit: {1:<10} average time: {2:<10}us", ef, total_hit,
                                   total_time / test.size())
                    << std::endl;
    }

    delete alg_hnsw;
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

    std::vector<uint64_t> Ms{4, 8, 12, 16, 24, 36, 48, 64, 96};
    std::vector<uint64_t> ef_constructions{500};
    done_number += Ms.size() * ef_constructions.size();

    for (auto &M : Ms)
    {
        for (auto &ef_construction : ef_constructions)
        {
            available_thread.acquire();
            auto one_thread = std::thread(test_hnsw, M, ef_construction);
            one_thread.detach();
        }
    }

    done.acquire();
    return 0;
}
