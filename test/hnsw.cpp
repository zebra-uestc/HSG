#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

#include "../../hnswlib/hnswlib/hnswlib.h"
#include "distance.h"

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
                euclidean2::distance(test[i].data(), train[neighbors[i][j]].data(), train[0].size());
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
            euclidean2::distance(test[t].data(), train[query_result.top().second].data(), train[0].size());
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

void test_hnsw(uint64_t M, uint64_t ef_construction)
{
    auto test_result =
        std::ofstream(std::format("result/hnsw/{0}-{1}.txt", M, ef_construction), std::ios::app | std::ios::out);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << std::format("M: {0:>4} {1:>4}", M, ef_construction) << std::endl;
    std::vector<uint64_t> p{10, 20, 40, 80, 120, 200, 400, 600, 800};
    test_result << "p: [" << p[0];
    for (auto i = 1; i < p.size(); ++i)
    {
        test_result << ", " << p[i];
    }
    test_result << "]" << std::endl;

    int dim = train[0].size();
    int max_elements = train.size();
    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Add data to index
    uint64_t total_time = 0;
    for (int i = 0; i < max_elements; i++)
    {
        // auto begin = std::chrono::high_resolution_clock::now();
        alg_hnsw->addPoint(train[i].data(), i);
        // auto end = std::chrono::high_resolution_clock::now();
        // total_time += std::chrono::duration_cast<std::chrono::microseconds>(end
        // - begin).count();
    }
    // std::cout << "build average time: " << total_time / train.size() <<
    // std::endl; std::cout << "build time: " << total_time << std::endl;

    for (auto &pp : p)
    {
        uint64_t total_hit = 0;
        uint64_t total_time = 0;
        alg_hnsw->setEf(pp);
        // Query the elements for themselves and measure recall
        for (int i = 0; i < test.size(); ++i)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(test[i].data(), 100);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            total_hit += verify(i, result);
        }
        // std::cout << "query average time: " << total_time / test.size() <<
        // std::endl; std::cout << "total hit: " << total_hit << std::endl;
        // std::cout << total_hit << "    " << total_time / test.size() << std::endl;
        test_result << std::format("total hit: {0:>13} average time: {1:>13}us", total_hit, total_time / test.size())
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
/*
 * 测试比较不同参数下的HNSW算法性能*/
int main(int argc, char **argv)
{/*
* 检查CPU是否支持SIMD指令集（AVX512、AVX、SSE）*/
#if defined(__AVX512F__)
    std::cout << "AVX512 supported. " << std::endl;
#elif defined(__AVX__)
    std::cout << "AVX supported. " << std::endl;
#elif defined(__SSE__)
    std::cout << "SSE supported. " << std::endl;
#else
    std::cout << "no SIMD supported. " << std::endl;
#endif
    /*输出CPU的物理单元数量*/
    std::cout << "CPU physical units: " << std::thread::hardware_concurrency() << std::endl;
/*加载向量和邻居*/
    train = load_vector(argv[1]);
    test = load_vector(argv[2]);
    neighbors = load_neighbors(argv[3]);
    reference_answer = get_reference_answer();
/*参数设置*/
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
