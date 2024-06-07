//
// Created by 李昭 on 2024/6/6.
//
#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>
#include <../..//IndexNSG.h>
#include <../../distances.h>
#include <../../random.h>


#include "universal.h"

//向量中的每一个元素都是一个样本，样本中每一个元素都是该样本的一个特征值
std::vector<std::vector<float>> train;
std::vector<std::vector<float>> test;
std::vector<std::vector<uint64_t>> neighbors;
//参考答案
std::vector<std::vector<float>> reference_answer;

std::string name;

auto available_thread = std::counting_semaphore<>(12);
auto done_semaphore = std::counting_semaphore<>(1);
uint64_t done_thread = 0;
uint64_t done_number = 0;
auto done = std::counting_semaphore<>(0);

int dim = train[0].size();
int elements_number = train.size();
const int k_nerghbors = 100;       // 每个节点的邻居数量
const faiss::MetricType metric = faiss::METRIC_L2; // 度量类型，这里是欧氏距离

void test_nsg()
{
    auto test_result = std::ofstream(std::format("result/nsg/{0}.txt", name),
                                     std::ios::app | std::ios::out);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    faiss::IndexNSG *index = new faiss::IndexNSG(dim,neighboors_number,metric);
    index->add(max_elements,train);

    // Query the elements for themselves and measure recall
    for (int i = 0; i < test.size(); ++i)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        // 准备接收搜索结果的缓冲区
        std::vector<float> distances(test.size() * k_nerghbors);
        std::vector<idx_t> labels(test.size() * k_nerghbors);
        // 执行搜索
        index->search(1, test[i].data(), k_nerghbors, distances.data(), labels.data());
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        total_hit += verify(train, test[i], reference_answer[i], result);
    }

    test_result << std::format("total hit: {1:<10} average time: {2:<10}us", total_hit,
                               total_time / test.size())
                << std::endl;
    delete index;
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
            auto one_thread = std::thread(test_nsg(), M, ef_construction);
            one_thread.detach();
        }
    }

    done.acquire();
    return 0;
}
