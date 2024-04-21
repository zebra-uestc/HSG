#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "distance.h"
#include "miluann.h"

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

void base_test(uint64_t short_edge_bound, uint64_t build_magnification, uint64_t search_magnification)
{
    auto test_result = std::ofstream(std::format("detail-analysis.txt", short_edge_bound, build_magnification),
                                     std::ios::app | std::ios::out);

    auto time = std::time(nullptr);
    auto UTC_time = std::gmtime(&time);
    test_result << UTC_time->tm_year + 1900 << "年" << UTC_time->tm_mon + 1 << "月" << UTC_time->tm_mday << "日"
                << UTC_time->tm_hour + 8 << "时" << UTC_time->tm_min << "分" << UTC_time->tm_sec << "秒" << std::endl;

    test_result << "short edge bound: " << short_edge_bound << std::endl;
    test_result << "build magnification: " << build_magnification << std::endl;
    test_result << "search magnification: " << search_magnification << std::endl;

    miluann::Index index(Distance_Type::Euclidean2, train[0].size(), short_edge_bound, build_magnification, 1.2);

    for (auto i = 0; i < train.size(); ++i)
    {
        miluann::add(index, i, train[i]);
    }

    auto times = std::vector<uint64_t>(27, 0);
    for (auto i = 0; i < test.size(); ++i)
    {
        auto query_result = miluann::search(times, index, test[i], neighbors[i].size(), search_magnification);
    }
    for (auto i = 0; i < times.size(); ++i)
    {
        test_result << times[i] << std::endl;
    }

    test_result.close();
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

    base_test(std::stoul(argv[4]), std::stoul(argv[5]), std::stoul(argv[6]));

    return 0;
}
