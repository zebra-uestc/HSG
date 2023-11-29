#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "bruteforce.h"
#include "nnhnsw.h"

uint64_t verify(const std::vector<uint64_t> &neighbors, const std::map<float, uint64_t> &query_result)
{
    uint64_t hit = 0;
    auto query_result_iterator = query_result.begin();
    for (auto &neighbor : neighbors)
    {
        if (neighbor == query_result_iterator->second)
        {
            ++hit;
            ++query_result_iterator;
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
    auto begin = std::chrono::high_resolution_clock::now();
    nnhnsw::Index<float> index(train, Distance_Type::Euclidean2, 2, 1, 10, 10);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "building index costs(us): "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    uint64_t total_hit = 0;
    for (auto i = 0; i < test.size(); ++i)
    {
        begin = std::chrono::high_resolution_clock::now();
        auto query_result = nnhnsw::query<float>(index, test[i], neighbors[i].size());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "one query costs(us): "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
        auto hit = verify(neighbors[i], query_result);
        total_hit += hit;
        std::cout << "hit: " << hit << std::endl;
    }
    std::cout << "total hit: " << total_hit << std::endl;

    // debug
    //    {
    //        for (auto i = 0; i < 10; ++i)
    //        {
    //            auto begin = std::chrono::high_resolution_clock::now();
    //            auto query_result = nnhnsw::query<float>(index, train[i], 100);
    //            auto end = std::chrono::high_resolution_clock::now();
    //            std::cout << "one query costs(us): "
    //                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    //            auto hit = verify(std::vector<uint64_t>(1, i), query_result);
    //            total_hit += hit;
    //            std::cout << "hit: " << hit << std::endl;
    //        }
    //        {
    //            std::cout << "total hit: " << total_hit << std::endl;
    //        }
    //    }
    // debug

    return 0;
}
