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
    //    std::cout << count << "  " << dimension << std::endl;
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
    //    std::cout << count << "  " << neighbor_count << std::endl;
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
    //    for (auto &i : vectors)
    //    {
    //        for (auto &j : i)
    //        {
    //            std::cout << j << "  ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << vectors.size() << "  " << vectors.begin()->size() << std::endl;
    auto test = load_vector(argv[2]);
    //    for (auto &i : query)
    //    {
    //        for (auto &j : i)
    //        {
    //            std::cout << j << "  ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << query.size() << "  " << query.begin()->size() << std::endl;
    auto neighbors = load_neighbors(argv[3]);
    //    for (auto &i : neighbors)
    //    {
    //        for (auto &j : i)
    //        {
    //            std::cout << j << "  ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << neighbors.size() << "  " << neighbors.begin()->size() << std::endl;
    //    for (int i = 0; i < query.size(); ++i)
    //    {
    //        auto begin = std::chrono::high_resolution_clock::now();
    //        auto query_result = bruteforce::search<float>(vectors, query[i], neighbors[i].size());
    //        auto end = std::chrono::high_resolution_clock::now();
    //        std::cout << "one query(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    //        begin).count()
    //                  << std::endl;
    //        auto hit = verify(neighbors[i], query_result);
    //        std::cout << "recall: " << hit << std::endl;
    //    }
    auto begin = std::chrono::high_resolution_clock::now();
    nnhnsw::Index<float> index(train, Distance_Type::Euclidean2, 20, 1);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "building index costs(us): "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    auto hit_results = std::vector<uint64_t>();
    for (auto i = 0; i < test.size(); ++i)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        auto query_result = nnhnsw::query<float>(index, test[i], 10);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "one query costs(us): "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
        auto hit = verify(neighbors[i], query_result);
        hit_results.push_back(hit);
        std::cout << "hit: " << hit << std::endl;
    }
    {
        uint64_t total_hit = 0;
        for (auto &hit : hit_results)
        {
            total_hit += hit;
        }
        std::cout << "total hit: " << total_hit << std::endl;
    }
    //    for (auto i = 0; i < 10; ++i)
    //    {
    //        auto begin = std::chrono::high_resolution_clock::now();
    //        auto query_result = nnhnsw::query<float>(index, train[i], 10);
    //        auto end = std::chrono::high_resolution_clock::now();
    //        std::cout << "one query costs(us): "
    //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    //        auto hit = verify(std::vector<uint64_t>(1, i), query_result);
    //        hit_results.push_back(hit);
    //        std::cout << "hit: " << hit << std::endl;
    //    }
    //    {
    //        uint64_t total_hit = 0;
    //        for (auto &hit : hit_results)
    //        {
    //            total_hit += hit;
    //        }
    //        std::cout << "total hit: " << total_hit << std::endl;
    //    }
    return 0;
}
