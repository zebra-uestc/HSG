#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "bruteforce.h"
#include "nnhnsw.h"

uint64_t verify(const std::vector<uint64_t> &result, const std::map<float, uint64_t> &query_result)
{
    uint64_t hit = 0;
    auto query_result_iterator = query_result.begin();
    for (auto &result_iterator : result)
    {
        if (result_iterator == query_result_iterator->second)
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
    auto vectors = load_vector(argv[1]);
    //    for (auto &i : vectors)
    //    {
    //        for (auto &j : i)
    //        {
    //            std::cout << j << "  ";
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout << vectors.size() << "  " << vectors.begin()->size() << std::endl;
    auto query = load_vector(argv[2]);
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
    nnhnsw::Index<float> index(vectors, Distance_Type::Euclidean2, 10, 1);
    for (auto i = 0; i < query.size(); ++i)
    {
        auto query_result = nnhnsw::query<float>(index, query[i], neighbors[i].size());
        auto hit = verify(neighbors[i], query_result);
        std::cout << "recall: " << hit / neighbors[i].size() << std::endl;
    }
    return 0;
}
