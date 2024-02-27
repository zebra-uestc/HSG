#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "dehnsw.h"
#include "distance.h"

std::vector<std::vector<float>> get_reference_answer(const std::vector<std::vector<float>> &train,
                                                     const std::vector<std::vector<float>> &test,
                                                     const std::vector<std::vector<uint64_t>> &neighbors)
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
    return std::move(reference_answer);
}

uint64_t verify(const std::vector<std::vector<float>> &reference_answer, const uint64_t test,
                std::priority_queue<std::pair<float, uint64_t>> &query_result)
{
    auto result = std::vector<float>(query_result.size(), 0);
    while (!query_result.empty())
    {
        result[query_result.size() - 1] = query_result.top().first;
        query_result.pop();
    }
    uint64_t hit = 0;
    for (auto &distance : reference_answer[test])
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

void performence_test(const std::vector<std::vector<float>> &train, const std::vector<std::vector<float>> &test,
                      const std::vector<std::vector<uint64_t>> &neighbors,
                      const std::vector<std::vector<float>> &reference_answer)
{
    auto connects = std::vector<uint64_t>{4, 5, 6, 7, 8};
    auto steps = std::vector<uint64_t>{2, 3, 4, 5};
    auto querys = std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (auto &connect : connects)
    {
        for (auto &step : steps)
        {
            dehnsw::Index index(Distance_Type::Euclidean2, train[0].size(), connect, 128, step, 10000000);
            for (auto i = 0; i < train.size(); ++i)
            {
                dehnsw::insert(index, i, train[i].data());
            }
            auto deep_copy = index;
            for (auto &query : querys)
            {
                uint64_t total_hit = 0;
                uint64_t total_time = 0;
                for (auto i = 0; i < test.size(); ++i)
                {
                    auto begin = std::chrono::high_resolution_clock::now();
                    auto query_result = dehnsw::query(deep_copy, test[i].data(), neighbors[i].size(), query);
                    auto end = std::chrono::high_resolution_clock::now();
                    total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                    auto hit = verify(reference_answer, i, query_result);
                    total_hit += hit;
                }
                std::cout << total_hit << "       " << total_time / test.size() << std::endl;
            }
        }
    }
}

//// 压缩
// void compress(float *vector, uint64_t number, const float *result)
//{
//     auto *part1_pointer = vector;
//     auto *part2_pointer = vector + number;
//     const float *end = part2_pointer;
//     auto *result_pointer = result;
//     float __attribute__((aligned(32))) temporary_result[8];
//     __m256 vector1, vector2, sum;
//     while (part1_pointer < end)
//     {
//         vector1 = _mm256_loadu_ps(part1_pointer);
//         part1_pointer += 8;
//         vector2 = _mm256_loadu_ps(part2_pointer);
//         part2_pointer += 8;
//         sum = _mm256_add_ps(vector1, vector2);
//         _mm256_store_ps(temporary_result, sum);
//         memcpy((void *)result_pointer, temporary_result, sizeof(float) * 8);
//         result_pointer += 8;
//     }
// }
//
//// 计算误差
// float calculate_error(float *part1, uint64_t length)
//{
//     float *part1_pointer = part1;
//     float *part2_pointer = part1 + length;
//     const float *end = part2_pointer;
//     float __attribute__((aligned(32))) temporary_result[8];
//     __m256 vector1, vector2;
//     __m256 sum = _mm256_set1_ps(0);
//     while (part1_pointer < end)
//     {
//         vector1 = _mm256_loadu_ps(part1_pointer);
//         part1_pointer += 8;
//         vector2 = _mm256_loadu_ps(part2_pointer);
//         part2_pointer += 8;
//         sum = _mm256_add_ps(sum, _mm256_mul_ps(vector1, vector2));
//     }
//     _mm256_store_ps(temporary_result, sum);
//     float error = temporary_result[0] + temporary_result[1] + temporary_result[2] + temporary_result[3] +
//                   temporary_result[4] + temporary_result[5] + temporary_result[6] + temporary_result[7];
//     return error;
// }
//
// class T2
//{
//   public:
//     uint64_t dimension;
//     uint64_t avx;
//     std::vector<uint64_t> dimensions;
//     std::vector<uint64_t> compress_number;
//
//     explicit T2(uint64_t dimension) : dimension(dimension), avx(8)
//     {
//         //
//         auto calculate_number = dimension / avx;
//         if (dimension % avx != 0)
//         {
//             ++calculate_number;
//         }
//         this->dimensions.emplace_back(calculate_number * avx);
//         this->compress_number.emplace_back(0);
//         while (calculate_number != 1)
//         {
//             if (calculate_number % 2 == 0)
//             {
//                 calculate_number /= 2;
//                 this->dimensions.emplace_back(calculate_number * avx);
//                 this->compress_number.emplace_back(this->dimensions.back());
//             }
//             else
//             {
//                 calculate_number /= 2;
//                 this->dimensions.emplace_back(calculate_number * avx);
//                 this->compress_number.emplace_back(this->dimensions.back());
//                 this->dimensions.back() += avx;
//                 ++calculate_number;
//             }
//         }
//
//         for (auto i = 0; i < this->dimensions.size(); ++i)
//         {
//             std::cout << this->dimensions[i] << "  " << this->compress_number[i] << std::endl;
//         }
//     }
// };
//
// class T1
//{
//   public:
//     std::vector<std::vector<float>> data;
//     std::vector<float> error;
//
//     explicit T1(const std::vector<float> &v, const T2 &t2)
//     {
//         this->data.emplace_back(v);
//         //        memcpy(this->data[0].data(), v.data(), sizeof(float) * v.size());
//         this->error.emplace_back(0);
//         for (auto i = 1; i < t2.dimensions.size(); ++i)
//         {
//             this->data.emplace_back(t2.dimensions[i]);
//             compress(this->data[i - 1].data(), t2.compress_number[i], this->data[i].data());
//             this->error.emplace_back(2 * calculate_error(this->data[i - 1].data(), t2.compress_number[i]));
//         }
//     };
// };

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

    //    {
    //        auto t2 = T2(train[0].size());
    //        auto a = T1(test[0], t2);
    //        for (auto i = 0; i < 100; ++i)
    //        {
    //            auto begin = std::chrono::high_resolution_clock::now();
    //            auto t = T1(train[neighbors[0][i]], t2);
    //            auto end = std::chrono::high_resolution_clock::now();
    //            std::cout << "compress a vector consts "
    //                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "us"
    //                      << std::endl;
    //            for (auto j = 0; j < t2.dimensions.size(); ++j)
    //            {
    //                auto d = euclidean2::distance(a.data[j].data(), t.data[j].data(), t2.dimensions[j]) - a.error[j] -
    //                         t.error[j];
    //                std::cout << d << "  ";
    //            }
    //            std::cout << std::endl;
    //        }
    //    }

    auto reference_answer = get_reference_answer(train, test, neighbors);
    performence_test(train, test, neighbors, reference_answer);
    //    uint64_t query_relaxed_monotonicity = 1;
    //    if (argc > 4)
    //    {
    //        query_relaxed_monotonicity = std::stoull(argv[4]);
    //    }
    //    std::cout << "query relaxed monotonicity: " << query_relaxed_monotonicity << std::endl;
    //    dehnsw::Index index(Distance_Type::Euclidean2, train[0].size(), 4, 128, 4, 10000000);
    //    {
    //        uint64_t total_time = 0;
    //        for (auto i = 0; i < train.size(); ++i)
    //        {
    //            auto begin = std::chrono::high_resolution_clock::now();
    //            dehnsw::insert(index, i, train[i].data());
    //            auto end = std::chrono::high_resolution_clock::now();
    //            std::cout << "inserted ths " << i << "th vector, costs(us): "
    //                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    //            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //        }
    //        std::cout << "building index consts(us): " << total_time << std::endl;
    //    }
    //    {
    //        uint64_t total_hit = 0;
    //        uint64_t total_time = 0;
    //        for (auto i = 0; i < test.size(); ++i)
    //        {
    //            auto begin = std::chrono::high_resolution_clock::now();
    //            auto query_result = dehnsw::query(index, test[i].data(), neighbors[i].size(),
    //            query_relaxed_monotonicity); auto end = std::chrono::high_resolution_clock::now();
    //            //        std::cout << "one query costs(us): "
    //            //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<
    //            //                  std::endl;
    //            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //            auto hit = verify(reference_answer, i, query_result);
    //            total_hit += hit;
    //            //        std::cout << "hit: " << hit << std::endl;
    //        }
    //        std::cout << "average time: " << total_time / test.size() << std::endl;
    //        std::cout << "total hit: " << total_hit << std::endl;
    //    }
    //    {
    //        auto index_deep_copy = index;
    //        uint64_t total_hit = 0;
    //        uint64_t total_time = 0;
    //        for (auto i = 0; i < test.size(); ++i)
    //        {
    //            auto begin = std::chrono::high_resolution_clock::now();
    //            auto query_result =
    //                dehnsw::query(index_deep_copy, test[i].data(), neighbors[i].size(), query_relaxed_monotonicity);
    //            auto end = std::chrono::high_resolution_clock::now();
    //            //        std::cout << "one query costs(us): "
    //            //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<
    //            //                  std::endl;
    //            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //            auto hit = verify(reference_answer, i, query_result);
    //            total_hit += hit;
    //            //        std::cout << "hit: " << hit << std::endl;
    //        }
    //        std::cout << "average time: " << total_time / test.size() << std::endl;
    //        std::cout << "total hit: " << total_hit << std::endl;
    //    }
    //    {
    //        dehnsw::save(index, "./test_dehnsw_save");
    //        auto index1 = dehnsw::load("./test_dehnsw_save");
    //        uint64_t total_hit = 0;
    //        uint64_t total_time = 0;
    //        for (auto i = 0; i < test.size(); ++i)
    //        {
    //            auto begin = std::chrono::high_resolution_clock::now();
    //            auto query_result = dehnsw::query(index1, test[i].data(), neighbors[i].size(),
    //            query_relaxed_monotonicity); auto end = std::chrono::high_resolution_clock::now();
    //            //        std::cout << "one query costs(us): "
    //            //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<
    //            //                  std::endl;
    //            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    //            auto hit = verify(reference_answer, i, query_result);
    //            total_hit += hit;
    //            //        std::cout << "hit: " << hit << std::endl;
    //        }
    //        std::cout << "average time: " << total_time / test.size() << std::endl;
    //        std::cout << "total hit: " << total_hit << std::endl;
    //    }
    return 0;
}
