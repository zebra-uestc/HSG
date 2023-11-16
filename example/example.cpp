#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "bruteforce.h"
#include "nnhnsw.h"

// 每一“行”中，第一个数表示数据的维度dim，后面跟着的dim个数便是向量各维度的值。(注：fvecs中的f指float32)
// 因此，一“行”表示的便是一个向量。
// 那么行数是如何计算的呢？我们一般使用下面的式子：
// line_num = filesize / (dim + 1 ) / 4
// 其中line_num是行数，filesize是文件字节数，dim+1指每个向量占用的数值个数，4指每4个字节存储一个数值。
// ivecs
// ivecs其实和fvecs的格式是一样的，只不过它存储的不是向量，而是每一条查询的答案。
// 就是说，ivecs里的每一“行”里，第一个数据是查询答案的数量n，后面n个数是答案向量的id。(注：ivecs中的i指int32)
std::vector<std::vector<float>> load_vector_from_fvecs(const char *filename)
{
    std::fstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    // 向量维度
    uint64_t dimension = 0;
    // 读取向量维度
    file.read((char *)&dimension, 4);
    // 移动文件读指针到文件末尾
    file.seekg(0, std::ios::end);
    // 此时读指针偏移量即为文件大小
    size_t file_size = file.tellg();
    // 计算向量个数
    //  文件大小 / (维度 + 1) / 4
    uint64_t number = (file_size / (dimension + 1) / 4);
    std::vector<std::vector<float>> vector(number, std::vector<float>(dimension));
    // 重置读指针到文件开头
    file.seekg(0, std::ios::beg);
    for (auto i = 0; i < number; ++i)
    {
        file.seekg(4, std::ios::cur);
        file.read((char *)vector[i].data(), dimension * 4);
    }
    file.close();
    // if (vector.size() != 0)
    // {
    //     std::cout << "vector number: " << vector.size() << std::endl
    //               << "vector dimension: " << vector[0].size() << std::endl;
    // }
    // for (auto i = 0; i < vector.size(); ++i)
    // {
    //     for (auto j = 0; j < vector[i].size(); ++j)
    //     {
    //         std::cout << vector[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    return vector;
}

std::vector<std::pair<int32_t, std::vector<int32_t>>> load_result_from_ivecs(const char *filename)
{
    std::fstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    std::vector<std::pair<int32_t, std::vector<int32_t>>> result;
    while (!file.eof())
    {
        // 结果数量
        int32_t number = 0;
        file.read((char *)&number, 4);
        // 结果
        std::vector<int32_t> temporary(number);
        file.read((char *)temporary.data(), number * 4);
        result.emplace_back(number, temporary);
    }
    file.close();
    // for (auto i = 0; i < result.size(); ++i)
    // {
    //     std::cout << "result number: " << result[i].first << std::endl << "result: ";
    //     for (auto j = 0; j < result[i].first; ++j)
    //     {
    //         std::cout << result[i].second[j] << "  ";
    //     }
    //     std::cout << std::endl;
    // }
    return result;
}

uint64_t verify(const std::pair<int32_t, std::vector<int32_t>> &result, const std::map<float, int32_t> &query_result)
{
    uint64_t hit = 0;
    auto query_result_iteration = query_result.begin();
    for (auto i = 0; i < result.first; ++i)
    {
        if (result.second[i] == query_result_iteration->second)
        {
            ++hit;
            ++query_result_iteration;
        }
    }
    // {
    //     for (auto i = 0; i < result.second.size(); ++i)
    //     {
    //         std::cout << result.second[i] << "  ";
    //     }
    //     std::cout << std::endl;
    //     auto query_result_iteration = query_result.begin();
    //     for (auto i = 0; i < result.second.size(); ++i)
    //     {
    //         std::cout << query_result_iteration.operator->()->second << "  ";
    //         ++query_result_iteration;
    //     }
    //     std::cout << std::endl;
    // }
    return hit;
}

uint64_t verify(const std::pair<int32_t, std::vector<int32_t>> &result, const std::map<float, uint64_t> &query_result)
{
    uint64_t hit = 0;
    auto query_result_iteration = query_result.begin();
    for (auto i = 0; i < result.first; ++i)
    {
        if (result.second[i] == query_result_iteration->second)
        {
            ++hit;
            ++query_result_iteration;
        }
    }
    // {
    //     for (auto i = 0; i < result.second.size(); ++i)
    //     {
    //         std::cout << result.second[i] << "  ";
    //     }
    //     std::cout << std::endl;
    //     auto query_result_iteration = query_result.begin();
    //     for (auto i = 0; i < result.second.size(); ++i)
    //     {
    //         std::cout << query_result_iteration.operator->()->second << "  ";
    //         ++query_result_iteration;
    //     }
    //     std::cout << std::endl;
    // }
    return hit;
}

// class T
//{
//   public:
//     std::string s;
//     explicit T(const std::string &s)
//     {
//         this->s = s;
//     }
// };

int main(int argc, char **argv)
{
    auto vectors = load_vector_from_fvecs(argv[1]);
    auto query = load_vector_from_fvecs(argv[2]);
    auto result = load_result_from_ivecs(argv[3]);
    //    for (auto i = 0; i < query.size(); ++i)
    //    {
    //        auto query_result = bruteforce::search<float>(vectors, query[i], result[i].first);
    //        auto hit = verify(result[i], query_result);
    //        std::cout << "recall: " << (double)hit / result[i].first << std::endl;
    //    }
    nnhnsw::Index<float> index(vectors, Distance_Type::Euclidean2, 10, 1);
    for (auto i = 0; i < query.size(); ++i)
    {
        auto query_result = index.query(query[i], result[i].first);
        auto hit = verify(result[i], query_result);
        std::cout << "recall: " << (double)hit / result[i].first << std::endl;
    }
    return 0;
}
