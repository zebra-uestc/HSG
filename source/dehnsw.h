#pragma once

#include <chrono>
#include <cinttypes>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory.h>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "distance.h"

namespace dehnsw
{

// 向量
template <typename Dimension_Type> class Vector
{
  public:
    // 向量的最大层数
    uint64_t layer{};
    // 向量在子索引中的偏移量
    uint64_t offset{};
    // 向量在整个数据集中的偏移量
    uint64_t global_offset{};
    // 向量的原始数据
    std::vector<Dimension_Type> data;
    // 指向的邻居向量
    std::vector<std::map<float, uint64_t>> out;
    // 指向该向量的邻居向量
    std::vector<std::unordered_set<uint64_t>> in;

    explicit Vector(const uint64_t global_offset, const uint64_t offset, const std::vector<Dimension_Type> &data)
        : layer(0), offset(offset), global_offset(global_offset), data(data)
    {
        this->out.emplace_back();
        this->in.emplace_back();
    }
};

// 索引
template <typename Dimension_Type> class Sub_Index
{
  public:
    // 子索引中向量的数量
    uint64_t count{};
    // 子索引最大层数
    uint64_t layer_count{};
    // 最高层中的向量的偏移量
    uint64_t vector_in_highest_layer{};
    // 子索引中的向量
    std::vector<Vector<Dimension_Type>> vectors;

    explicit Sub_Index(const uint64_t max_count_bound) : count(0), layer_count(0), vector_in_highest_layer(0)
    {
        this->vectors.reserve(max_count_bound);
    }
};

class Index_Parameters
{
  public:
    // 步长
    uint64_t step{};
    // 每个子索引中向量的数量限制
    uint64_t sub_index_bound{};
    // 距离类型
    Distance_Type distance_type;
    // 插入向量时提前终止条件
    uint64_t relaxed_monotonicity{};
    // 每个向量的邻居个数
    uint64_t minimum_connect_number{};

    explicit Index_Parameters(const uint64_t step, const uint64_t sub_index_bound, const Distance_Type distance_type,
                              const uint64_t relaxed_monotonicity, const uint64_t minimum_connect_number)
        : step(step), sub_index_bound(sub_index_bound), distance_type(distance_type),
          relaxed_monotonicity(relaxed_monotonicity), minimum_connect_number(minimum_connect_number)
    {
    }
};

// 索引
template <typename Dimension_Type> class Index
{
  public:
    // 索引中向量的数量
    uint64_t count{};
    // 索引的参数
    Index_Parameters parameters;
    // 子索引
    std::vector<Sub_Index<Dimension_Type>> sub_indexes;
    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

    explicit Index(const Distance_Type distance_type, const uint64_t minimum_connect_number,
                   const uint64_t relaxed_monotonicity, const uint64_t step, const uint64_t sub_index_bound)
        : count(0), parameters(step, sub_index_bound, distance_type, relaxed_monotonicity, minimum_connect_number)
    {
        this->distance_calculation = get_distance_calculation_function<Dimension_Type>(distance_type);
    }
};

namespace
{

template <typename Dimension_Type>
bool connected(const Index<Dimension_Type> &index, const Sub_Index<Dimension_Type> &sub_index,
               const uint64_t layer_number, const uint64_t start,
               std::unordered_map<uint64_t, std::pair<float, uint64_t>> &deleted_edges)
{
    auto last = std::unordered_set<uint64_t>();
    auto next = std::unordered_set<uint64_t>();
    auto flag = std::unordered_set<uint64_t>();
    flag.insert(start);
    last.insert(start);
    for (auto round = 0; round < 4; ++round)
    {
        for (const auto &last_vector_offset : last)
        {
            for (const auto &neighbor : sub_index.vectors[last_vector_offset].out[layer_number])
            {
                deleted_edges.erase(neighbor.second);
                if (flag.insert(neighbor.second).second)
                {
                    next.insert(neighbor.second);
                }
            }
            for (const auto &neighbor_offset : sub_index.vectors[last_vector_offset].in[layer_number])
            {
                deleted_edges.erase(neighbor_offset);
                if (flag.insert(neighbor_offset).second)
                {
                    next.insert(neighbor_offset);
                }
            }
        }
        if (deleted_edges.empty())
        {
            return true;
        }
        std::swap(last, next);
        next.clear();
    }
    return false;
}

template <typename Dimension_Type>
bool insert_to_upper_layer(const Index<Dimension_Type> &index, const Sub_Index<Dimension_Type> &sub_index,
                           const uint64_t layer_number, const uint64_t vector_offset)
{
    auto last = std::unordered_set<uint64_t>();
    auto next = std::unordered_set<uint64_t>();
    auto flag = std::unordered_set<uint64_t>();
    last.insert(vector_offset);
    for (auto round = 0; round < index.parameters.step; ++round)
    {
        for (const auto &last_vector_offset : last)
        {
            for (auto &neighbor : sub_index.vectors[last_vector_offset].out[layer_number])
            {
                if (flag.insert(neighbor.second).second)
                {
                    if (layer_number < sub_index.vectors[neighbor.second].layer)
                    {
                        return false;
                    }
                    next.insert(neighbor.second);
                }
            }
            for (auto &neighbor_vector_offset : sub_index.vectors[last_vector_offset].in[layer_number])
            {
                if (flag.insert(neighbor_vector_offset).second)
                {
                    if (layer_number < sub_index.vectors[neighbor_vector_offset].layer)
                    {
                        return false;
                    }
                    next.insert(neighbor_vector_offset);
                }
            }
        }
        std::swap(last, next);
        next.clear();
    }
    return true;
}

// 从开始向量查询距离目标向量最近的"最小连接数"个向量
template <typename Dimension_Type>
std::map<float, uint64_t> nearest_neighbors_insert(const Index<Dimension_Type> &index,
                                                   const Sub_Index<Dimension_Type> &sub_index,
                                                   const uint64_t layer_number,
                                                   const std::vector<Dimension_Type> &query_vector,
                                                   const uint64_t start)
{
    // 优先队列
    auto nearest_neighbors = std::map<float, uint64_t>();
    // 标记簇中的向量是否被遍历过
    std::unordered_set<uint64_t> flags;
    uint64_t out_of_bound = 0;
    // 排队队列
    auto waiting_vectors = std::map<float, uint64_t>();
    waiting_vectors.insert(
        std::make_pair(index.distance_calculation(query_vector, sub_index.vectors[start].data), start));
    while (!waiting_vectors.empty())
    {
        auto processing_distance = waiting_vectors.begin()->first;
        auto processing_vector_offset = waiting_vectors.begin()->second;
        waiting_vectors.erase(waiting_vectors.begin());
        flags.insert(processing_vector_offset);
        // 如果已遍历的向量小于候选数量
        if (nearest_neighbors.size() < index.parameters.minimum_connect_number)
        {
            nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_offset));
        }
        else
        {
            // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
            if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
            {
                out_of_bound = 0;
                nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_offset));
                nearest_neighbors.erase(std::prev(nearest_neighbors.end()));
            }
            else if (index.parameters.relaxed_monotonicity < out_of_bound)
            {
                break;
            }
            else
            {
                ++out_of_bound;
            }
        }
        // 计算当前向量的出边指向的向量和目标向量的距离
        for (auto &vector : sub_index.vectors[processing_vector_offset].out[layer_number])
        {
            if (flags.insert(vector.second).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector, sub_index.vectors[vector.second].data), vector.second));
            }
        }
        // 计算当前向量的入边指向的向量和目标向量的距离
        for (auto &vector_offset : sub_index.vectors[processing_vector_offset].in[layer_number])
        {
            if (flags.insert(vector_offset).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector, sub_index.vectors[vector_offset].data), vector_offset));
            }
        }
    }
    return nearest_neighbors;
}

// 从开始向量查询距离目标向量最近的top-k个向量
// 该函数查询的是除最后一层外其它层中的最近邻居，所以返回的结果为向量在子索引中的偏移量
template <typename Dimension_Type>
std::map<float, uint64_t> nearest_neighbors_query(const Index<Dimension_Type> &index,
                                                  const Sub_Index<Dimension_Type> &sub_index,
                                                  const uint64_t layer_number,
                                                  const std::vector<Dimension_Type> &query_vector, const uint64_t start,
                                                  const uint64_t top_k, const uint64_t relaxed_monotonicity)
{
    // 优先队列
    auto nearest_neighbors = std::map<float, uint64_t>();
    // 标记簇中的向量是否被遍历过
    std::unordered_set<uint64_t> flags;
    uint64_t out_of_bound = 0;
    // 排队队列
    auto waiting_vectors = std::map<float, uint64_t>();
    waiting_vectors.insert(
        std::make_pair(index.distance_calculation(query_vector, sub_index.vectors[start].data), start));
    while (!waiting_vectors.empty())
    {
        auto processing_distance = waiting_vectors.begin()->first;
        auto processing_vector_offset = waiting_vectors.begin()->second;
        waiting_vectors.erase(waiting_vectors.begin());
        flags.insert(processing_vector_offset);
        // 如果已遍历的向量小于候选数量
        if (nearest_neighbors.size() < top_k)
        {
            nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_offset));
        }
        else
        {
            // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
            if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
            {
                out_of_bound = 0;
                nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_offset));
                nearest_neighbors.erase(std::prev(nearest_neighbors.end()));
            }
            else if (relaxed_monotonicity < out_of_bound)
            {
                break;
            }
            else
            {
                ++out_of_bound;
            }
        }
        // 计算当前向量的出边指向的向量和目标向量的距离
        for (auto &vector : sub_index.vectors[processing_vector_offset].out[layer_number])
        {
            if (flags.insert(vector.second).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector, sub_index.vectors[vector.second].data), vector.second));
            }
        }
        // 计算当前向量的入边指向的向量和目标向量的距离
        for (auto &vector_offset : sub_index.vectors[processing_vector_offset].in[layer_number])
        {
            if (flags.insert(vector_offset).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector, sub_index.vectors[vector_offset].data), vector_offset));
            }
        }
    }
    return nearest_neighbors;
}

// 从开始向量查询距离目标向量最近的top-k个向量
// 该函数查询的是最后一层中的邻居，所以返回的结果为向量的全局偏移量
template <typename Dimension_Type>
std::map<float, uint64_t> nearest_neighbors_last_layer(const Index<Dimension_Type> &index,
                                                       const Sub_Index<Dimension_Type> &sub_index,
                                                       const uint64_t layer_number,
                                                       const std::vector<Dimension_Type> &query_vector,
                                                       const uint64_t start, const uint64_t top_k,
                                                       const uint64_t relaxed_monotonicity, const float distance_bound)
{
    // 优先队列
    auto nearest_neighbors = std::map<float, uint64_t>();
    // 标记簇中的向量是否被遍历过
    std::unordered_set<uint64_t> flags;
    uint64_t out_of_bound = 0;
    // 排队队列
    auto waiting_vectors = std::map<float, uint64_t>();
    waiting_vectors.insert(
        std::make_pair(index.distance_calculation(query_vector, sub_index.vectors[start].data), start));
    while (!waiting_vectors.empty())
    {
        auto processing_distance = waiting_vectors.begin()->first;
        auto processing_vector_offset = waiting_vectors.begin()->second;
        waiting_vectors.erase(waiting_vectors.begin());
        flags.insert(processing_vector_offset);
        if (processing_distance < distance_bound)
        {
            // 如果已遍历的向量小于候选数量
            if (nearest_neighbors.size() < top_k)
            {
                nearest_neighbors.insert(
                    std::make_pair(processing_distance, sub_index.vectors[processing_vector_offset].global_offset));
            }
            else
            {
                // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
                if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
                {
                    out_of_bound = 0;
                    nearest_neighbors.insert(
                        std::make_pair(processing_distance, sub_index.vectors[processing_vector_offset].global_offset));
                    nearest_neighbors.erase(std::prev(nearest_neighbors.end()));
                }
                else if (relaxed_monotonicity < out_of_bound)
                {
                    break;
                }
                else
                {
                    ++out_of_bound;
                }
            }
        }
        else
        {
            if (relaxed_monotonicity < out_of_bound)
            {
                break;
            }
            else
            {
                ++out_of_bound;
            }
        }
        // 计算当前向量的出边指向的向量和目标向量的距离
        for (auto &vector : sub_index.vectors[processing_vector_offset].out[layer_number])
        {
            if (flags.insert(vector.second).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector, sub_index.vectors[vector.second].data), vector.second));
            }
        }
        // 计算当前向量的入边指向的向量和目标向量的距离
        for (auto &vector_offset : sub_index.vectors[processing_vector_offset].in[layer_number])
        {
            if (flags.insert(vector_offset).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector, sub_index.vectors[vector_offset].data), vector_offset));
            }
        }
    }
    return nearest_neighbors;
}

template <typename Dimension_Type>
void add(Index<Dimension_Type> &index, Sub_Index<Dimension_Type> &sub_index, Vector<Dimension_Type> &new_vector,
         uint64_t target_layer_number)
{
    // 记录被插入向量每一层中距离最近的max_connect个邻居向量
    auto every_layer_neighbors = std::stack<std::map<float, uint64_t>>();
    every_layer_neighbors.push(nearest_neighbors_insert(index, sub_index, sub_index.layer_count, new_vector.data,
                                                        sub_index.vector_in_highest_layer));
    // 逐层扫描
    // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
    // 所以不需要实际的层和簇
    // 直接通过上一层中返回的结果即可进行计算
    for (int64_t layer_number = sub_index.layer_count - 1; 0 <= layer_number; --layer_number)
    {
        every_layer_neighbors.push(nearest_neighbors_insert(index, sub_index, layer_number, new_vector.data,
                                                            every_layer_neighbors.top().begin()->second));
    }
    // 插入向量
    while (!every_layer_neighbors.empty())
    {
        auto deleted_edges = std::unordered_map<uint64_t, std::pair<float, uint64_t>>();
        for (const auto &neighbor : every_layer_neighbors.top())
        {
            auto &neighbor_vector = sub_index.vectors[neighbor.second];
            // 将新的向量指向邻居向量
            new_vector.out[target_layer_number].insert(neighbor);
            // 在邻居向量中记录指向自己的新向量
            neighbor_vector.in[target_layer_number].insert(new_vector.offset);
            // 新向量和邻居向量的距离小于邻居向量已指向的10个向量的距离
            if (neighbor_vector.out[target_layer_number].size() < index.parameters.minimum_connect_number ||
                neighbor_vector.out[target_layer_number].upper_bound(neighbor.first) !=
                    neighbor_vector.out[target_layer_number].end())
            {
                neighbor_vector.out[target_layer_number].insert(std::make_pair(neighbor.first, new_vector.offset));
                new_vector.in[target_layer_number].insert(neighbor.second);
            }
            // 如果邻居向量的出度大于最大出度数量
            if (index.parameters.minimum_connect_number < neighbor_vector.out[target_layer_number].size())
            {
                auto temporary = neighbor_vector.out[target_layer_number].begin();
                std::advance(temporary, index.parameters.minimum_connect_number);
                auto record = *temporary;
                deleted_edges.insert(std::make_pair(record.second, std::make_pair(record.first, neighbor.second)));
                neighbor_vector.out[target_layer_number].erase(temporary);
                sub_index.vectors[record.second].in[target_layer_number].erase(neighbor.second);
            }
        }
        if (!connected(index, sub_index, target_layer_number, new_vector.offset, deleted_edges))
        {
            for (const auto &edge : deleted_edges)
            {
                sub_index.vectors[edge.second.second].out[target_layer_number].insert(
                    std::make_pair(edge.second.first, edge.first));
                sub_index.vectors[edge.first].in[target_layer_number].insert(edge.second.second);
            }
        }
        // 如果新向量应该被插入上一层中
        if (insert_to_upper_layer(index, sub_index, target_layer_number, new_vector.offset))
        {
            every_layer_neighbors.pop();
            ++target_layer_number;
            if (sub_index.layer_count < target_layer_number)
            {
                sub_index.layer_count = target_layer_number;
                sub_index.vector_in_highest_layer = new_vector.offset;
            }
            ++new_vector.layer;
            new_vector.out.emplace_back();
            new_vector.in.emplace_back();
        }
        else
        {
            break;
        }
    }
}

} // namespace

// 查询
template <typename Dimension_Type>
std::map<float, uint64_t> query(const Index<Dimension_Type> &index, const std::vector<Dimension_Type> &query_vector,
                                uint64_t top_k, uint64_t relaxed_monotonicity = 0)
{
    //    if (index.vectors.empty())
    //    {
    //        throw std::logic_error("Empty vectors in index. ");
    //    }
    //    if (query_vector.size() != index.vectors[0].data.size())
    //    {
    //        throw std::invalid_argument("The dimension of query vector is not "
    //                                    "equality with vectors in index. ");
    //    }
    if (relaxed_monotonicity == 0)
    {
        relaxed_monotonicity = top_k;
    }
    auto result = std::map<float, uint64_t>();
    float distance_bound = MAXFLOAT;
    auto one_sub_index_result = std::map<float, uint64_t>();
    for (const auto &sub_index : index.sub_indexes)
    {
        //        auto begin = std::chrono::high_resolution_clock::now();
        one_sub_index_result.emplace(
            index.distance_calculation(query_vector, sub_index.vectors[sub_index.vector_in_highest_layer].data),
            sub_index.vector_in_highest_layer);
        if (sub_index.layer_count != 0)
        {
            // 逐层扫描
            for (uint64_t i = sub_index.layer_count - 1; 0 < i; --i)
            {
                one_sub_index_result = nearest_neighbors_query(index, sub_index, i, query_vector,
                                                               one_sub_index_result.begin()->second, 1, 10);
            }
            one_sub_index_result =
                nearest_neighbors_last_layer(index, sub_index, 0, query_vector, one_sub_index_result.begin()->second,
                                             top_k, relaxed_monotonicity, distance_bound);
        }
        result.insert(one_sub_index_result.begin(), one_sub_index_result.end());
        one_sub_index_result.clear();
        if (top_k < result.size())
        {
            auto temporary = result.begin();
            std::advance(temporary, top_k);
            result.erase(temporary, result.end());
        }
        distance_bound = std::prev(result.end())->first;
        //        auto end = std::chrono::high_resolution_clock::now();
        //        std::cout << "one sub-index costs(us): "
        //                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    }
    return result;
}

// 插入
template <typename Dimension_Type>
void insert(Index<Dimension_Type> &index, const std::vector<Dimension_Type> &inserted_vector)
{
    // 如果插入向量的维度不等于索引里向量的维度
    //    if (inserted_vector.size() != index.vectors[0].data.size())
    //    {
    //        throw std::invalid_argument("The dimension of insert vector is not "
    //                                    "equality with vectors in index. ");
    //    }
    // 插入向量在原始数据中的偏移量
    uint64_t inserted_vector_global_offset = index.count;
    ++index.count;
    if (inserted_vector_global_offset % index.parameters.sub_index_bound == 0)
    {
        index.sub_indexes.emplace_back(Sub_Index<Dimension_Type>(index.parameters.sub_index_bound));
        index.sub_indexes.back().vectors[index.sub_indexes.back().count] =
            Vector<Dimension_Type>(inserted_vector_global_offset, index.sub_indexes.back().count, inserted_vector);
        ++index.sub_indexes.back().count;
        return;
    }
    index.sub_indexes.back().vectors[index.sub_indexes.back().count] =
        Vector<Dimension_Type>(inserted_vector_global_offset, index.sub_indexes.back().count, inserted_vector);
    ++index.sub_indexes.back().count;
    add(index, index.sub_indexes.back(), index.sub_indexes.back().vectors[index.sub_indexes.back().count - 1], 0);
}

} // namespace dehnsw
