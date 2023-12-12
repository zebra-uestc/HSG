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
    // 向量的原始数据
    std::vector<Dimension_Type> data;
    // 向量在整个数据集中的偏移量
    uint64_t global_offset{};
    // 指向的邻居向量
    std::vector<std::map<float, uint64_t>> out;
    // 指向该向量的邻居向量
    std::vector<std::unordered_set<uint64_t>> in;

    explicit Vector(const uint64_t global_offset, const std::vector<Dimension_Type> &data)
    {
        this->global_offset = global_offset;
        this->data = data;
        this->out.emplace_back();
        this->in.emplace_back();
    }
};

// 索引
template <typename Dimension_Type> class Sub_Index
{
  public:
    // 子索引中的向量
    std::vector<Vector<Dimension_Type>> vectors;
    // 子索引最大层数
    uint64_t layer_count{};
    // 该层中的向量
    uint64_t vector_in_highest_layer{};

    explicit Sub_Index(const uint64_t max_count_bound)
    {
        this->layer_count = 0;
        this->vector_in_highest_layer = 0;
        this->vectors.reserve(max_count_bound + 1);
    }
};

// 索引
template <typename Dimension_Type> class Index
{
  public:
    // 索引中向量的数量
    uint64_t count{};
    // 每个子索引中向量的数量限制
    uint64_t sub_index_bound{};
    // 每个向量的最大邻居向量个数
    uint64_t minimum_connect_number{};
    // 插入向量时候选邻居向量系数
    uint64_t relaxed_monotonicity{};
    // 步长
    uint64_t step{};
    // 子索引
    std::vector<Sub_Index<Dimension_Type>> sub_indexes;
    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

    explicit Index(const Distance_Type distance_type, const uint64_t sub_index_bound = 0xFFFF)
    {
        this->count = 0;
        this->sub_index_bound = sub_index_bound;
        this->minimum_connect_number = 10;
        this->relaxed_monotonicity = 10;
        this->step = 3;
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
        for (const auto &vector_global_offset : last)
        {
            for (const auto &neighbor :
                 sub_index.vectors[vector_global_offset & index.sub_index_bound].out[layer_number])
            {
                deleted_edges.erase(neighbor.second);
                if (flag.insert(neighbor.second).second)
                {
                    next.insert(neighbor.second);
                }
            }
            for (const auto &neighbor_global_offset :
                 sub_index.vectors[vector_global_offset & index.sub_index_bound].in[layer_number])
            {
                deleted_edges.erase(neighbor_global_offset);
                if (flag.insert(neighbor_global_offset).second)
                {
                    next.insert(neighbor_global_offset);
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
                           const uint64_t layer_number, const uint64_t vector_global_offset, const uint64_t step)
{
    auto last = std::unordered_set<uint64_t>();
    auto next = std::unordered_set<uint64_t>();
    auto flag = std::unordered_set<uint64_t>();
    last.insert(vector_global_offset);
    for (auto round = 0; round < step; ++round)
    {
        for (const auto &last_vector_global_offset : last)
        {
            for (auto &neighbor : sub_index.vectors[vector_global_offset & index.sub_index_bound].out[layer_number])
            {
                if (flag.insert(neighbor.second).second)
                {
                    if (layer_number < sub_index.vectors[neighbor.second].out.size())
                    {
                        return false;
                    }
                    next.insert(neighbor.second);
                }
            }
            for (auto &neighbor_vector_global_offset :
                 sub_index.vectors[vector_global_offset & index.sub_index_bound].in[layer_number])
            {
                if (flag.insert(neighbor_vector_global_offset).second)
                {
                    if (layer_number < sub_index.vectors[neighbor_vector_global_offset].in.size())
                    {
                        return false;
                    }
                    next.insert(neighbor_vector_global_offset);
                }
            }
        }
        std::swap(last, next);
        next.clear();
    }
    return true;
}

// 从开始向量查询开始向量所在簇中距离最近的top-k个向量
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
    waiting_vectors.insert(std::make_pair(
        index.distance_calculation(query_vector, sub_index.vectors[start & index.sub_index_bound].data), start));
    while (!waiting_vectors.empty())
    {
        auto processing_distance = waiting_vectors.begin()->first;
        auto processing_vector_global_offset = waiting_vectors.begin()->second;
        waiting_vectors.erase(waiting_vectors.begin());
        flags.insert(processing_vector_global_offset);
        // 如果已遍历的向量小于候选数量
        if (nearest_neighbors.size() < index.minimum_connect_number)
        {
            nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_global_offset));
        }
        else
        {
            // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
            if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
            {
                out_of_bound = 0;
                nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_global_offset));
                nearest_neighbors.erase(std::prev(nearest_neighbors.end()));
            }
            else if (index.relaxed_monotonicity < out_of_bound)
            {
                break;
            }
            else
            {
                ++out_of_bound;
            }
        }
        // 计算当前向量的出边指向的向量和目标向量的距离
        for (auto &vector :
             sub_index.vectors[processing_vector_global_offset & index.sub_index_bound].out[layer_number])
        {
            if (flags.insert(vector.second).second)
            {
                waiting_vectors.insert(
                    std::make_pair(index.distance_calculation(
                                       query_vector, sub_index.vectors[vector.second & index.sub_index_bound].data),
                                   vector.second));
            }
        }
        // 计算当前向量的入边指向的向量和目标向量的距离
        for (auto &vector_global_offset :
             sub_index.vectors[processing_vector_global_offset & index.sub_index_bound].in[layer_number])
        {
            if (flags.insert(vector_global_offset).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector,
                                               sub_index.vectors[vector_global_offset & index.sub_index_bound].data),
                    vector_global_offset));
            }
        }
    }
    return nearest_neighbors;
}

// 从开始向量查询开始向量所在簇中距离最近的top-k个向量
template <typename Dimension_Type>
std::map<float, uint64_t> nearest_neighbors_query(const Index<Dimension_Type> &index,
                                                  const Sub_Index<Dimension_Type> &sub_index,
                                                  const uint64_t layer_number,
                                                  const std::vector<Dimension_Type> &query_vector, const uint64_t start,
                                                  const uint64_t top_k, const uint64_t relaxed_monotonicity,
                                                  const float distance_bound)
{
    // 优先队列
    auto nearest_neighbors = std::map<float, uint64_t>();
    // 标记簇中的向量是否被遍历过
    std::unordered_set<uint64_t> flags;
    uint64_t out_of_bound = 0;
    // 排队队列
    auto waiting_vectors = std::map<float, uint64_t>();
    waiting_vectors.insert(std::make_pair(
        index.distance_calculation(query_vector, sub_index.vectors[start & index.sub_index_bound].data), start));
    while (!waiting_vectors.empty())
    {
        auto processing_distance = waiting_vectors.begin()->first;
        auto processing_vector_global_offset = waiting_vectors.begin()->second;
        waiting_vectors.erase(waiting_vectors.begin());
        flags.insert(processing_vector_global_offset);
        // 如果已遍历的向量小于候选数量
        if (processing_distance < distance_bound)
        {
            if (nearest_neighbors.size() < top_k)
            {
                nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_global_offset));
            }
            else
            {
                // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
                if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
                {
                    out_of_bound = 0;
                    nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector_global_offset));
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
        for (auto &vector :
             sub_index.vectors[processing_vector_global_offset & index.sub_index_bound].out[layer_number])
        {
            if (flags.insert(vector.second).second)
            {
                waiting_vectors.insert(
                    std::make_pair(index.distance_calculation(
                                       query_vector, sub_index.vectors[vector.second & index.sub_index_bound].data),
                                   vector.second));
            }
        }
        // 计算当前向量的入边指向的向量和目标向量的距离
        for (auto &vector_global_offset :
             sub_index.vectors[processing_vector_global_offset & index.sub_index_bound].in[layer_number])
        {
            if (flags.insert(vector_global_offset).second)
            {
                waiting_vectors.insert(std::make_pair(
                    index.distance_calculation(query_vector,
                                               sub_index.vectors[vector_global_offset & index.sub_index_bound].data),
                    vector_global_offset));
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
            auto &neighbor_vector = sub_index.vectors[neighbor.second & index.sub_index_bound];
            // 将新的向量指向邻居向量
            new_vector.out[target_layer_number].insert(neighbor);
            // 在邻居向量中记录指向自己的新向量
            neighbor_vector.in[target_layer_number].insert(new_vector.global_offset);
            // 新向量和邻居向量的距离小于邻居向量已指向的10个向量的距离
            if (neighbor_vector.out[target_layer_number].upper_bound(neighbor.first) !=
                neighbor_vector.out[target_layer_number].end())
            {
                neighbor_vector.out[target_layer_number].insert(
                    std::make_pair(neighbor.first, new_vector.global_offset));
                new_vector.in[target_layer_number].insert(neighbor.second);
            }
            // 如果邻居向量的出度大于最大出度数量
            if (index.minimum_connect_number < neighbor_vector.out[target_layer_number].size())
            {
                auto temporary = neighbor_vector.out[target_layer_number].begin();
                std::advance(temporary, index.minimum_connect_number);
                auto record = *temporary;
                deleted_edges.insert(std::make_pair(record.second, std::make_pair(record.first, neighbor.second)));
                neighbor_vector.out[target_layer_number].erase(temporary);
                sub_index.vectors[record.second].in[target_layer_number].erase(neighbor.second);
            }
        }
        if (!connected(index, sub_index, target_layer_number, new_vector.global_offset, deleted_edges))
        {
            for (const auto &edge : deleted_edges)
            {
                sub_index.vectors[edge.second.second & index.sub_index_bound].out[target_layer_number].insert(
                    std::make_pair(edge.second.first, edge.first));
                sub_index.vectors[edge.first & index.sub_index_bound].in[target_layer_number].insert(
                    edge.second.second);
            }
        }
        // 如果新向量应该被插入上一层中
        if (insert_to_upper_layer(index, sub_index, target_layer_number, new_vector.global_offset, index.step))
        {
            every_layer_neighbors.pop();
            ++target_layer_number;
            if (sub_index.layer_count < target_layer_number)
            {
                sub_index.layer_count = target_layer_number;
                sub_index.vector_in_highest_layer = new_vector.global_offset;
            }
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
        one_sub_index_result.emplace(index.distance_calculation(query_vector, sub_index.vectors[0].data),
                                     sub_index.vectors[0].global_offset);
        if (sub_index.layer_count != 0)
        {
            // 逐层扫描
            for (uint64_t i = sub_index.layer_count - 1; 0 < i; --i)
            {
                one_sub_index_result = nearest_neighbors_query(index, sub_index, i, query_vector,
                                                               one_sub_index_result.begin()->second, 1, 10, MAXFLOAT);
            }
            one_sub_index_result =
                nearest_neighbors_query(index, sub_index, 0, query_vector, one_sub_index_result.begin()->second, top_k,
                                        relaxed_monotonicity, distance_bound);
        }
        result.insert(one_sub_index_result.begin(), one_sub_index_result.end());
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
    if ((inserted_vector_global_offset & index.sub_index_bound) == 0)
    {
        index.sub_indexes.emplace_back(Sub_Index<Dimension_Type>(index.sub_index_bound));
        index.sub_indexes.back().vectors[inserted_vector_global_offset & index.sub_index_bound] =
            Vector<Dimension_Type>(inserted_vector_global_offset, inserted_vector);
        return;
    }
    index.sub_indexes.back().vectors[inserted_vector_global_offset & index.sub_index_bound] =
        Vector<Dimension_Type>(inserted_vector_global_offset, inserted_vector);
    add(index, index.sub_indexes.back(),
        index.sub_indexes.back().vectors[inserted_vector_global_offset & index.sub_index_bound], 0);
}

} // namespace dehnsw
