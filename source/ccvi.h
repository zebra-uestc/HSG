#pragma once

#include <chrono>
#include <cinttypes>
#include <iostream>
#include <iterator>
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

namespace ccvi
{

// 索引中的向量
template <typename Dimension_Type> class Vector
{
  public:
    // 向量的最大层数
    uint64_t layer{};
    // 向量在整个数据集中的偏移量
    uint64_t global_offset{};
    // 向量的原始数据
    std::vector<Dimension_Type> data;
    // 指向的邻居向量
    std::vector<std::map<float, uint64_t>> out;
    // 指向该向量的邻居向量
    std::vector<std::unordered_set<uint64_t>> in;

    explicit Vector(const uint64_t global_offset, const std::vector<Dimension_Type> &data)
    {
        this->layer = 0;
        this->global_offset = global_offset;
        this->data = data;
        this->out.emplace_back();
        this->in.emplace_back();
    }
};

// 索引
template <typename Dimension_Type> class Index
{
  public:
    // 索引中向量的数量
    uint64_t count{};
    // 每个向量的邻居个数
    uint64_t minimum_connect_number{};
    // 插入向量时候选邻居向量系数
    uint64_t relaxed_monotonicity{};
    // 步长
    uint64_t step{};
    // 索引的最大层数
    uint64_t layer_count{};
    // 最高层中的向量的全局偏移量
    uint64_t vector_in_highest_layer{};
    // 索引中的向量
    std::vector<Vector<Dimension_Type>> vectors;
    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

    explicit Index(const Distance_Type distance_type)
    {
        this->count = 0;
        this->minimum_connect_number = 10;
        this->relaxed_monotonicity = 10;
        this->step = 3;
        this->layer_count = 0;
        this->vector_in_highest_layer = 0;
        this->distance_calculation = get_distance_calculation_function<Dimension_Type>(distance_type);
    }
};

namespace
{

template <class Dimension_Type>
bool connected(const Index<Dimension_Type> &index, const uint64_t layer_number, const uint64_t start,
               std::unordered_map<uint64_t, std::pair<float, uint64_t>> &deleted_edges)
{
    // 上一轮中被遍历到向量
    auto last = std::unordered_set<uint64_t>();
    // 每轮中被遍历到的向量
    auto next = std::unordered_set<uint64_t>();
    // 记录已遍历过的向量
    auto flag = std::unordered_set<uint64_t>();
    flag.insert(start);
    last.insert(start);
    for (auto round = 0; round < 4; ++round)
    {
        // 遍历上一轮中被遍历到的向量的所有邻居向量
        for (const auto &last_vector_global_offset : last)
        {
            // 遍历出边
            for (const auto &neighbor : index.vectors[last_vector_global_offset].out[layer_number])
            {
                deleted_edges.erase(neighbor.second);
                // 如果插入成功
                // 即
                // 还没有被遍历过
                if (flag.insert(neighbor.second).second)
                {
                    next.insert(neighbor.second);
                }
            }
            // 遍历入边
            for (const auto &neighbor_global_offset : index.vectors[last_vector_global_offset].in[layer_number])
            {
                deleted_edges.erase(neighbor_global_offset);
                // 如果插入成功
                // 即
                // 还没有被遍历过
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

template <class Dimension_Type>
bool insert_to_upper_layer(const Index<Dimension_Type> &index, const uint64_t layer_number,
                           const uint64_t vector_global_offset)
{
    // 上一轮中被遍历到向量
    auto last = std::unordered_set<uint64_t>();
    last.insert(vector_global_offset);
    // 每轮中被遍历到的向量
    auto next = std::unordered_set<uint64_t>();
    // 记录已遍历过的向量
    auto flag = std::unordered_set<uint64_t>();
    flag.insert(vector_global_offset);
    for (auto round = 0; round < index.step; ++round)
    {
        // 遍历上一轮中被遍历到的向量的所有邻居向量
        for (auto &last_vector_global_offset : last)
        {
            // 遍历出边
            for (auto &neighbor : index.vectors[last_vector_global_offset].out[layer_number])
            {
                // 如果插入成功
                // 即
                // 还没有被遍历过
                if (flag.insert(neighbor.second).second)
                {
                    // 如果在step内有向量被选则添加到上一层
                    // 则该向量不需要被添加到上一层
                    if (layer_number < index.vectors[neighbor.second].layer)
                    {
                        return false;
                    }
                    next.insert(neighbor.second);
                }
            }
            // 遍历入边
            for (auto &neighbor_global_offset : index.vectors[last_vector_global_offset].in[layer_number])
            {
                // 如果插入成功
                // 即
                // 还没有被遍历过
                if (flag.insert(neighbor_global_offset).second)
                {
                    // 如果在step内有向量被选则添加到上一层
                    // 则该向量不需要被添加到上一层
                    if (layer_number < index.vectors[neighbor_global_offset].layer)
                    {
                        return false;
                    }
                    next.insert(neighbor_global_offset);
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
std::map<float, uint64_t> nearest_neighbors(const Index<Dimension_Type> &index, const uint64_t layer_number,
                                            const std::vector<Dimension_Type> &query_vector, const uint64_t start,
                                            const uint64_t top_k, const uint64_t relaxed_monotonicity)
{
    uint64_t out_of_bound = 0;
    // 优先队列
    auto nearest_neighbors = std::map<float, uint64_t>();
    // 标记向量是否被遍历过
    std::unordered_set<uint64_t> flags;
    // 排队队列
    auto waiting_vectors = std::map<float, uint64_t>();
    waiting_vectors.insert(std::make_pair(index.distance_calculation(query_vector, index.vectors[start].data), start));
    while (!waiting_vectors.empty())
    {
        auto processing_distance = waiting_vectors.begin()->first;
        auto processing_vector = waiting_vectors.begin()->second;
        waiting_vectors.erase(waiting_vectors.begin());
        flags.insert(processing_vector);
        // 如果已遍历的向量小于候选数量
        if (nearest_neighbors.size() < top_k)
        {
            nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector));
        }
        else
        {
            // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
            if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
            {
                out_of_bound = 0;
                nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector));
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
        for (auto i = layer_number; i <= index.vectors[processing_vector].layer; ++i)
        {
            // 计算当前向量的出边指向的向量和目标向量的距离
            for (auto &vector : index.vectors[processing_vector].out[i])
            {
                if (flags.insert(vector.second).second)
                {
                    waiting_vectors.insert(std::make_pair(
                        index.distance_calculation(query_vector, index.vectors[vector.second].data), vector.second));
                }
            }
            // 计算当前向量的入边指向的向量和目标向量的距离
            for (auto &vector_global_offset : index.vectors[processing_vector].in[i])
            {
                if (flags.insert(vector_global_offset).second)
                {
                    waiting_vectors.insert(std::make_pair(
                        index.distance_calculation(query_vector, index.vectors[vector_global_offset].data),
                        vector_global_offset));
                }
            }
        }
    }
    return nearest_neighbors;
}

template <typename Dimension_Type> void add(Index<Dimension_Type> &index, Vector<Dimension_Type> new_vector)
{
    // 记录被插入向量每一层中距离最近的max_connect个邻居向量
    auto every_layer_neighbors = std::stack<std::map<float, uint64_t>>();
    every_layer_neighbors.push(nearest_neighbors(index, index.layer_count, new_vector.data,
                                                 index.vector_in_highest_layer, index.minimum_connect_number,
                                                 index.relaxed_monotonicity));
    // 逐层扫描
    // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
    // 所以不需要实际的层和簇
    // 直接通过上一层中返回的结果即可进行计算
    for (int64_t layer_number = index.layer_count - 1; 0 <= layer_number; --layer_number)
    {
        every_layer_neighbors.push(nearest_neighbors(index, layer_number, new_vector.data,
                                                     every_layer_neighbors.top().begin()->second,
                                                     index.minimum_connect_number, index.relaxed_monotonicity));
    }
    // 插入向量
    for (uint64_t layer_number = 0; layer_number <= index.layer_count; ++layer_number)
    {
        // <1, <2, 3>>
        // 1：被删除边的终点
        // 2：两向量个的距离
        // 3：被删除边的起点
        auto deleted_edges = std::unordered_map<uint64_t, std::pair<float, uint64_t>>();
        for (auto neighbor : every_layer_neighbors.top())
        {
            // 将新的向量指向邻居向量
            new_vector.out[layer_number].insert(neighbor);
            auto &neighbor_vector = index.vectors[neighbor.second];
            // 在邻居向量中记录指向自己的新向量
            neighbor_vector.in[layer_number].insert(new_vector.global_offset);
            // 新向量和邻居向量的距离小于邻居向量已指向的10个向量的距离
            if (neighbor_vector.out[layer_number].upper_bound(neighbor.first) !=
                neighbor_vector.out[layer_number].end())
            {
                neighbor_vector.out[layer_number].insert(std::pair(neighbor.first, new_vector.global_offset));
                new_vector.in[layer_number].insert(neighbor.second);
            }
            // 如果邻居向量的出度大于最大出度数量
            if (index.minimum_connect_number < neighbor_vector.out[layer_number].size())
            {
                auto temporary = neighbor_vector.out[layer_number].begin();
                std::advance(temporary, index.minimum_connect_number);
                auto record = *temporary;
                deleted_edges.insert(std::make_pair(record.second, std::make_pair(record.first, neighbor.second)));
                neighbor_vector.out[layer_number].erase(temporary);
                index.vectors[record.second].in[layer_number].erase(neighbor.second);
            }
        }
        if (!connected(index, layer_number, new_vector.global_offset, deleted_edges))
        {
            for (const auto &edge : deleted_edges)
            {
                index.vectors[edge.second.second].out[layer_number].insert(
                    std::make_pair(edge.second.first, edge.first));
                index.vectors[edge.first].in[layer_number].insert(edge.second.second);
            }
        }
        // 如果新向量应该被插入上一层中
        if (insert_to_upper_layer(index, layer_number, new_vector.global_offset))
        {
            every_layer_neighbors.pop();
            ++new_vector.layer;
            new_vector.out.emplace_back();
            new_vector.in.emplace_back();
            if (index.layer_count == layer_number)
            {
                ++index.layer_count;
                index.vector_in_highest_layer = new_vector.global_offset;
                break;
            }
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
    //    if (query_vector.size() != index.vectors[0]->data.size())
    //    {
    //        throw std::invalid_argument("The dimension of query vector is not "
    //                                    "equality with vectors in index. ");
    //    }
    if (relaxed_monotonicity == 0)
    {
        relaxed_monotonicity = top_k;
    }
    std::map<float, uint64_t> result;
    // 如果索引中的向量数量小于top-k
    // 直接暴力搜索返回排序后的全部结果
    if (index.vectors.size() < top_k)
    {
        for (uint64_t global_offset = 0; global_offset < index.vectors.size(); ++global_offset)
        {
            result.insert(std::make_pair(index.distance_calculation(query_vector, index.vectors[global_offset].data),
                                         global_offset));
        }
    }
    else
    {
        result = nearest_neighbors(index, 0, query_vector, index.vector_in_highest_layer, top_k, relaxed_monotonicity);
    }
    return result;
}

// 插入
template <typename Dimension_Type>
void insert(Index<Dimension_Type> &index, const std::vector<Dimension_Type> &inserted_vector)
{
    // 如果插入向量的维度不等于索引里向量的维度
    //    if (inserted_vector.size() != index.vectors[0]->data.size())
    //    {
    //        throw std::invalid_argument("The dimension of insert vector is not "
    //                                    "equality with vectors in index. ");
    //    }
    // 插入向量在原始数据中的偏移量
    uint64_t inserted_vector_global_offset = index.count;
    ++index.count;
    // 如果是空的索引
    if (inserted_vector_global_offset == 0)
    {
        index.vectors.push_back(Vector<Dimension_Type>(inserted_vector_global_offset, inserted_vector));
        return;
    }
    //    auto temporary = std::to_string(inserted_vector_global_offset);
    //    switch (temporary.length())
    //    {
    //    case 5:
    //        index.max_connect = 10;
    //        index.relaxed_monotonicity = 10;
    //        break;
    //    case 6:
    //        index.max_connect = 20;
    //        index.relaxed_monotonicity = 20;
    //        break;
    //    case 7:
    //        index.max_connect = 30;
    //        index.relaxed_monotonicity = 30;
    //        break;
    //    case 8:
    //        index.max_connect = 40;
    //        index.relaxed_monotonicity = 40;
    //        break;
    //    case 9:
    //        index.max_connect = 50;
    //        index.relaxed_monotonicity = 50;
    //        break;
    //    case 10:
    //        index.max_connect = 60;
    //        index.relaxed_monotonicity = 60;
    //        break;
    //    }
    //    if (5 < temporary.size())
    //    {
    //        switch (temporary[1])
    //        {
    //        case '0':
    //        case '1':
    //            index.step = 6;
    //            break;
    //        case '2':
    //        case '3':
    //            index.step = 5;
    //            break;
    //        case '4':
    //        case '5':
    //            index.step = 4;
    //            break;
    //        case '6':
    //        case '7':
    //            index.step = 3;
    //            break;
    //        case '8':
    //        case '9':
    //            index.step = 2;
    //            break;
    //        }
    //    }
    index.vectors.push_back(Vector<Dimension_Type>(inserted_vector_global_offset, inserted_vector));
    add(index, index.vectors.back());
}

} // namespace ccvi
