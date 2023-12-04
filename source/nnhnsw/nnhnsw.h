#pragma once

#include <cinttypes>
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

namespace nnhnsw
{

// 向量
template <typename Dimension_Type> class Vector
{
  public:
    std::vector<Dimension_Type> data;

    explicit Vector(const std::vector<Dimension_Type> &data)
    {
        this->data = data;
    }
};

class Layer;

// 每层中的向量
class Vector_In_Cluster
{
  public:
    // 向量在原始数据的偏移量
    uint64_t global_offset;
    // 向量对应的下一层中的向量
    std::weak_ptr<Vector_In_Cluster> lower_layer;
    // 指向的邻居向量
    std::map<float, std::weak_ptr<Vector_In_Cluster>> out;
    // 指向该向量的邻居向量
    std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>> in;
    // 指向该向量所在的层
    std::weak_ptr<Layer> layer;

    explicit Vector_In_Cluster(const uint64_t global_offset)
    {
        this->global_offset = global_offset;
        this->out = std::map<float, std::weak_ptr<Vector_In_Cluster>>();
        this->in = std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>>();
    }
};

// 索引中的一层
class Layer
{
  public:
    // 簇中的向量
    std::unordered_map<uint64_t, std::shared_ptr<Vector_In_Cluster>> vectors;
    // 该簇中被选出的向量在原始数据的偏移量
    std::unordered_set<uint64_t> selected_vectors;
    // 上一层
    std::weak_ptr<Layer> upper_layer;
    // 下一层
    std::weak_ptr<Layer> lower_layer;
    // 在插入向量的过程中
    // 如果一个新向量的插入导致其中某个邻居需要剪枝
    // 而该剪枝操作会导致图的不连通
    // 则不进行该剪枝操作，记录这条边
    // 在之后的向量插入完成后再检验这些边如果删去是否影响图的连通性
    //    std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>> temporary_connecting;

    Layer() = default;
};

// 索引
template <typename Dimension_Type> class Index
{
  public:
    // 向量原始数据
    std::vector<Vector<Dimension_Type>> vectors;

    // 自底向上存放每一层
    std::vector<std::shared_ptr<Layer>> layers;

    // 每个向量的最大邻居向量个数
    uint64_t max_connect{};

    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

    // 插入向量时候选邻居向量系数
    uint64_t relaxed_monotonicity_factor{};

    uint64_t step{};

    Index(const std::vector<std::vector<Dimension_Type>> &vectors, const Distance_Type distance_type,
          const uint64_t max_connect = 10, const uint64_t relaxed_monotonicity_factor = 10, uint64_t step = 3)
    {
        this->vectors = std::vector<Vector<Dimension_Type>>();
        this->max_connect = max_connect;
        this->distance_calculation = get_distance_calculation_function<Dimension_Type>(distance_type);
        this->relaxed_monotonicity_factor = relaxed_monotonicity_factor;
        this->step = step;
        // 判断原始向量数据是否为空
        if (!vectors.empty() && !vectors.begin()->empty())
        {
            // debug
            //            {
            //                uint64_t pause = 10736;
            //                for (auto i = 0; i < pause; ++i)
            //                {
            //                    std::cout << "inserting " << i << std::endl;
            //                    auto begin = std::chrono::high_resolution_clock::now();
            //                    insert(*this, vectors[i]);
            //                    auto end = std::chrono::high_resolution_clock::now();
            //                    std::cout << "inserting one vector costs(us): "
            //                              << std::chrono::duration_cast<std::chrono::microseconds>(end -
            //                              begin).count()
            //                              << std::endl;
            //                }
            //                std::cout << "inserting " << pause << std::endl;
            //                insert(*this, vectors[pause]);
            //                std::cout << "insert done " << std::endl;
            //                for (auto i = pause; i < vectors.size(); ++i)
            //                {
            //                    std::cout << "inserting " << i << std::endl;
            //                    insert(*this, vectors[i]);
            //                    std::cout << "insert done " << std::endl;
            //                }
            //            }
            // debug

            uint64_t total_time = 0;
            for (auto global_offset = 0; global_offset < vectors.size(); ++global_offset)
            {
                auto begin = std::chrono::high_resolution_clock::now();
                insert(*this, vectors[global_offset]);
                auto end = std::chrono::high_resolution_clock::now();
                std::cout << "inserting ths " << global_offset << "th vector costs(us): "
                          << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
                total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            }
            std::cout << "building index consts(us): " << total_time << std::endl;
            for (auto i = 0; i < this->layers.size(); ++i)
            {
                std::cout << "layers[" << i << "]: " << this->layers[i]->vectors.size() << " vectors. " << std::endl;
            }
        }
    }
};

namespace
{

bool connected(const std::shared_ptr<Vector_In_Cluster> &start,
               std::unordered_map<std::shared_ptr<Vector_In_Cluster>,
                                  std::pair<float, std::shared_ptr<Vector_In_Cluster>>> &deleted_edges)
{
    auto layer = start->layer.lock();
    auto last = std::unordered_set<std::shared_ptr<Vector_In_Cluster>>();
    auto next = std::unordered_set<std::shared_ptr<Vector_In_Cluster>>();
    auto flag = std::unordered_set<uint64_t>();
    flag.insert(start->global_offset);
    last.insert(start);
    while (!last.empty())
    {
        for (const auto &vector : last)
        {
            for (const auto &neighbor : vector->out)
            {
                auto neighbor_vector = neighbor.second.lock();
                deleted_edges.erase(neighbor_vector);
                if (flag.insert(neighbor_vector->global_offset).second)
                {
                    next.insert(neighbor_vector);
                }
            }
            for (const auto &neighbor : vector->in)
            {
                auto neighbor_vector = neighbor.second.lock();
                deleted_edges.erase(neighbor_vector);
                if (flag.insert(neighbor_vector->global_offset).second)
                {
                    next.insert(neighbor_vector);
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

bool insert_to_upper_layer(const std::shared_ptr<Vector_In_Cluster> &vector, const uint64_t step)
{
    auto layer = vector->layer.lock();
    auto last = std::vector<std::shared_ptr<Vector_In_Cluster>>();
    auto next = std::vector<std::shared_ptr<Vector_In_Cluster>>();
    auto flag = std::unordered_set<uint64_t>();
    last.push_back(vector);
    for (auto round = 0; round < step; ++round)
    {
        for (auto &last_vector : last)
        {
            for (auto &neighbor : last_vector->out)
            {
                auto neighbor_vector = neighbor.second.lock();
                if (flag.insert(neighbor_vector->global_offset).second)
                {
                    if (layer->selected_vectors.contains(neighbor_vector->global_offset))
                    {
                        return false;
                    }
                    next.push_back(neighbor_vector);
                }
            }
            for (auto &neighbor : last_vector->in)
            {
                auto neighbor_vector = neighbor.second.lock();
                if (flag.insert(neighbor_vector->global_offset).second)
                {
                    if (layer->selected_vectors.contains(neighbor_vector->global_offset))
                    {
                        return false;
                    }
                    next.push_back(neighbor_vector);
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
std::map<float, std::weak_ptr<Vector_In_Cluster>> nearest_neighbors(const Index<Dimension_Type> &index,
                                                                    const std::vector<Dimension_Type> &query_vector,
                                                                    const std::shared_ptr<Vector_In_Cluster> &start,
                                                                    const uint64_t top_k,
                                                                    const uint64_t relaxed_monotonicity_factor)
{
    // 优先队列
    auto nearest_neighbors = std::map<float, std::weak_ptr<Vector_In_Cluster>>();
    auto layer = start->layer.lock();
    if (layer->vectors.size() < top_k + relaxed_monotonicity_factor)
    {
        for (auto &vector : layer->vectors)
        {
            nearest_neighbors.insert(std::make_pair(
                index.distance_calculation(query_vector, index.vectors[vector.first].data), vector.second));
        }
    }
    else
    {
        // 标记簇中的向量是否被遍历过
        std::unordered_set<uint64_t> flags;
        // 如果最近遍历的向量的距离的中位数大于优先队列的最大值，提前结束
        std::set<float> sorted_recent_distance;
        // 最近便利的向量的距离
        std::queue<float> recent_distance;
        // 排队队列
        auto waiting_vectors = std::map<float, std::weak_ptr<Vector_In_Cluster>>();
        waiting_vectors.insert(
            std::make_pair(index.distance_calculation(query_vector, index.vectors[start->global_offset].data), start));
        while (!waiting_vectors.empty())
        {
            auto processing_distance = waiting_vectors.begin()->first;
            auto processing_vector = waiting_vectors.begin()->second.lock();
            waiting_vectors.erase(waiting_vectors.begin());
            flags.insert(processing_vector->global_offset);
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
                    auto temporary_recent_distance = std::queue<float>();
                    std::swap(recent_distance, temporary_recent_distance);
                    auto temporary_sorted_recent_distance = std::set<float>();
                    std::swap(sorted_recent_distance, temporary_sorted_recent_distance);
                    nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector));
                    nearest_neighbors.erase(std::prev(nearest_neighbors.end()));
                }
                else
                {
                    recent_distance.push(processing_distance);
                    sorted_recent_distance.insert(processing_distance);
                    // 如果优先队列中的最大值小于最近浏览的向量的距离的中值
                    // 结束遍历
                    if (relaxed_monotonicity_factor < recent_distance.size())
                    {
                        sorted_recent_distance.erase(recent_distance.front());
                        recent_distance.pop();
                        if (std::prev(nearest_neighbors.end())->first < *(sorted_recent_distance.begin()))
                        {
                            break;
                        }
                    }
                }
            }
            // 计算当前向量的出边指向的向量和目标向量的距离
            for (auto &vector : processing_vector->out)
            {
                auto temporary_vector_pointer = vector.second.lock();
                if (flags.insert(temporary_vector_pointer->global_offset).second)
                {
                    waiting_vectors.insert(
                        std::make_pair(index.distance_calculation(
                                           query_vector, index.vectors[temporary_vector_pointer->global_offset].data),
                                       temporary_vector_pointer));
                }
            }
            // 计算当前向量的入边指向的向量和目标向量的距离
            for (auto &vector : processing_vector->in)
            {
                if (flags.insert(vector.first).second)
                {
                    waiting_vectors.insert(std::make_pair(
                        index.distance_calculation(query_vector, index.vectors[vector.first].data), vector.second));
                }
            }
        }
    }
    return nearest_neighbors;
}

template <typename Dimension_Type>
void insert(Index<Dimension_Type> &index, std::shared_ptr<Vector_In_Cluster> &new_vector, uint64_t target_layer_number)
{
    // 记录被插入向量每一层中距离最近的max_connect个邻居向量
    auto every_layer_neighbors = std::stack<std::map<float, std::weak_ptr<Vector_In_Cluster>>>();
    every_layer_neighbors.push(nearest_neighbors(index, index.vectors[new_vector->global_offset].data,
                                                 index.layers.back()->vectors.begin()->second, index.max_connect,
                                                 index.relaxed_monotonicity_factor));
    // 逐层扫描
    // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
    // 所以不需要实际的层和簇
    // 直接通过上一层中返回的结果即可进行计算
    while (every_layer_neighbors.size() != index.layers.size() - target_layer_number)
    {
        every_layer_neighbors.push(
            nearest_neighbors(index, index.vectors[new_vector->global_offset].data,
                              every_layer_neighbors.top().begin()->second.lock()->lower_layer.lock(), index.max_connect,
                              index.relaxed_monotonicity_factor));
    }
    // 插入向量
    while (!every_layer_neighbors.empty())
    {
        auto deleted_edges = std::unordered_map<std::shared_ptr<Vector_In_Cluster>,
                                                std::pair<float, std::shared_ptr<Vector_In_Cluster>>>();
        auto &layer = index.layers[target_layer_number];
        new_vector->layer = layer;
        layer->vectors.insert(std::pair(new_vector->global_offset, new_vector));
        auto neighbor_iterator = every_layer_neighbors.top().begin();
        while (neighbor_iterator != every_layer_neighbors.top().end())
        {
            auto neighbor = *neighbor_iterator;
            auto neighbor_vector = neighbor.second.lock();
            // 将新的向量指向邻居向量
            new_vector->out.insert(neighbor);
            // 在邻居向量中记录指向自己的新向量
            neighbor_vector->in.insert(std::pair(new_vector->global_offset, new_vector));
            // 新向量和邻居向量的距离小于邻居向量已指向的10个向量的距离
            if (neighbor_vector->out.upper_bound(neighbor.first) != neighbor_vector->out.end())
            {
                neighbor_vector->out.insert(std::pair(neighbor.first, new_vector));
                new_vector->in.insert(std::pair(neighbor_vector->global_offset, neighbor_vector));
            }
            // 如果邻居向量的出度大于最大出度数量
            if (index.max_connect < neighbor_vector->out.size())
            {
                auto temporary = neighbor_vector->out.begin();
                std::advance(temporary, index.max_connect);
                auto record = *temporary;
                deleted_edges.insert(
                    std::make_pair(record.second.lock(), std::make_pair(record.first, neighbor_vector)));
                neighbor_vector->out.erase(temporary);
                record.second.lock()->in.erase(neighbor_vector->global_offset);
            }
            ++neighbor_iterator;
        }
        if (!connected(new_vector, deleted_edges))
        {
            for (const auto &edge : deleted_edges)
            {
                edge.second.second->out.insert(std::make_pair(edge.second.first, edge.first));
                edge.first->in.insert(std::make_pair(edge.second.second->global_offset, edge.second.second));
            }
        }
        // 如果新向量应该被插入上一层中
        if (insert_to_upper_layer(new_vector, index.step))
        {
            every_layer_neighbors.pop();
            layer->selected_vectors.insert(new_vector->global_offset);
            auto temporary = std::make_shared<Vector_In_Cluster>(new_vector->global_offset);
            temporary->lower_layer = new_vector;
            new_vector = temporary;
            ++target_layer_number;
            if (every_layer_neighbors.empty())
            {
                auto new_layer = std::make_shared<Layer>();
                new_layer->lower_layer = index.layers.back();
                index.layers.back()->upper_layer = new_layer;
                index.layers.push_back(new_layer);
                new_layer->vectors.insert(std::pair(new_vector->global_offset, new_vector));
                new_vector->layer = new_layer;
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
    if (index.vectors.empty())
    {
        throw std::logic_error("Empty vectors in index. ");
    }
    if (query_vector.size() != index.vectors[0].data.size())
    {
        throw std::invalid_argument("The dimension of query vector is not "
                                    "equality with vectors in index. ");
    }
    if (relaxed_monotonicity == 0)
    {
        relaxed_monotonicity = top_k / 2;
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
        if (index.layers.size() == 1)
        {
            auto temporary =
                nearest_neighbors(index, query_vector, index.layers[index.layers.size() - 1]->vectors.begin()->second,
                                  100, relaxed_monotonicity);
            for (const auto &neighbor : temporary)
            {
                result.insert(std::make_pair(neighbor.first, neighbor.second.lock()->global_offset));
            }
        }
        else
        {
            // 记录被插入向量每一层中距离最近的top_k个邻居向量
            auto temporary = nearest_neighbors(index, query_vector, index.layers.back()->vectors.begin()->second, 1, 5);
            // 逐层扫描
            // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
            // 所以不需要实际的层和簇
            // 直接通过上一层中返回的结果即可进行计算
            for (auto i = 2; i < index.layers.size(); ++i)
            {
                temporary =
                    nearest_neighbors(index, query_vector, temporary.begin()->second.lock()->lower_layer.lock(), 1, 5);
            }
            for (const auto &neighbor :
                 nearest_neighbors(index, query_vector, temporary.begin()->second.lock()->lower_layer.lock(), 100,
                                   relaxed_monotonicity))
            {
                result.insert(std::make_pair(neighbor.first, neighbor.second.lock()->global_offset));
            }
        }
    }
    return result;
}

// 插入
template <typename Dimension_Type>
void insert(Index<Dimension_Type> &index, const std::vector<Dimension_Type> &inserted_vector)
{
    // 插入向量在原始数据中的偏移量
    uint64_t inserted_vector_global_offset = index.vectors.size();
    // 如果是空的索引
    if (inserted_vector_global_offset == 0)
    {
        index.vectors.push_back(Vector<Dimension_Type>(inserted_vector));
        index.layers.push_back(std::make_shared<Layer>());
        auto new_vector = std::make_shared<Vector_In_Cluster>(inserted_vector_global_offset);
        new_vector->layer = index.layers[0];
        index.layers[0]->vectors.insert(std::make_pair(inserted_vector_global_offset, new_vector));
        return;
    }
    // 如果插入向量的维度不等于索引里向量的维度
    if (inserted_vector.size() != index.vectors[0].data.size())
    {
        throw std::invalid_argument("The dimension of insert vector is not "
                                    "equality with vectors in index. ");
    }
    index.vectors.push_back(Vector<Dimension_Type>(inserted_vector));
    auto new_vector = std::make_shared<Vector_In_Cluster>(inserted_vector_global_offset);
    insert(index, new_vector, 0);
}

} // namespace nnhnsw
