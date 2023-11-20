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

class Cluster;

// 簇中的向量
class Vector_In_Cluster
{
  public:
    // 向量在原始数据的偏移量
    uint64_t global_offset;
    // 向量对应的下一层中的向量
    std::weak_ptr<Vector_In_Cluster> lower_layer;
    // 指向自己所在的簇
    std::weak_ptr<Cluster> cluster;
    // 指向的邻居向量
    std::map<float, std::weak_ptr<Vector_In_Cluster>> out;
    // 指向该向量的邻居向量
    std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>> in;

    explicit Vector_In_Cluster(const uint64_t global_offset)
    {
        this->global_offset = global_offset;
        this->out = std::map<float, std::weak_ptr<Vector_In_Cluster>>();
        this->in = std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>>();
    }
};

class Layer;

// 每层中的簇
class Cluster
{
  public:
    // 该簇所在的层
    std::weak_ptr<Layer> layer;
    // 簇中的向量
    std::unordered_map<uint64_t, std::shared_ptr<Vector_In_Cluster>> vectors;
    // 该簇中被选出的向量在原始数据的偏移量
    std::unordered_set<uint64_t> selected_vectors;

    explicit Cluster(const std::weak_ptr<Layer> &layer)
    {
        this->layer = layer;
        this->vectors = std::unordered_map<uint64_t, std::shared_ptr<Vector_In_Cluster>>();
        this->selected_vectors = std::unordered_set<uint64_t>();
    }
};

// 索引中的一层
class Layer
{
  public:
    // 每层中的多个簇
    std::vector<std::shared_ptr<Cluster>> clusters;
    // 上一层
    std::weak_ptr<Layer> upper_layer;
    // 下一层
    std::weak_ptr<Layer> lower_layer;

    Layer()
    {
        this->clusters = std::vector<std::shared_ptr<Cluster>>();
    }
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

    // 最大距离系数
    uint64_t distance_bound{};

    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

    Index(const std::vector<std::vector<Dimension_Type>> &vectors, const Distance_Type distance_type,
          const uint64_t max_connect, const uint64_t distance_bound)
    {
        this->vectors = std::vector<Vector<Dimension_Type>>();
        this->distance_bound = distance_bound;
        this->max_connect = max_connect;
        this->distance_calculation = get_distance_calculation_function<Dimension_Type>(distance_type);
        // 判断原始向量数据是否为空
        if (!vectors.empty() && !vectors.begin()->empty())
        {
            for (auto &vector : vectors)
            {
                insert(*this, vector);
            }
        }
    }
};

namespace
{

// 合并两个簇
void merge_two_clusters(const std::shared_ptr<Cluster> &target_cluster, const std::shared_ptr<Cluster> &merged_cluster)
{
    // 移动被合并的簇中的向量到目标簇中
    for (auto &merged_cluster_vector : merged_cluster->vectors)
    {
        merged_cluster_vector.second->cluster = target_cluster;
    }
    target_cluster->vectors.insert(merged_cluster->vectors.begin(), merged_cluster->vectors.end());
    // 移动被合并的簇中被选出的代表向量到目标簇中
    target_cluster->selected_vectors.insert(merged_cluster->selected_vectors.begin(),
                                            merged_cluster->selected_vectors.end());
    //    for (auto &merged_cluster_selected_vector : merged_cluster->selected_vectors)
    //    {
    //        target_cluster->selected_vectors.insert(merged_cluster_selected_vector);
    //    }
    // 删除簇
    auto layer = target_cluster->layer.lock();
    for (auto cluster_iterator = layer->clusters.begin(); cluster_iterator != layer->clusters.end(); ++cluster_iterator)
    {
        if (*cluster_iterator == merged_cluster)
        {
            layer->clusters.erase(cluster_iterator);
            break;
        }
    }
}

// 返回当前簇中所有的簇
std::vector<std::shared_ptr<Cluster>> calculate_clusters(const std::shared_ptr<Cluster> &cluster)
{
    auto new_clusters = std::vector<std::shared_ptr<Cluster>>();
    // key：向量的全局偏移量
    // value：向量在第几个新的簇中
    std::unordered_map<uint64_t, uint64_t> flag;
    //
    {
        flag.insert(std::make_pair(cluster->vectors.begin()->first, 0));
        auto waiting_vectors = std::queue<uint64_t>();
        waiting_vectors.push(cluster->vectors.begin()->first);
        while (!waiting_vectors.empty())
        {
            uint64_t vector_global_offset = waiting_vectors.front();
            waiting_vectors.pop();
            for (auto &out_neighbor : cluster->vectors.find(vector_global_offset)->second->out)
            {
                flag.insert(std::make_pair(out_neighbor.second.lock()->global_offset, 0));
            }
            for (auto &in_neighbor : cluster->vectors.find(vector_global_offset)->second->in)
            {
                flag.insert(std::make_pair(in_neighbor.first, 0));
            }
        }
    }
    if (flag.size() != cluster->vectors.size())
    {
        new_clusters.push_back(cluster);
        uint64_t new_cluster_number = 1;
        for (auto &vector : cluster->vectors)
        {
            if (flag.insert(std::make_pair(vector.first, new_cluster_number)).second)
            {
                auto new_cluster = std::make_shared<Cluster>(cluster->layer);
                vector.second->cluster = new_cluster;
                new_cluster->vectors.insert(vector);
                auto waiting_vectors = std::queue<uint64_t>();
                waiting_vectors.push(vector.first);
                while (!waiting_vectors.empty())
                {
                    uint64_t vector_global_offset = waiting_vectors.front();
                    waiting_vectors.pop();
                    for (auto &out_neighbor : cluster->vectors.find(vector_global_offset)->second->out)
                    {
                        auto temporary_vector_pointer = out_neighbor.second.lock();
                        if (flag.insert(std::make_pair(temporary_vector_pointer->global_offset, new_cluster_number))
                                .second)
                        {
                            temporary_vector_pointer->cluster = new_cluster;
                            new_cluster->vectors.insert(
                                std::make_pair(temporary_vector_pointer->global_offset, temporary_vector_pointer));
                        }
                    }
                    for (auto &in_neighbor : cluster->vectors.find(vector_global_offset)->second->in)
                    {
                        if (flag.insert(std::make_pair(in_neighbor.first, new_cluster_number)).second)
                        {
                            in_neighbor.second.lock()->cluster = new_cluster;
                            new_cluster->vectors.insert(in_neighbor);
                        }
                    }
                }
                new_clusters.push_back(new_cluster);
                ++new_cluster_number;
            }
            if (flag.size() == cluster->vectors.size())
            {
                break;
            }
        }
        auto temporary = std::unordered_set<uint64_t>();
        std::swap(temporary, cluster->selected_vectors);
        for (auto &selected_vector : temporary)
        {
            new_clusters[flag.find(selected_vector)->second]->selected_vectors.insert(selected_vector);
        }
    }
    return new_clusters;
}

// 分裂一个簇
// 新的簇中可能没有被选出的向量在上一层中
// 返回需要插入到上一层中的向量
std::vector<std::shared_ptr<Vector_In_Cluster>> divide_a_cluster(const std::shared_ptr<Cluster> &cluster)
{
    auto new_clusters = calculate_clusters(cluster);
    auto selected_vectors = std::vector<std::shared_ptr<Vector_In_Cluster>>();
    // 如果分裂成多个簇
    if (new_clusters.size() > 1)
    {
        if (new_clusters[0]->selected_vectors.empty())
        {
            auto selected_vector =
                std::make_shared<Vector_In_Cluster>(new_clusters[0]->vectors.begin()->second->global_offset);
            selected_vectors.push_back(selected_vector);
            selected_vector->lower_layer = new_clusters[0]->vectors[0];
            new_clusters[0]->selected_vectors.insert(selected_vector->global_offset);
        }
        auto layer = cluster->layer.lock();
        for (auto new_cluster_offset = 1; new_cluster_offset < new_clusters.size(); ++new_cluster_offset)
        {
            if (new_clusters[new_cluster_offset]->selected_vectors.empty())
            {
                auto selected_vector =
                    std::make_shared<Vector_In_Cluster>(new_clusters[new_cluster_offset]->vectors[0]->global_offset);
                selected_vector->lower_layer = new_clusters[new_cluster_offset]->vectors[0];
                selected_vectors.push_back(selected_vector);
                new_clusters[new_cluster_offset]->selected_vectors.insert(selected_vector->global_offset);
            }
            layer->clusters.push_back(new_clusters[new_cluster_offset]);
        }
    }
    return selected_vectors;
}

// 从开始向量查询开始向量所在簇中距离最近的top-k个向量
template <typename Dimension_Type>
std::map<float, std::weak_ptr<Vector_In_Cluster>> nearest_neighbors(const Index<Dimension_Type> &index,
                                                                    const std::vector<Dimension_Type> &query_vector,
                                                                    const std::shared_ptr<Vector_In_Cluster> &start,
                                                                    uint64_t top_k = 0)
{
    if (top_k == 0)
    {
        top_k = index.max_connect;
    }
    // 标记簇中的向量是否被遍历过
    std::unordered_set<uint64_t> flags;
    // 如果最近遍历的向量的距离的中位数大于优先队列的最大值，提前结束
    std::set<float> sorted_recent_distance;
    // 最近便利的向量的距离
    std::queue<float> recent_distance;
    // 排队队列
    auto waiting_vectors = std::map<float, std::weak_ptr<Vector_In_Cluster>>();
    // 优先队列
    auto nearest_neighbors = std::map<float, std::weak_ptr<Vector_In_Cluster>>();
    waiting_vectors.insert(
        std::make_pair(index.distance_calculation(query_vector, index.vectors[start->global_offset].data), start));
    while (!waiting_vectors.empty())
    {
        auto processing_distance = waiting_vectors.begin()->first;
        auto processing_vector = waiting_vectors.begin()->second.lock();
        waiting_vectors.erase(waiting_vectors.begin());
        flags.insert(processing_vector->global_offset);
        recent_distance.push(processing_distance);
        sorted_recent_distance.insert(processing_distance);
        // 如果已遍历的向量小于top-k
        if (nearest_neighbors.size() < top_k)
        {
            nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector));
        }
        else
        {
            sorted_recent_distance.erase(recent_distance.front());
            recent_distance.pop();
            // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
            if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
            {
                nearest_neighbors.insert(std::make_pair(processing_distance, processing_vector));
                nearest_neighbors.erase(std::prev(nearest_neighbors.end()));
            }
            auto median = sorted_recent_distance.begin();
            std::advance(median, sorted_recent_distance.size() / 2);
            // 如果优先队列中的最大值小于最近浏览的向量的距离的中值
            // 结束遍历
            if (std::prev(nearest_neighbors.end())->first < *median)
            {
                break;
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
                    index.distance_calculation(query_vector, index.vectors[vector.second.lock()->global_offset].data),
                    vector.second));
            }
        }
    }
    return nearest_neighbors;
}

template <typename Dimension_Type>
void insert(Index<Dimension_Type> &index, std::shared_ptr<Vector_In_Cluster> &new_vector,
            const uint64_t target_layer_number)
{
    if (target_layer_number == index.layers.size())
    {
        auto new_layer = std::make_shared<Layer>();
        new_layer->lower_layer = index.layers[index.layers.size() - 1];
        index.layers[index.layers.size() - 1]->upper_layer = new_layer;
        index.layers.push_back(new_layer);
        auto new_cluster = std::make_shared<Cluster>(new_layer);
        new_cluster->vectors.insert(std::pair(new_vector->global_offset, new_vector));
        new_vector->cluster = new_cluster;
        new_layer->clusters.push_back(new_cluster);
        return;
    }
    // 记录被插入向量每一层中距离最近的max_connect个邻居向量
    std::stack<std::map<float, std::weak_ptr<Vector_In_Cluster>>> every_layer_neighbors;
    every_layer_neighbors.push(
        nearest_neighbors(index, index.vectors[new_vector->global_offset].data,
                          index.layers[index.layers.size() - 1]->clusters[0]->vectors.begin()->second));
    // 逐层扫描
    // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
    // 所以不需要实际的层和簇
    // 直接通过上一层中返回的结果即可进行计算
    while (every_layer_neighbors.size() != index.layers.size() - target_layer_number)
    {
        // 一层中有好多的簇
        // 每个簇之间是不连通的
        // 所以要进行多次计算
        // 最后汇总计算结果
        std::map<float, std::weak_ptr<Vector_In_Cluster>> one_layer_neighbors;
        for (auto &start_vector : every_layer_neighbors.top())
        {
            auto one_cluster_neighbors = nearest_neighbors(index, index.vectors[new_vector->global_offset].data,
                                                           start_vector.second.lock()->lower_layer.lock());
            one_layer_neighbors.insert(one_cluster_neighbors.begin(), one_cluster_neighbors.end());
            //            for (auto &neighbor : one_cluster_neighbors)
            //            {
            //                one_layer_neighbors.insert(neighbor);
            //            }
            auto last_neighbor = one_layer_neighbors.begin();
            std::advance(last_neighbor, index.max_connect);
            one_layer_neighbors.erase(last_neighbor, one_layer_neighbors.end());
        }
        every_layer_neighbors.push(one_layer_neighbors);
    }
    // 插入向量
    while (!every_layer_neighbors.empty())
    {
        bool insert_to_upper_layer = false;
        auto base_cluster = every_layer_neighbors.top().begin()->second.lock()->cluster.lock();
        auto base_distance = every_layer_neighbors.top().begin()->first;
        uint64_t distance_rank = 1;
        // 把新的向量加入到距离最短的向量所在的簇里
        new_vector->cluster = base_cluster;
        base_cluster->vectors.insert(std::pair(new_vector->global_offset, new_vector));
        for (auto &neighbor : every_layer_neighbors.top())
        {
            if (base_distance * distance_rank * index.distance_bound < neighbor.first)
            {
                break;
            }
            auto neighbor_vector = neighbor.second.lock();
            // 如果插入向量与邻居向量是在一个簇中
            if (base_cluster == neighbor_vector->cluster.lock())
            {
                // 将新的向量指向邻居向量
                new_vector->out.insert(std::pair(neighbor.first, neighbor_vector));
                // 在邻居向量中记录指向自己的新向量
                neighbor_vector->in.insert(std::pair(new_vector->global_offset, new_vector));
                // 如果邻居向量也指向新向量
                if (neighbor_vector->out.upper_bound(neighbor.first) != neighbor_vector->out.end())
                {
                    neighbor_vector->out.insert(std::pair(neighbor.first, new_vector));
                    new_vector->in.insert(std::pair(neighbor_vector->global_offset, neighbor_vector));
                    // 如果新向量是距离邻居向量最近的向量
                    // 检查是否满足距离限制条件
                    if (neighbor_vector->out.begin()->first == neighbor.first)
                    {
                        uint64_t offset = 1;
                        for (auto iterator = neighbor_vector->out.begin(); iterator != neighbor_vector->out.end();
                             ++iterator)
                        {
                            if (neighbor.first * offset * index.distance_bound < iterator->first)
                            {
                                for (auto j = iterator; j != neighbor_vector->out.end(); ++j)
                                {
                                    j->second.lock()->in.erase(neighbor_vector->global_offset);
                                }
                                neighbor_vector->out.erase(iterator, neighbor_vector->out.end());
                                auto selected_vectors = divide_a_cluster(base_cluster);
                                for (auto &selected_vector : selected_vectors)
                                {
                                    insert(index, selected_vector, target_layer_number + 1);
                                }
                                break;
                            }
                            ++offset;
                        }
                    }
                    else if (index.max_connect < neighbor_vector->out.size())
                    {
                        neighbor_vector->out.rbegin()->second.lock()->in.erase(neighbor_vector->global_offset);
                        neighbor_vector->out.erase(std::prev(neighbor_vector->out.end()));
                        auto selected_vectors = divide_a_cluster(base_cluster);
                        for (auto &selected_vector : selected_vectors)
                        {
                            insert(index, selected_vector, target_layer_number + 1);
                        }
                    }
                }
            }
            else
            {
                // 如果邻居向量指向新的向量
                if (neighbor_vector->out.upper_bound(neighbor.first) != neighbor_vector->out.end())
                {
                    // 新向量是距离邻居向量距离最短的向量
                    if (neighbor.first < neighbor_vector->out.begin()->first)
                    {
                        uint64_t offset = 1;
                        for (auto iterator = neighbor_vector->out.begin(); iterator != neighbor_vector->out.end();
                             ++iterator)
                        {
                            if (neighbor.first * offset * index.distance_bound < iterator->first)
                            {
                                for (auto j = iterator; j != neighbor_vector->out.end(); ++j)
                                {
                                    j->second.lock()->in.erase(neighbor_vector->global_offset);
                                }
                                neighbor_vector->out.erase(iterator, neighbor_vector->out.end());
                                auto selected_vectors = divide_a_cluster(neighbor_vector->cluster.lock());
                                for (auto &selected_vector : selected_vectors)
                                {
                                    insert(index, selected_vector, target_layer_number + 1);
                                }
                                break;
                            }
                            ++offset;
                        }
                    }
                    else if (index.max_connect == neighbor_vector->out.size())
                    {
                        neighbor_vector->out.rbegin()->second.lock()->in.erase(neighbor_vector->global_offset);
                        neighbor_vector->out.erase(std::prev(neighbor_vector->out.end()));
                        auto selected_vectors = divide_a_cluster(neighbor_vector->cluster.lock());
                        for (auto &selected_vector : selected_vectors)
                        {
                            insert(index, selected_vector, target_layer_number + 1);
                        }
                    }
                    neighbor_vector->out.insert(std::make_pair(neighbor.first, new_vector));
                    new_vector->in.insert(std::make_pair(neighbor_vector->global_offset, neighbor_vector));
                }
                // 将新的向量指向邻居向量
                new_vector->out.insert(std::make_pair(neighbor.first, neighbor.second));
                // 在邻居向量中记录指向自己的新向量
                neighbor_vector->in.insert(std::make_pair(new_vector->global_offset, new_vector));
                // 合并两个簇
                merge_two_clusters(base_cluster, neighbor_vector->cluster.lock());
                // 如果合并完成后该层中只剩一个簇
                if (new_vector->cluster.lock()->layer.lock()->clusters.size() == 1)
                {
                    while (index.layers[index.layers.size() - 1] != new_vector->cluster.lock()->layer.lock())
                    {
                        index.layers.pop_back();
                    }
                    base_cluster->selected_vectors.clear();
                }
                else
                {
                    insert_to_upper_layer = true;
                    auto temporary = std::make_shared<Vector_In_Cluster>(new_vector->global_offset);
                    temporary->lower_layer = new_vector;
                    new_vector = temporary;
                }
            }
            ++distance_rank;
        }
        if (!insert_to_upper_layer)
        {
            break;
        }
        every_layer_neighbors.pop();
    }
}

} // namespace

// 查询
template <typename Dimension_Type>
std::map<float, uint64_t> query(const Index<Dimension_Type> &index, const std::vector<Dimension_Type> &query_vector,
                                uint64_t top_k)
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
        // 记录被插入向量每一层中距离最近的top_k个邻居向量
        std::map<float, std::weak_ptr<Vector_In_Cluster>> every_layer_neighbors =
            nearest_neighbors(index, query_vector, index.layers[0]->clusters[0]->vectors.begin()->second, top_k);
        // 逐层扫描
        // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
        // 所以不需要实际的层和簇
        // 直接通过上一层中返回的结果即可进行计算
        for (auto i = 1; i < index.layers.size(); ++i)
        {
            // 一层中有好多的簇
            // 每个簇之间是不连通的
            // 所以要进行多次计算
            // 最后汇总计算结果
            std::map<float, std::weak_ptr<Vector_In_Cluster>> one_layer_neighbors;
            for (auto &start_vector : every_layer_neighbors)
            {
                auto temporary_nearest_neighbors =
                    nearest_neighbors(index, query_vector, start_vector.second.lock()->lower_layer.lock(), top_k);
                one_layer_neighbors.insert(temporary_nearest_neighbors.begin(), temporary_nearest_neighbors.end());
                auto last_neighbor = one_layer_neighbors.begin();
                std::advance(last_neighbor, index.max_connect);
                one_layer_neighbors.erase(last_neighbor, one_layer_neighbors.end());
            }
            every_layer_neighbors.clear();
            std::swap(every_layer_neighbors, one_layer_neighbors);
        }
        for (auto &nearest_vector : every_layer_neighbors)
        {
            result.insert(std::make_pair(nearest_vector.first, nearest_vector.second.lock()->global_offset));
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
        index.layers[0]->clusters.push_back(std::make_shared<Cluster>(index.layers[0]));
        auto new_vector = std::make_shared<Vector_In_Cluster>(inserted_vector_global_offset);
        new_vector->cluster = index.layers[0]->clusters[0];
        index.layers[0]->clusters[0]->vectors.insert(std::make_pair(inserted_vector_global_offset, new_vector));
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
    insert(index, std::move(new_vector), 0);
}

} // namespace nnhnsw
