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
#include <vector>

#include "data_struct.h"
#include "distance.h"

namespace nnhnsw
{

// 索引
template <typename Dimension_Type> class Index
{
  private:
    // 向量原始数据
    std::vector<Vector<Dimension_Type>> vectors;

    // 自底向上存放每一层
    std::vector<std::shared_ptr<Layer>> layers;

    // 每个向量的最大邻居向量个数
    uint64_t max_connect{};

    // 最大距离倍数
    uint64_t distance_bound{};

    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

    // 从开始向量查询开始向量所在簇中距离最近的topk个向量
    std::multimap<float, std::weak_ptr<Vector_In_Cluster>> nearest_neighbors(
        const std::vector<Dimension_Type> &query_vector, const std::shared_ptr<Vector_In_Cluster> &start,
        uint64_t top_k = 0)
    {
        if (top_k == 0)
        {
            top_k = this->max_connect;
        }
        // 标记簇中的向量是否被遍历过
        std::unordered_set<uint64_t> flags;
        // 如果最近遍历的向量的距离的中位数大于优先队列的最大值，提前结束
        std::multiset<float> sorted_recent_distance;
        // 最近便利的向量的距离
        std::queue<float> recent_distance;
        // 排队队列
        std::multimap<float, std::weak_ptr<Vector_In_Cluster>> waiting_vectors;
        // 优先队列
        std::multimap<float, std::weak_ptr<Vector_In_Cluster>> nearest_neighbors;
        waiting_vectors.emplace(this->distance_calculation(query_vector, this->vectors[start->global_offset].data),
                                start);
        while (!waiting_vectors.empty())
        {
            auto processing_distance = waiting_vectors.begin()->first;
            auto processing_vector = waiting_vectors.begin()->second.lock();
            waiting_vectors.erase(waiting_vectors.begin());
            flags.insert(processing_vector->global_offset);
            recent_distance.push(processing_distance);
            sorted_recent_distance.insert(processing_distance);
            if (nearest_neighbors.size() < top_k)
            {
                nearest_neighbors.emplace(processing_distance, processing_vector);
            }
            else
            {
                sorted_recent_distance.erase(recent_distance.front());
                recent_distance.pop();
                if (nearest_neighbors.upper_bound(processing_distance) != nearest_neighbors.end())
                {
                    nearest_neighbors.emplace(processing_distance, processing_vector);
                    nearest_neighbors.erase(std::prev(nearest_neighbors.end()));
                }
                auto median = sorted_recent_distance.begin();
                std::advance(median, sorted_recent_distance.size() / 2);
                if (nearest_neighbors.rbegin()->first < *median)
                {
                    break;
                }
            }
            for (auto &out_iterator : processing_vector->out)
            {
                auto temporary_vector_pointer = out_iterator.second.lock();
                if (flags.insert(temporary_vector_pointer->global_offset).second)
                {
                    waiting_vectors.emplace(
                        this->distance_calculation(query_vector,
                                                   this->vectors[temporary_vector_pointer->global_offset].data),
                        temporary_vector_pointer);
                }
            }
            for (auto &in_iterator : processing_vector->in)
            {
                if (flags.insert(in_iterator.first).second)
                {
                    waiting_vectors.emplace(
                        this->distance_calculation(query_vector,
                                                   this->vectors[in_iterator.second.lock()->global_offset].data),
                        in_iterator.second);
                }
            }
        }
        return nearest_neighbors;
    }

    void insert(std::shared_ptr<Vector_In_Cluster> &new_vector, const uint64_t target_layer_number)
    {
        if (target_layer_number == this->layers.size())
        {
            auto new_layer = std::make_shared<Layer>();
            new_layer->lower_layer = this->layers[this->layers.size() - 1];
            this->layers[this->layers.size() - 1]->upper_layer = new_layer;
            this->layers.push_back(new_layer);
            auto new_cluster = std::make_shared<Cluster>(new_layer);
            new_cluster->vectors.insert(std::pair(new_vector->global_offset, new_vector));
            new_vector->cluster = new_cluster;
            new_layer->clusters.push_back(new_cluster);
            return;
        }
        // 记录被插入向量每一层中距离最近的max_connect个邻居向量
        std::stack<std::multimap<float, std::weak_ptr<Vector_In_Cluster>>> every_layer_neighbors;
        every_layer_neighbors.push(
            this->nearest_neighbors(this->vectors[new_vector->global_offset].data,
                                    this->layers[this->layers.size() - 1]->clusters[0]->vectors.begin()->second));
        // 逐层扫描
        // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
        // 所以不需要实际的层和簇
        // 直接通过上一层中返回的结果即可进行计算
        while (every_layer_neighbors.size() != this->layers.size() - target_layer_number)
        {
            // 一层中有好多的簇
            // 每个簇之间是不连通的
            // 所以要进行多次计算
            // 最后汇总计算结果
            std::multimap<float, std::weak_ptr<Vector_In_Cluster>> one_layer_neighbors;
            for (auto &start_vector_iterator : every_layer_neighbors.top())
            {
                auto one_cluster_neighbors =
                    this->nearest_neighbors(this->vectors[new_vector->global_offset].data,
                                            start_vector_iterator.second.lock()->lower_layer.lock());
                for (auto &neighbor_iterator : one_cluster_neighbors)
                {
                    one_layer_neighbors.insert(neighbor_iterator);
                }
                auto last_neighbor = one_layer_neighbors.begin();
                std::advance(last_neighbor, this->max_connect);
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
            for (auto &neighbor_iterator : every_layer_neighbors.top())
            {
                if (base_distance * distance_rank * this->distance_bound < neighbor_iterator.first)
                {
                    break;
                }
                auto neighbor_vector = neighbor_iterator.second.lock();
                // 插入向量与邻居向量是否在一个簇中
                if (base_cluster == neighbor_vector->cluster.lock())
                {
                    // 将新的向量指向邻居向量
                    new_vector->out.insert(std::pair(neighbor_iterator.first, neighbor_vector));
                    // 在邻居向量中记录指向自己的新向量
                    neighbor_vector->in.insert(std::pair(new_vector->global_offset, new_vector));
                    // 计算邻居向量是否指向新的向量
                    if (neighbor_vector->out.upper_bound(neighbor_iterator.first) != neighbor_vector->out.end())
                    {
                        neighbor_vector->out.insert(std::pair(neighbor_iterator.first, new_vector));
                        new_vector->in.insert(std::pair(neighbor_vector->global_offset, neighbor_vector));
                        if (neighbor_vector->out.begin()->first == neighbor_iterator.first)
                        {
                            uint64_t offset = 1;
                            for (auto iterator = neighbor_vector->out.begin(); iterator != neighbor_vector->out.end();
                                 ++iterator)
                            {
                                if (neighbor_iterator.first * offset * this->distance_bound < iterator->first)
                                {
                                    for (auto j = iterator; j != neighbor_vector->out.end(); ++j)
                                    {
                                        j->second.lock()->in.erase(neighbor_vector->global_offset);
                                    }
                                    neighbor_vector->out.erase(iterator, neighbor_vector->out.end());
                                    auto selected_vectors =
                                        neighbor_vector->cluster.lock()->layer.lock()->divide_a_cluster(
                                            neighbor_vector->cluster.lock());
                                    for (auto &selected_vector : selected_vectors)
                                    {
                                        this->insert(selected_vector, target_layer_number + 1);
                                    }
                                    break;
                                }
                                ++offset;
                            }
                        }
                        else if (this->max_connect < neighbor_vector->out.size())
                        {
                            neighbor_vector->out.rbegin()->second.lock()->in.erase(neighbor_vector->global_offset);
                            neighbor_vector->out.erase(std::prev(neighbor_vector->out.end()));
                            auto selected_vectors = base_cluster->layer.lock()->divide_a_cluster(base_cluster);
                            for (auto &selected_vector : selected_vectors)
                            {
                                this->insert(selected_vector, target_layer_number + 1);
                            }
                        }
                    }
                    // 暂定为每个簇中每max_connect个向量即选出一个代表向量进入下一层
                    if (every_layer_neighbors.size() != 1 &&
                        base_cluster->selected_vectors.size() <= base_cluster->vectors.size() / this->max_connect)
                    {
                        insert_to_upper_layer = true;
                        auto temporary = std::make_shared<Vector_In_Cluster>(new_vector->global_offset);
                        temporary->lower_layer = new_vector;
                        new_vector = temporary;
                    }
                }
                else
                {
                    // 计算邻居向量是否指向新的向量
                    if (neighbor_vector->out.upper_bound(neighbor_iterator.first) != neighbor_vector->out.end())
                    {
                        if (neighbor_iterator.first < neighbor_vector->out.begin()->first)
                        {
                            uint64_t offset = 1;
                            for (auto iterator = neighbor_vector->out.begin(); iterator != neighbor_vector->out.end();
                                 ++iterator)
                            {
                                if (neighbor_iterator.first * offset * this->distance_bound < iterator->first)
                                {
                                    for (auto j = iterator; j != neighbor_vector->out.end(); ++j)
                                    {
                                        j->second.lock()->in.erase(neighbor_vector->global_offset);
                                    }
                                    neighbor_vector->out.erase(iterator, neighbor_vector->out.end());
                                    auto selected_vectors =
                                        neighbor_vector->cluster.lock()->layer.lock()->divide_a_cluster(
                                            neighbor_vector->cluster.lock());
                                    for (auto &selected_vector : selected_vectors)
                                    {
                                        this->insert(selected_vector, target_layer_number + 1);
                                    }
                                    break;
                                }
                                ++offset;
                            }
                        }
                        else if (this->max_connect == neighbor_vector->out.size())
                        {
                            neighbor_vector->out.rbegin()->second.lock()->in.erase(neighbor_vector->global_offset);
                            neighbor_vector->out.erase(std::prev(neighbor_vector->out.end()));
                            auto selected_vectors = neighbor_vector->cluster.lock()->layer.lock()->divide_a_cluster(
                                neighbor_vector->cluster.lock());
                            for (auto &selected_vector : selected_vectors)
                            {
                                this->insert(selected_vector, target_layer_number + 1);
                            }
                        }
                        neighbor_vector->out.emplace(neighbor_iterator.first, new_vector);
                        new_vector->in.emplace(neighbor_vector->global_offset, neighbor_vector);
                    }
                    // 将新的向量指向邻居向量
                    new_vector->out.emplace(neighbor_iterator.first, neighbor_iterator.second);
                    // 在邻居向量中记录指向自己的新向量
                    neighbor_vector->in.emplace(new_vector->global_offset, new_vector);
                    // 合并两个簇
                    new_vector->cluster.lock()->layer.lock()->merge_two_clusters(new_vector->cluster.lock(),
                                                                                 neighbor_vector->cluster.lock());
                    if (new_vector->cluster.lock()->layer.lock()->clusters.size() == 1)
                    {
                        while (this->layers[this->layers.size() - 1] != new_vector->cluster.lock()->layer.lock())
                        {
                            this->layers.pop_back();
                        }
                        for (auto &top_layer_cluster : new_vector->cluster.lock()->layer.lock()->clusters)
                        {
                            top_layer_cluster->selected_vectors.clear();
                        }
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

  public:
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
                this->insert(vector);
            }
        }
    }

    std::multimap<float, uint64_t> query(const std::vector<Dimension_Type> &query_vector, uint64_t top_k)
    {
        if (this->vectors.empty())
        {
            throw std::logic_error("Empty vectors in index. ");
        }
        if (query_vector.size() != this->vectors[0].data.size())
        {
            throw std::invalid_argument("The dimension of query vector is not "
                                        "equality with vectors in index. ");
        }
        std::multimap<float, uint64_t> result;
        // 如果索引中的向量数量小于top-k
        // 直接暴力搜索返回排序后的全部结果
        if (this->vectors.size() < top_k)
        {
            for (uint64_t global_offset = 0; global_offset < this->vectors.size(); ++global_offset)
            {
                result.emplace(this->distance_calculation(query_vector, this->vectors[global_offset].data),
                               global_offset);
            }
        }
        else
        {
            // 记录被插入向量每一层中距离最近的top_k个邻居向量
            std::multimap<float, std::weak_ptr<Vector_In_Cluster>> every_layer_neighbors =
                this->nearest_neighbors(query_vector, this->layers[0]->clusters[0]->vectors.begin()->second, top_k);
            // 逐层扫描
            // 因为Vector_InCluster中每个向量记录了自己在下层中对应的向量
            // 所以不需要实际的层和簇
            // 直接通过上一层中返回的结果即可进行计算
            for (auto i = 1; i < this->layers.size(); ++i)
            {
                // 一层中有好多的簇
                // 每个簇之间是不连通的
                // 所以要进行多次计算
                // 最后汇总计算结果
                std::multimap<float, std::weak_ptr<Vector_In_Cluster>> one_layer_neighbors;
                for (auto &start_vector_iterator : every_layer_neighbors)
                {
                    auto temporary_nearest_neighbors = this->nearest_neighbors(
                        query_vector, start_vector_iterator.second.lock()->lower_layer.lock(), top_k);
                    for (auto &neighbor_iterator : temporary_nearest_neighbors)
                    {
                        one_layer_neighbors.insert(neighbor_iterator);
                    }
                    auto last_neighbor = one_layer_neighbors.begin();
                    std::advance(last_neighbor, this->max_connect);
                    one_layer_neighbors.erase(last_neighbor, one_layer_neighbors.end());
                }
                every_layer_neighbors.clear();
                std::swap(every_layer_neighbors, one_layer_neighbors);
            }
            for (auto &nearest_vector_iterator : every_layer_neighbors)
            {
                result.emplace(nearest_vector_iterator.first, nearest_vector_iterator.second.lock()->global_offset);
            }
        }
        return result;
    }

    void insert(const std::vector<Dimension_Type> &inserted_vector)
    {
        // 插入向量在原始数据中的偏移量
        uint64_t inserted_vector_global_offset = this->vectors.size();
        if (inserted_vector_global_offset == 0)
        {
            this->vectors.push_back(Vector<Dimension_Type>(inserted_vector));
            this->layers.push_back(std::make_shared<Layer>());
            this->layers[0]->clusters.push_back(std::make_shared<Cluster>(this->layers[0]));
            auto new_vector = std::make_shared<Vector_In_Cluster>(inserted_vector_global_offset);
            new_vector->cluster = this->layers[0]->clusters[0];
            this->layers[0]->clusters[0]->vectors.emplace(inserted_vector_global_offset, new_vector);
            return;
        }
        if (inserted_vector.size() != this->vectors[0].data.size())
        {
            throw std::invalid_argument("The dimension of insert vector is not "
                                        "equality with vectors in index. ");
        }
        this->vectors.push_back(Vector<Dimension_Type>(inserted_vector));
        auto new_vector = std::make_shared<Vector_In_Cluster>(inserted_vector_global_offset);
        this->insert(new_vector, 0);
    }
};

} // namespace nnhnsw
