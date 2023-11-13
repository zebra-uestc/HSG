#pragma once

#include <cinttypes>
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
    std::vector<std::unique_ptr<Layer>> layers;
    // 每个向量的最大邻居向量个数
    uint64_t max_connect{};
    // 最大距离倍数
    uint64_t distance_bound{};
    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

  public:
    Index(const std::vector<std::vector<Dimension_Type>> &vectors, const Distance_Type distance_type,
          const uint64_t max_connect, const uint64_t distance_bound)
    {
        // 判断原始向量数据是否为空
        if (vectors.empty() || vectors.begin()->empty())
        {
            this->vectors = std::vector<Vector<Dimension_Type>>();
        }
        else
        {
            this->vectors.resize(vectors.size(), Vector<Dimension_Type>(vectors[0]));
            for (auto i = 0; i < vectors.size(); ++i)
            {
                this->insert(vectors[i]);
            }
        }
        this->distance_bound = distance_bound;
        this->max_connect = max_connect;
        this->distance_calculation = get_distance_calculation_function<Dimension_Type>(distance_type);
    }

    std::priority_queue<Query_Result, std::vector<Query_Result>, Compare_By_Distance> query(
        const std::vector<Dimension_Type> &query_vector, uint64_t topk)
    {
        if (this->vectors.empty())
        {
            throw std::logic_error("Empty vectors in index. ");
        }
        if (query_vector.size() != this->vectors[0].vector.size())
        {
            throw std::invalid_argument("The dimension of query vector is not "
                                        "equality with vectors in index. ");
        }
        std::priority_queue<Query_Result, std::vector<Query_Result>, Compare_By_Distance> result;
        if (this->vectors.size() <= topk)
        {
            for (auto i = 0; i < this->vectors.size(); ++i)
            {
                auto distance = this->distance_calculation(query_vector, this->vectors[i].vector);
                result.push(Query_Result(distance, i));
            }
        }
        else
        {
            // 记录下一层中要扫描的簇
            std::priority_queue<Query_Result, std::vector<Query_Result>, Compare_By_Distance> next_round;
            result.emplace(0, 0);
            // 从最上层开始扫描
            // todo
            // 优化查询过程
            for (auto layer_iteration = this->layers.rbegin(); layer_iteration != this->layers.rend();
                 ++layer_iteration)
            {
                while (!result.empty())
                {
                    Cluster &cluster = *(layer_iteration.clusters[result.top().offset].get());
                    for (auto &vector_in_cluster : cluster.vectors)
                    {
                        float distance = this->distance_calculation(
                            query_vector, this->vectors[vector_in_cluster.data_offset].vector);
                        if (next_round.size() < topk)
                        {
                            next_round.emplace(distance, vector_in_cluster.cluster_offset);
                        }
                        else
                        {
                            if (distance < next_round.top().distance)
                            {
                                next_round.emplace(distance, vector_in_cluster.cluster_offset);
                                next_round.pop();
                            }
                        }
                    }
                    result.pop();
                }
                std::swap(result, next_round);
            }
        }
        return result;
    }

    void insert(const std::vector<Dimension_Type> &inserted_vector)
    {
        // 插入向量在原始数据中的偏移量
        uint64_t inserted_vector_offset = this->vectors.size();
        if (inserted_vector_offset == 0)
        {
            this->vectors.push_back(Vector<Dimension_Type>(inserted_vector));
            this->layers.push_back(std::make_unique<Layer>());
            this->layers[0]->clusters.push_back(std::make_unique<Cluster>());
            this->layers[0]->clusters[0]->vectors.push_back(
                Vector_In_Cluster(inserted_vector_offset, inserted_vector_offset));
            return;
        }
        if (inserted_vector.size() != this->vectors[0].vector.size())
        {
            throw std::invalid_argument("The dimension of insert vector is not "
                                        "equality with vectors in index. ");
        }
        this->vectors.push_back(Vector<Dimension_Type>(inserted_vector));
        // 记录被插入向量每一层中距离最近的max_connect个邻居向量
        std::vector<std::multimap<float, Insert_Result>> every_layer_neighbors;
        // 从最上层开始扫描
        // todo
        // 优化查询过程
        // 记录应该被插入向量连接的邻居向量
        std::multimap<float, Insert_Result> neighbors;
        // 记录当前层中要扫描的簇
        std::multimap<float, Insert_Result> next_round;
        neighbors.insert(std::make_pair(0, Insert_Result(0, 0)));
        for (auto layer_index = this->layers.rbegin(); layer_index != this->layers.rend(); ++layer_index)
        {
            const auto &layer = *(layer_index->get());
            for (auto neighbor_iteration = neighbors.begin(); neighbor_iteration != neighbors.end();
                 ++neighbor_iteration)
            {
                const Cluster &cluster = *(layer.clusters[neighbor_iteration->second.cluster_offset].get());
                for (auto vector_offset = 0; vector_offset < cluster.vectors.size(); ++vector_offset)
                {
                    float distance = this->distance_calculation(
                        inserted_vector, this->vectors[cluster.vectors[vector_offset].data_offset].vector);
                    if (next_round.size() < this->max_connect)
                    {
                        next_round.insert(std::make_pair(
                            distance, Insert_Result(cluster.vectors[vector_offset].cluster_offset, vector_offset)));
                    }
                    else
                    {
                        if (next_round.upper_bound(distance) != next_round.end())
                        {
                            next_round.insert(std::make_pair(
                                distance, Insert_Result(cluster.vectors[vector_offset].cluster_offset, vector_offset)));
                            next_round.erase(std::prev(next_round.end()));
                        }
                    }
                }
            }
            every_layer_neighbors.push_back(std::move(neighbors));
            std::swap(neighbors, next_round);
        }

        // 插入向量
        {
            auto layer_number = 0;
            for (auto every_layer_neighbors_iteration = every_layer_neighbors.rbegin();
                 every_layer_neighbors_iteration != every_layer_neighbors.rend();
                 ++every_layer_neighbors_iteration, ++layer_number)
            {
                float shortest_distance = every_layer_neighbors_iteration->begin()->first;
                auto base_cluster_number = neighbors.begin()->second.cluster_offset;
                auto distance_rank = 1;
                for (auto neighbor_iteration = every_layer_neighbors_iteration->begin();
                     neighbor_iteration != every_layer_neighbors_iteration->end();
                     ++neighbor_iteration, ++distance_rank)
                {
                    // 以与最近的向量的距离为基准距离
                    // 如果基准距离*系数*距离排名 < 距离
                    // 则不将该向量及其后面的向量视为邻居
                    if (shortest_distance * this->distance_bound * distance_rank < neighbor_iteration->first)
                    {
                        break;
                    }
                    auto cluster_number = neighbor_iteration->second.cluster_offset;
                    // 如果邻居向量在不同的簇中，则合并簇
                    if (base_cluster_number != cluster_number)
                    {
                        for (auto upper_layer_vector_iteration =
                                 this->layers[layer_number]->clusters[cluster_number]->selected_vectors_offset.begin();
                             upper_layer_vector_iteration !=
                             this->layers[layer_number]->clusters[cluster_number]->selected_vectors_offset.end();
                             ++upper_layer_vector_iteration)
                        {
                            this->layers[layer_number + 1]
                                ->clusters[upper_layer_vector_iteration->first]
                                ->vectors[upper_layer_vector_iteration->second]
                                .cluster_offset += base_cluster_number;
                        }
                        this->layers[layer_number]->merge_two_clusters(base_cluster_number, cluster_number);
                    }
                    else
                    {
                        // 添加
                        this->layers[layer_number]->clusters[base_cluster_number]->vectors.push_back(
                            Vector_In_Cluster(inserted_vector_offset, inserted_vector_offset));
                        this->layers[layer_number]->clusters[base_cluster_number]->vectors.rbegin()->point_to.push_back(
                            neighbor_iteration->second.offset_in_cluster);
                    }
                }
            }
        }
    }

    void remove(const std::vector<Dimension_Type> &vector)
    {
    }
};

} // namespace nnhnsw
