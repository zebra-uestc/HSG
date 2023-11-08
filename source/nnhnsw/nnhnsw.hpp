#ifndef NNHNSW
#define NNHNSW

#include <cinttypes>
#include <map>
#include <memory.h>
#include <memory>
#include <queue>
#include <stack>
#include <string>
#include <vector>

#include "../distance/distance.hpp"
#include "data_struct.hpp"

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
    uint64_t max_connect;
    // 最大距离倍数
    uint64_t distance_bound;
    // 距离计算
    float (*distance_calculation)(const std::vector<Dimension_Type> &vector1,
                                  const std::vector<Dimension_Type> &vector2);

  public:
    Index(std::vector<std::vector<Dimension_Type>> &vectors,
          const Distance_Type distance_type = Distance_Type::Euclidean2,
          const uint64_t max_connect = 10, const uint64_t distance_bound = 1)
    {
        this->vectors.resize(vectors.size(), std::vector<Dimension_Type>(vectors[0].size()));
        for (auto i = 0; i < vectors.size(); ++i)
        {
            this->add(vectors[i]);
        }
        this->distance_bound = distance_bound;
        this->max_connect = max_connect;
        this->distance_calculation =
            get_distance_calculation_function<Dimension_Type>(distance_type);
    }

    std::map<float, uint64_t> query(const std::vector<Dimension_Type> &query_vector, uint64_t topk)
    {
        if (this->vectors.size() <= topk)
        {
            std::map<float, uint64_t> result;
            for (auto i = 0; i < this->vectors.size(); ++i)
            {
                auto distance = this->distance_calculation(query_vector, this->vectors[i].vector);
                result.insert(std::pair<float, uint64_t>(distance, i));
            }
            return result;
        }
        else
        {
            // 记录当前层中要扫描的簇
            std::map<float, uint64_t> next_scanned;
            // 记录下一层中要扫描的簇
            std::map<float, uint64_t> scanning;
            scanning.insert(std::pair<float, uint64_t>(0, 0));
            // 从最上层开始扫描
            for (auto layer_index = this->layers.size() - 1;; --layer_index)
            {
                const Layer &layer = *(this->layers[layer_index].get());
                for (auto cluster_index = scanning.begin(); cluster_index != scanning.end();
                     ++cluster_index)
                {
                    auto cluster_number = cluster_index->second;
                    Cluster &cluster = *(layer.clusters[cluster_number].get());
                    auto vectors_in_cluster = cluster.vectors;
                    for (auto vector_index = 0; vector_index < vectors_in_cluster.size();
                         ++vector_index)
                    {
                        float distance = this->distance_calculation(
                            query_vector, this->vectors[vectors_in_cluster[vector_index]].vector);
                        next_scanned.insert(
                            std::make_pair(distance, vectors_in_cluster[vector_index]));
                    }
                }
                scanning.clear();
                std::swap(next_scanned, scanning);
                if (layer_index == 0)
                {
                    break;
                }
            }
            return scanning;
        }
    }

    void add(const std::vector<Dimension_Type> &vector)
    {
        this->vectors.push_back(Vector<Dimension_Type>(vector));
        // 记录当前层中要扫描的簇
        std::map<float, uint64_t> next_scanned;
        // 记录下一层中要扫描的簇
        std::map<float, uint64_t> scanning;
        scanning.insert(std::pair<float, uint64_t>(0, 0));
        // 从最上层开始扫描
        for (auto layer_index = this->layers.size() - 1;; --layer_index)
        {
            const Layer &layer = *(this->layers[layer_index].get());
            for (auto cluster_index = scanning.begin(); cluster_index != scanning.end();
                 ++cluster_index)
            {
                auto cluster_number = cluster_index->second;
                Cluster &cluster = *(layer.clusters[cluster_number].get());
                auto vectors_in_cluster = cluster.vectors;
                for (auto vector_index = 0; vector_index < vectors_in_cluster.size();
                     ++vector_index)
                {
                    auto i = this->vectors;
                    float distance = this->distance_calculation(
                        vector, this->vectors[vectors_in_cluster[vector_index]].vector);
                    next_scanned.insert(std::make_pair(distance, vectors_in_cluster[vector_index]));
                }
            }
            scanning.clear();
            std::swap(next_scanned, scanning);
            if (layer_index == 0)
            {
                break;
            }
        }
    }

    void remove(const std::vector<Dimension_Type> &vector)
    {
    }
};

} // namespace nnhnsw

#endif
