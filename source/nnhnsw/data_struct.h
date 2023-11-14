#pragma once

#include <cinttypes>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// 向量
template <typename Dimension_Type> class Vector
{
  private:
  public:
    std::vector<Dimension_Type> vector;

    explicit Vector(const std::vector<Dimension_Type> &vector)
    {
        this->vector = vector;
    }
};

// 簇中的向量
class Vector_In_Cluster
{
  private:
  public:
    // 向量在原始数据的偏移量
    uint64_t global_offset;
    // 向量对应的下一层中向量
    std::weak_ptr<Vector_In_Cluster> lower_layer;
    // 指向的邻居向量
    std::multimap<float, std::weak_ptr<Vector_In_Cluster>> out;
    // 指向该向量的邻居向量
    std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>> in;

    explicit Vector_In_Cluster(uint64_t global_offset)
    {
        this->global_offset = global_offset;
        this->out = std::multimap<float, std::weak_ptr<Vector_In_Cluster>>();
        this->in = std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>>();
    }
};

// 每层中的簇
class Cluster
{
  private:
  public:
    // 簇中的向量
    std::unordered_map<uint64_t, std::shared_ptr<Vector_In_Cluster>> vectors;
    // 该簇中被选出的向量在原始数据的偏移量和在上一层的向量
    std::unordered_set<uint64_t> selected_vectors;

    Cluster()
    {
        this->vectors = std::unordered_map<uint64_t, std::shared_ptr<Vector_In_Cluster>>();
        this->selected_vectors = std::unordered_set<uint64_t>();
    }

    // 计算当前簇的连通性
    // 如果不连通则返回所有的簇
    std::vector<std::unique_ptr<Cluster>> calculate_clusters()
    {
        uint64_t hit = 0;
        std::vector<std::unique_ptr<Cluster>> new_clusters;
        // key：向量的全局偏移量
        // value：向量在第几个新的簇中
        std::unordered_map<uint64_t, uint64_t> flag;
        uint64_t new_cluster_number = 0;
        for (auto vector_iteration = this->vectors.begin();
             vector_iteration != this->vectors.end() && hit != this->vectors.size(); ++vector_iteration)
        {
            if (!flag.contains(vector_iteration->first))
            {
                ++hit;
                flag.insert(std::make_pair(vector_iteration->first, new_cluster_number));
                std::unique_ptr<Cluster> temporary_cluster = std::make_unique<Cluster>();
                temporary_cluster->vectors.insert(std::make_pair(vector_iteration->first, vector_iteration->second));
                std::queue<uint64_t> waiting_vectors;
                waiting_vectors.push(vector_iteration->first);
                while (!waiting_vectors.empty())
                {
                    uint64_t neighbor_iteration = waiting_vectors.front();
                    waiting_vectors.pop();
                    for (auto &out_iteration : this->vectors.find(neighbor_iteration)->second->out)
                    {
                        if (!flag.contains(out_iteration.second.lock()->global_offset))
                        {
                            ++hit;
                            flag.insert(std::make_pair(out_iteration.second.lock()->global_offset, new_cluster_number));
                            temporary_cluster->vectors.insert(
                                std::make_pair(out_iteration.second.lock()->global_offset, out_iteration.second));
                        }
                    }
                    for (auto &in_iteration : this->vectors.find(neighbor_iteration)->second->in)
                    {
                        if (!flag.contains(in_iteration.first))
                        {
                            ++hit;
                            flag.insert(std::make_pair(in_iteration.first, new_cluster_number));
                            temporary_cluster->vectors.insert(std::make_pair(in_iteration.first, in_iteration.second));
                        }
                    }
                }
                new_clusters.push_back(std::move(temporary_cluster));
                ++new_cluster_number;
            }
        }
        for (auto &selected_vector : this->selected_vectors)
        {
            new_clusters[flag.find(selected_vector)->second]->selected_vectors.insert(selected_vector);
        }
        return new_clusters;
    }
};

// 索引中的一层
class Layer
{
  private:
  public:
    // 每层中的多个簇
    std::vector<std::unique_ptr<Cluster>> clusters;

    Layer()
    {
        this->clusters = std::vector<std::unique_ptr<Cluster>>();
    }

    // 合并两个簇
    void merge_two_clusters(uint64_t target_cluster, uint64_t merged_cluster)
    {
        // 移动被合并的簇中的向量到目标簇中
        for (auto vector_iteration = this->clusters[merged_cluster]->vectors.begin();
             vector_iteration != this->clusters[merged_cluster]->vectors.end(); ++vector_iteration)
        {
            this->clusters[target_cluster]->vectors.insert(
                std::make_pair(vector_iteration->first, vector_iteration->second));
        }
        // 移动被合并的簇中被选出的代表向量到目标簇中且更改这些向量的“represented_cluster_offset”的值
        for (auto selected_vector_iteration = this->clusters[merged_cluster]->selected_vectors.begin();
             selected_vector_iteration != this->clusters[merged_cluster]->selected_vectors.end();
             ++selected_vector_iteration)
        {
            this->clusters[target_cluster]->selected_vectors.insert(*selected_vector_iteration);
        }
        // 删除簇
        this->clusters.erase(this->clusters.begin() + merged_cluster);
    }

    // 分裂一个簇
    // 新的簇中可能没有被选出的向量在上一层中
    // 将这些簇的编号返回
    std::vector<uint64_t> divide_a_cluster(uint64_t cluster_number)
    {
        auto new_clusters = this->clusters[cluster_number]->calculate_clusters();
        if (new_clusters.size() != 1)
        {
            std::vector<uint64_t> no_selected_clusters;
            if (new_clusters[0]->selected_vectors.empty())
            {
                no_selected_clusters.push_back(cluster_number);
            }
            std::swap(this->clusters[cluster_number], new_clusters[0]);
            for (auto new_cluster_iteration = new_clusters.begin() + 1; new_cluster_iteration != new_clusters.end();
                 ++new_cluster_iteration)
            {
                if ((*new_cluster_iteration)->selected_vectors.empty())
                {
                    no_selected_clusters.push_back(this->clusters.size());
                }
                this->clusters.push_back(std::move(*new_cluster_iteration));
            }
        }
        return {};
    }
};

class Query_Result
{
  private:
  public:
    // 查询向量与结果向量的距离
    float distance;
    // 结果向量指向的簇在其所在层的偏移量
    // 或者
    // 结果向量在原始数据中的偏移量
    uint64_t offset;

    Query_Result(float distance, uint64_t offset)
    {
        this->distance = distance;
        this->offset = offset;
    }
};

// class Insert_Result
//{
//   private:
//   public:
//     // 结果向量所在的簇在其所在层的偏移量
//     uint64_t cluster_offset;
//     // 结果向量在簇中的偏移量
//     uint64_t offset_in_cluster;
//
//     Insert_Result(uint64_t cluster_offset, uint64_t offset_in_cluster)
//     {
//         this->cluster_offset = cluster_offset;
//         this->offset_in_cluster = offset_in_cluster;
//     }
// };

// 自定义仿函数
struct Compare_By_Distance
{
    bool operator()(const Query_Result &result1, const Query_Result &result2)
    {
        return result1.distance < result2.distance;
    }
};
