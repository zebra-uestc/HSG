#pragma once

#include <cinttypes>
#include <iterator>
#include <map>
#include <memory>
#include <set>
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
    uint64_t data_offset;
    // 向量对应的下一层中的簇的偏移量
    // 最下一层为向量在原始数据的偏移量
    uint64_t cluster_offset;
    // 指向的邻居向量在簇中的偏移量
    std::map<float, uint64_t> point_to;
    // 指向该向量的邻居向量在簇中的偏移量
    std::unordered_set<uint64_t> be_pointed;

    explicit Vector_In_Cluster(uint64_t data_offset, uint64_t cluster_offset)
    {
        this->cluster_offset = data_offset;
        this->data_offset = cluster_offset;
        this->point_to = std::map<float, uint64_t>();
        this->be_pointed = std::unordered_set<uint64_t>();
    }
};

// 每层中的簇
class Cluster
{
  private:
  public:
    // 簇中的向量
    std::vector<Vector_In_Cluster> vectors;
    // 该簇中被选出的向量在簇中的偏移量和在上一层中所属的簇以及在簇中的偏移量
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> selected_vectors_offset;

    Cluster()
    {
        this->vectors = std::vector<Vector_In_Cluster>();
        this->selected_vectors_offset = std::map<uint64_t, std::pair<uint64_t, uint64_t>>();
    }

    // 计算当前簇的连通性
    // 如果不连通则返回所有的簇
    // 如果连通返回空的数组
    std::vector<std::unique_ptr<Cluster>> calculate_clusters()
    {
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
    void merge_two_clusters(uint64_t base_cluster, uint64_t merged_cluster)
    {
        auto base_cluster_size = this->clusters[base_cluster]->vectors.size();
        std::unordered_set<uint64_t> temporary;
        for (auto vector_iteration = this->clusters[merged_cluster]->vectors.begin();
             vector_iteration != this->clusters[merged_cluster]->vectors.end(); ++vector_iteration)
        {
            // 被合并的簇中的每个向量指向的向量在簇中所在的偏移量增加目标簇中向量的个数
            for (auto point_to_iteration = vector_iteration->point_to.begin();
                 point_to_iteration != vector_iteration->point_to.end(); ++point_to_iteration)
            {
                point_to_iteration->second += base_cluster_size;
            }

            // 被合并的簇中的每个向量被指向的向量在簇中所在的偏移量增加目标簇中向量的个数
            for (auto be_pointed_iteration = vector_iteration->be_pointed.begin();
                 be_pointed_iteration != vector_iteration->be_pointed.end(); ++be_pointed_iteration)
            {
                temporary.insert(*be_pointed_iteration + base_cluster_size);
            }
            std::swap(vector_iteration->be_pointed, temporary);
            temporary.clear();
        }
        this->clusters[base_cluster]->vectors.insert(this->clusters[base_cluster]->vectors.end(),
                                                     this->clusters[merged_cluster]->vectors.begin(),
                                                     this->clusters[merged_cluster]->vectors.end());
        for (auto selected_vector_in_merged_cluster_iteration =
                 this->clusters[merged_cluster]->selected_vectors_offset.begin();
             selected_vector_in_merged_cluster_iteration !=
             this->clusters[merged_cluster]->selected_vectors_offset.end();
             ++selected_vector_in_merged_cluster_iteration)
        {
            this->clusters[base_cluster]->selected_vectors_offset.insert(
                std::make_pair(selected_vector_in_merged_cluster_iteration->first + base_cluster_size,
                               std::make_pair(selected_vector_in_merged_cluster_iteration->second.first,
                                              selected_vector_in_merged_cluster_iteration->second.second)));
        }
        this->clusters.erase(this->clusters.begin() + merged_cluster);
    }

    // 分裂一个簇
    void divide_a_cluster(uint64_t cluster_number)
    {
        auto new_clusters = this->clusters[cluster_number]->calculate_clusters();
        if (!new_clusters.empty())
        {
            std::swap(this->clusters[cluster_number], new_clusters[0]);
            this->clusters.resize(this->clusters.size() + new_clusters.size() - 1);
            for (auto new_cluster_iteration = new_clusters.begin() + 1; new_cluster_iteration != new_clusters.end();
                 ++new_cluster_iteration)
            {
                this->clusters.push_back(std::move(*new_cluster_iteration));
            }
        }
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

class Insert_Result
{
  private:
  public:
    // 结果向量所在的簇在其所在层的偏移量
    uint64_t cluster_offset;
    // 结果向量在簇中的偏移量
    uint64_t offset_in_cluster;

    Insert_Result(uint64_t cluster_offset, uint64_t offset_in_cluster)
    {
        this->cluster_offset = cluster_offset;
        this->offset_in_cluster = offset_in_cluster;
    }
};

// 自定义仿函数
struct Compare_By_Distance
{
    bool operator()(const Query_Result &result1, const Query_Result &result2)
    {
        return result1.distance < result2.distance;
    }
};
