#pragma once

#include <cinttypes>
#include <memory>
#include <vector>

// 向量
template <typename Dimension_Type> class Vector
{
  private:
  public:
    std::vector<Dimension_Type> vector;

    Vector() = default;

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
    // 邻居向量在簇中的偏移量
    std::vector<uint64_t> neighbors;

    explicit Vector_In_Cluster(uint64_t data_offset, uint64_t cluster_offset, uint64_t max_connect)
    {
        this->cluster_offset = data_offset;
        this->data_offset = cluster_offset;
        this->neighbors = std::vector<uint64_t>(max_connect);
    }
};

// 每层中的簇
class Cluster
{
  private:
  public:
    // 簇中的向量
    std::vector<Vector_In_Cluster> vectors;
    // 该簇中被选的向量在上一层中所属的簇和其在簇中的偏移量
    std::vector<std::pair<uint64_t, uint64_t>> selected_vectors_offset;

    Cluster()
    {
        this->vectors = std::vector<Vector_In_Cluster>();
        this->selected_vectors_offset = std::vector<std::pair<uint64_t, uint64_t>>();
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
    // 插入向量与结果向量的距离
    float distance;
    // 结果向量所的在簇在其所在层的偏移量
    uint64_t cluster_offset;
    // 结果向量在簇中的偏移量
    uint64_t offset_in_cluster;

    Insert_Result(float distance, uint64_t cluster_offset, uint64_t offset_in_cluster)
    {
        this->distance = distance;
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
