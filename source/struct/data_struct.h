#pragma once

#include <inttypes.h>

// 向量
template <class Dimension_Type> class Vector
{
  private:
  public:
    std::vector<Dimension_Type> vector;

    Vector(const std::vector<Dimension_Type> &vectors)
    {
        this->vector = vectors;
    }
};

// 每层中的簇
class Cluster
{
  private:
  public:
    // 簇中的向量
    std::vector<uint64_t> vectors;
    // 每个向量对应下一层的簇
    std::vector<uint64_t> cluster_offset;

    Cluster()
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
    // 每个簇的被选出的向量在上一层的偏移量
    std::vector<uint64_t> vector_offset;

    Layer()
    {
    }
};

// 距离定义
enum class Distance_Type : uint64_t
{
    Euclidean2,
    Inner_Product,
    Cosine_Similarity,
};
