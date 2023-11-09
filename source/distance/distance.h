#ifndef DISTANCE_TYPE
#define DISTANCE_TYPE

#include <cinttypes>
#include <cmath>
#include <vector>

// 距离定义
enum class Distance_Type : uint64_t
{
    Euclidean2,
    Inner_Product,
    Cosine_Similarity,
};

namespace euclidean2
{
template <typename Dimension_Type>
float distance(const std::vector<Dimension_Type> &vector1, const std::vector<Dimension_Type> &vector2)
{
    float square_distance = 0;
    for (size_t i = 0; i < vector1.size(); ++i)
    {
        square_distance += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
    }
    // 根据距离定义应开方，但是不影响距离对比所以省略
    return square_distance;
}
} // namespace euclidean2

namespace inner_product
{
template <typename Dimension_Type>
float distance(const std::vector<Dimension_Type> &vector1, const std::vector<Dimension_Type> &vector2)
{
    float product = 0;
    for (size_t i = 0; i < vector1.size(); ++i)
    {
        product += vector1[i] * vector2[i];
    }
    return 1 - product;
}
} // namespace inner_product

namespace cosine_similarity
{
template <typename Dimension_Type>
float distance(const std::vector<Dimension_Type> &vector1, const std::vector<Dimension_Type> &vector2)
{
    float product = 0;
    float square_sum1 = 0;
    float square_sum2 = 0;
    for (size_t i = 0; i < vector1.size(); ++i)
    {
        product += vector1[i] * vector2[i];
        square_sum1 += vector1[i] * vector1[i];
        square_sum2 += vector2[i] * vector2[i];
    }
    return 1 - product / sqrt((square_sum1 * square_sum2));
}
} // namespace cosine_similarity

template <typename Dimension_Type> auto get_distance_calculation_function(Distance_Type distance)
{
    switch (distance)
    {
    case Distance_Type::Euclidean2:
        return euclidean2::distance<Dimension_Type>;
        break;
    case Distance_Type::Inner_Product:
        return inner_product::distance<Dimension_Type>;
        break;
    case Distance_Type::Cosine_Similarity:
        return cosine_similarity::distance<Dimension_Type>;
        break;
    default:
        break;
    }
}

#endif
