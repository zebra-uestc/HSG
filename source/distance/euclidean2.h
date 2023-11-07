#pragma once

#include <vector>

namespace euclidean2
{
template <typename Dimension_Type>
float distance(const std::vector<Dimension_Type> &vector1,
               const std::vector<Dimension_Type> &vector2)
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
