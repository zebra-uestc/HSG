#pragma once
#include <vector>

namespace inner_product
{
template <typename Dimension_Type>
float distance(const std::vector<Dimension_Type> &vector1,
               const std::vector<Dimension_Type> &vector2)
{
    float product = 0;
    for (size_t i = 0; i < vector1.size(); ++i)
    {
        product += vector1[i] * vector2[i];
    }
    return 1 - product;
}
} // namespace inner_product
