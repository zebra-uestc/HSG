#pragma once
#include <math.h>
#include <vector>

namespace cosine_similarity
{
template <typename Dimension_Type>
float distance(const std::vector<Dimension_Type> &vector1,
               const std::vector<Dimension_Type> &vector2)
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
