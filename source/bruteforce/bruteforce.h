#ifndef BRUTEFORCE
#define BRUTEFORCE

#include <inttypes.h>
#include <map>
#include <vector>

#include "../distance/distance.h"

namespace bruteforce
{
template <typename Dimension_Type>
std::map<float, int32_t> search(const std::vector<std::vector<Dimension_Type>> &vector,
                                const std::vector<Dimension_Type> &query, const int32_t k)
{
    std::map<Dimension_Type, int32_t> result;
    for (auto i = 0; i < k; ++i)
    {
        float distance = euclidean2::distance<Dimension_Type>(vector[i], query);
        result.insert(std::pair<float, int32_t>(distance, i));
    }
    for (auto i = k; i < vector.size(); ++i)
    {
        float distance = euclidean2::distance<Dimension_Type>(vector[i], query);
        if (result.upper_bound(distance) != result.end())
        {
            result.insert(std::pair<float, int32_t>(distance, i));
            result.erase(std::prev(result.end()));
        }
    }
    return result;
}
} // namespace bruteforce

#endif
