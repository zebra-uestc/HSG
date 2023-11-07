#pragma once

#include <inttypes.h>
#include <map>
#include <utility>
#include <vector>

#include "../distance/euclidean2.h"

namespace bruteforce
{
std::map<float, int32_t> search(const std::vector<std::vector<float>> &vector,
                                const std::vector<float> &query, const int32_t k)
{
    std::map<float, int32_t> result;
    for (auto i = 0; i < k; ++i)
    {
        float distance = euclidean2::distance(vector[i], query);
        result.insert(std::pair<float, int32_t>(distance, i));
    }
    for (auto i = k; i < vector.size(); ++i)
    {
        float distance = euclidean2::distance(vector[i], query);
        if (result.upper_bound(distance) != result.end())
        {
            result.insert(std::pair<float, int32_t>(distance, i));
            result.erase(std::prev(result.end()));
        }
    }
    return result;
}
} // namespace bruteforce