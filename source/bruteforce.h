#pragma once

#include <cinttypes>
#include <map>
#include <vector>

#include "distance.h"

namespace bruteforce
{

template <typename Dimension_Type>
std::map<float, uint64_t> search(const std::vector<std::vector<Dimension_Type>> &vectors,
                                 const std::vector<Dimension_Type> &query, const uint64_t k)
{
    std::map<Dimension_Type, uint64_t> result;
    for (auto i = 0; i < k && i < vectors.size(); ++i)
    {
        auto distance = euclidean2::distance(vectors[i], query);
        result.emplace(distance, i);
    }
    for (auto i = k; i < vectors.size(); ++i)
    {
        float distance = euclidean2::distance<Dimension_Type>(vectors[i], query);
        if (result.upper_bound(distance) != result.end())
        {
            result.emplace(distance, i);
            result.erase(std::prev(result.end()));
        }
    }
    return result;
}

} // namespace bruteforce
