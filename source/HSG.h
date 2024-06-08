#pragma once

#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "space.h"

namespace HSG
{

    // 向量内部的唯一标识符
    // 顶点偏移量
    // index.vectors[offset]
    using Offset = uint64_t;

    // 向量外部的唯一标识符
    using ID = uint64_t;

    // 向量
    class Vector
    {
      public:
        // 向量的外部id
        ID id;
        // 向量内部的唯一标识符
        Offset offset;
        // 向量的数据
        const float *data;
        //
        float zero;
        // 短的出边
        std::multimap<float, Offset> short_edge_out;
        // 短的入边
        std::unordered_map<Offset, float> short_edge_in;
        // 长的出边
        std::unordered_map<Offset, float> long_edge_out;
        // 长的入边
        std::unordered_map<Offset, float> long_edge_in;
        //
        std::unordered_set<Offset> keep_connected;

        explicit Vector(const ID id, Offset offset, const float *const data_address, float zero)
            : id(id), offset(offset), data(data_address), zero(zero)
        {
        }
    };

    class Index_Parameters
    {
      public:
        // 向量的维度
        uint64_t dimension;
        // 距离类型
        Space::Metric space_metric;
        //
        uint64_t magnification;
        // 短边数量下限
        uint64_t short_edge_lower_limit;
        // 短边数量上限
        uint64_t short_edge_upper_limit;
        // 覆盖范围
        uint64_t cover_range;

        explicit Index_Parameters(const uint64_t dimension, const Space::Metric space_metric,
                                  const uint64_t magnification, const uint64_t short_edge_lower_limit,
                                  const uint64_t short_edge_upper_limit, const uint64_t cover_range)
            : dimension(dimension), space_metric(space_metric), magnification(magnification),
              short_edge_lower_limit(short_edge_lower_limit), short_edge_upper_limit(short_edge_upper_limit),
              cover_range(cover_range)
        {
        }
    };

    // 索引
    //
    // 索引使用零点作为默认起始点
    //
    // 使用64位无符号整形的最大值作为零点的id
    //
    // 所以向量的id应大于等于0且小于64位无符号整形的最大值
    class Index
    {
      public:
        // 索引的参数
        Index_Parameters parameters;
        // 距离计算
        float (*similarity)(const float *vector1, const float *vector2, uint64_t dimension);
        // 索引中向量的数量
        uint64_t count;
        // 索引中的向量
        std::vector<Vector> vectors;
        // 记录存放向量的数组中的空位
        std::stack<uint64_t> empty;
        // 零点向量
        std::vector<float> zero;
        // 记录向量的 id 和 offset 的对应关系
        std::unordered_map<ID, Offset> id_to_offset;

        explicit Index(const Space::Metric space, const uint64_t dimension, const uint64_t short_edge_lower_limit,
                       const uint64_t short_edge_upper_limit, const uint64_t cover_range, const uint64_t magnification)
            : parameters(dimension, space, magnification, short_edge_lower_limit, short_edge_upper_limit, cover_range),
              similarity(Space::get_similarity(space)), count(1), zero(dimension, 0.0)
        {
            this->vectors.push_back(Vector(std::numeric_limits<uint64_t>::max(), 0, this->zero.data(), 0));
            this->id_to_offset.insert({std::numeric_limits<uint64_t>::max(), 0});
        }
    };

    inline Offset Get_Offset(const Index &index, const ID id)
    {
        return index.id_to_offset.find(id)->second;
    }

    inline void Delete_Vector(Index &index, const Offset offset)
    {
        auto &vector = index.vectors[offset];

        vector.data = nullptr;
        vector.short_edge_in.clear();
        vector.short_edge_out.clear();
        vector.keep_connected.clear();
        vector.long_edge_in.clear();
        vector.long_edge_out.clear();
    }

    inline bool Adjacent(const Index &index, const Offset offset1, const Offset offset2)
    {
        const auto &v1 = index.vectors[offset1];
        const auto &v2 = index.vectors[offset2];

        if (v1.keep_connected.contains(v2.offset))
        {
            return true;
        }

        if (v1.short_edge_in.contains(v2.offset))
        {
            return true;
        }

        if (v2.short_edge_in.contains(v1.offset))
        {
            return true;
        }

        return false;
    }

    inline void Get_Pool_From_LEO(const Index &index, const Offset processing_offset, std::vector<bool> &visited,
                                  std::vector<Offset> &pool)
    {
        const auto &processing_vector = index.vectors[processing_offset];

        for (auto iterator = processing_vector.long_edge_out.begin(); iterator != processing_vector.long_edge_out.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->first;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                pool.push_back(neighbor_offset);
            }
        }
    }

    inline void Get_Pool_From_SE(const Index &index, const Offset processing_offset, std::vector<bool> &visited,
                                 std::vector<Offset> &pool)
    {
        const auto &processing_vector = index.vectors[processing_offset];

        for (auto iterator = processing_vector.short_edge_out.begin();
             iterator != processing_vector.short_edge_out.end(); ++iterator)
        {
            const auto &neighbor_offset = iterator->second;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                pool.push_back(neighbor_offset);
            }
        }

        for (auto iterator = processing_vector.short_edge_in.begin(); iterator != processing_vector.short_edge_in.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->first;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                pool.push_back(neighbor_offset);
            }
        }

        for (auto iterator = processing_vector.keep_connected.begin();
             iterator != processing_vector.keep_connected.end(); ++iterator)
        {
            const auto &neighbor_offset = *iterator;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                pool.push_back(neighbor_offset);
            }
        }
    }

    inline void Prefetch(const float *const data)
    {
#if defined(__SSE__)
        _mm_prefetch(data, _MM_HINT_T0);
#endif
    }

    inline bool Similarity(const Index &index, const float *const target_vector, std::vector<Offset> &pool,
                           std::priority_queue<std::pair<float, ID>> &nearest_neighbors,
                           std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                               std::greater<>> &waiting_vectors)
    {
        if (!pool.empty())
        {
            auto farest_offset = std::numeric_limits<uint64_t>::max();

            if (nearest_neighbors.size() == index.parameters.magnification)
            {
                farest_offset = nearest_neighbors.top().second;
            }

            const auto number = pool.size() - 1;

            Prefetch(index.vectors[pool.front()].data);

            for (auto i = 0; i < number; ++i)
            {
                const auto &neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto &next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];
                const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

                if (nearest_neighbors.size() < index.parameters.magnification)
                {
                    nearest_neighbors.push({distance, neighbor_offset});
                }
                else if (distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.pop();
                    nearest_neighbors.push({distance, neighbor_offset});
                }

                Prefetch(next_vector.data);
                waiting_vectors.push({distance, neighbor_offset});
            }

            auto &neighbor_offset = pool.back();
            auto &neighbor_vector = index.vectors[neighbor_offset];
            auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            if (nearest_neighbors.size() < index.parameters.magnification)
            {
                nearest_neighbors.push({distance, neighbor_offset});
            }
            else if (distance < nearest_neighbors.top().first)
            {
                nearest_neighbors.pop();
                nearest_neighbors.push({distance, neighbor_offset});
            }

            waiting_vectors.push({distance, neighbor_offset});
            pool.clear();

            if (farest_offset == std::numeric_limits<uint64_t>::max() ||
                farest_offset != nearest_neighbors.top().second)
            {
                return false;
            }

            return true;
        }

        return true;
    }

    inline bool Similarity_Add(const Index &index, const float *const target_vector, std::vector<Offset> &pool,
                               std::priority_queue<std::pair<float, ID>> &nearest_neighbors,
                               std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                   std::greater<>> &waiting_vectors,
                               std::vector<std::pair<Offset, float>> &all)
    {
        if (!pool.empty())
        {
            auto farest_offset = std::numeric_limits<uint64_t>::max();

            if (nearest_neighbors.size() == index.parameters.magnification)
            {
                farest_offset = nearest_neighbors.top().second;
            }

            const auto number = pool.size() - 1;

            Prefetch(index.vectors[pool.front()].data);

            for (auto i = 0; i < number; ++i)
            {
                const auto &neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto &next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];
                const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

                if (nearest_neighbors.size() < index.parameters.magnification)
                {
                    nearest_neighbors.push({distance, neighbor_offset});
                }
                else if (distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.pop();
                    nearest_neighbors.push({distance, neighbor_offset});
                }

                waiting_vectors.push({distance, neighbor_offset});

                if (neighbor_vector.short_edge_out.size() < index.parameters.short_edge_lower_limit ||
                    distance < neighbor_vector.short_edge_out.rbegin()->first)
                {
                    all.push_back({neighbor_offset, distance});
                }

                Prefetch(next_vector.data);
            }

            auto &neighbor_offset = pool.back();
            auto &neighbor_vector = index.vectors[neighbor_offset];
            auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            if (nearest_neighbors.size() < index.parameters.magnification)
            {
                nearest_neighbors.push({distance, neighbor_offset});
            }
            else if (distance < nearest_neighbors.top().first)
            {
                nearest_neighbors.pop();
                nearest_neighbors.push({distance, neighbor_offset});
            }

            waiting_vectors.push({distance, neighbor_offset});

            if (neighbor_vector.short_edge_out.size() < index.parameters.short_edge_lower_limit ||
                distance < neighbor_vector.short_edge_out.rbegin()->first)
            {
                all.push_back({neighbor_offset, distance});
            }

            pool.clear();

            if (farest_offset == std::numeric_limits<uint64_t>::max() ||
                farest_offset != nearest_neighbors.top().second)
            {
                return false;
            }

            return true;
        }

        return true;
    }

    // 查询距离目标向量最近的k个向量
    //
    // k = index.parameters.short_edge_lower_limit
    //
    // 返回最近邻和不属于最近邻但是在路径上的顶点
    inline void Search_Add(const Index &index, const Offset offset, std::vector<std::pair<float, Offset>> &long_path,
                           uint64_t &short_path_length,
                           std::priority_queue<std::pair<float, Offset>> &nearest_neighbors,
                           std::vector<std::pair<Offset, float>> &all)
    {
        const auto &new_vector = index.vectors[offset];

        // 等待队列
        auto waiting_vectors =
            std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();

        waiting_vectors.push({new_vector.zero, 0});
        long_path.push_back({new_vector.zero, 0});

        const auto &zero_vector = index.vectors.front();

        if (zero_vector.short_edge_out.size() < index.parameters.short_edge_lower_limit ||
            new_vector.zero < zero_vector.short_edge_out.rbegin()->first)
        {
            all.push_back({0, new_vector.zero});
        }

        // 标记是否被遍历过
        auto visited = std::vector<bool>(index.vectors.size(), false);

        visited[0] = true;

        // 计算池子
        auto pool = std::vector<Offset>();

        while (true)
        {
            auto processing_offset = waiting_vectors.top().second;

            long_path.push_back(waiting_vectors.top());
            waiting_vectors.pop();
            Get_Pool_From_SE(index, processing_offset, visited, pool);
            Similarity_Add(index, new_vector.data, pool, nearest_neighbors, waiting_vectors, all);

            const auto short_offset = waiting_vectors.top().second;

            Get_Pool_From_LEO(index, processing_offset, visited, pool);
            Similarity_Add(index, new_vector.data, pool, nearest_neighbors, waiting_vectors, all);

            const auto &nearest_offset = waiting_vectors.top().second;

            if (short_offset == nearest_offset)
            {
                break;
            }
        }

        // 阶段二：
        // 利用短边找到和目标向量最近的向量
        while (true)
        {
            const auto processing_offset = waiting_vectors.top().second;

            waiting_vectors.pop();
            Get_Pool_From_SE(index, processing_offset, visited, pool);
            Similarity_Add(index, new_vector.data, pool, nearest_neighbors, waiting_vectors, all);

            const auto &nearest_offset = waiting_vectors.top().second;

            if (processing_offset == nearest_offset)
            {
                break;
            }

            ++short_path_length;
        }

        // 阶段三：
        // 查找与目标向量相似度最高（距离最近）的k个向量
        while (!waiting_vectors.empty())
        {
            const auto &processing_offset = waiting_vectors.top().second;

            waiting_vectors.pop();
            Get_Pool_From_SE(index, processing_offset, visited, pool);

            auto end = Similarity_Add(index, new_vector.data, pool, nearest_neighbors, waiting_vectors, all);

            if (end)
            {
                break;
            }
        }

        while (index.parameters.short_edge_lower_limit < nearest_neighbors.size())
        {
            nearest_neighbors.pop();
        }
    }

    inline bool Connected(const Index &index, const Offset start, const Offset offset)
    {
        auto visited = std::unordered_set<Offset>();

        visited.insert(start);

        auto last = std::vector<Offset>();

        last.push_back(start);

        auto next = std::vector<Offset>();

        for (auto round = 0; round < 3; ++round)
        {
            for (auto iterator = last.begin(); iterator != last.end(); ++iterator)
            {
                const auto &t = index.vectors[*iterator];

                for (auto iterator = t.short_edge_in.begin(); iterator != t.short_edge_in.end(); ++iterator)
                {
                    const auto &t1 = iterator->first;

                    if (!visited.contains(t1))
                    {
                        visited.insert(t1);
                        next.push_back(t1);
                    }
                }

                for (auto iterator = t.short_edge_out.begin(); iterator != t.short_edge_out.end(); ++iterator)
                {
                    const auto &t1 = iterator->second;

                    if (!visited.contains(t1))
                    {
                        visited.insert(t1);
                        next.push_back(t1);
                    }
                }

                for (auto iterator = t.keep_connected.begin(); iterator != t.keep_connected.end(); ++iterator)
                {
                    const auto &t1 = *iterator;

                    if (!visited.contains(t1))
                    {
                        visited.insert(t1);
                        next.push_back(t1);
                    }
                }
            }

            if (visited.contains(offset))
            {
                return true;
            }

            std::swap(last, next);
            next.clear();
        }

        for (auto iterator = last.begin(); iterator != last.end(); ++iterator)
        {
            const auto &t = index.vectors[*iterator];

            for (auto iterator = t.short_edge_in.begin(); iterator != t.short_edge_in.end(); ++iterator)
            {
                const auto &t1 = iterator->first;

                visited.insert(t1);
            }

            for (auto iterator = t.short_edge_out.begin(); iterator != t.short_edge_out.end(); ++iterator)
            {
                const auto &t1 = iterator->second;

                visited.insert(t1);
            }

            for (auto iterator = t.keep_connected.begin(); iterator != t.keep_connected.end(); ++iterator)
            {
                const auto &t1 = *iterator;

                visited.insert(t1);
            }
        }

        if (visited.contains(offset))
        {
            return true;
        }

        return false;
    }

    // 计算角A的余弦值
    //
    // 余弦值没有除以2
    inline float Cosine_Value(const float a, const float b, const float c)
    {
        return (b * b + c * c - a * a) / (b * c);
    }

    // 添加长边
    inline void Add_Long_Edges(Index &index, const std::vector<std::pair<float, Offset>> &long_path,
                               const uint64_t short_path_length, const Offset offset)
    {

        if (index.parameters.cover_range < short_path_length)
        {
            auto &vector = index.vectors[offset];
            float added_distance = 0;
            float maximum_cosine = -2;
            Offset added_offset = 0;

            for (int i = long_path.size() - 1; 0 < i; --i)
            {
                const auto &D = long_path[i].first;
                const auto &neighbor_offset = long_path[i].second;
                const auto &neighbor_vector = index.vectors[neighbor_offset];

                if (neighbor_vector.zero < vector.zero && !Adjacent(index, offset, neighbor_offset))
                {
                    const auto CV = Cosine_Value(added_distance, vector.zero, neighbor_vector.zero);

                    if (maximum_cosine < CV)
                    {
                        added_distance = D;
                        maximum_cosine = CV;
                        added_offset = neighbor_offset;
                    }
                }
            }

            if (1.732 < maximum_cosine)
            {
                auto &neighbor_vector = index.vectors[added_offset];

                neighbor_vector.long_edge_out.insert({offset, added_distance});
                vector.long_edge_in.insert({added_offset, added_distance});
            }
            else
            {
                auto &neighbor_vector = index.vectors.front();

                neighbor_vector.long_edge_out.insert({offset, added_distance});
                vector.long_edge_in.insert({added_offset, added_distance});
            }
        }
    }

    inline void Neighbor_Optimize(Index &index, const Offset offset, std::vector<std::pair<Offset, float>> &all)
    {
        auto &new_vector = index.vectors[offset];

        for (auto i = 0; i < all.size(); ++i)
        {
            const auto &neighbor_offset = all[i].first;
            const auto &distance = all[i].second;
            auto &neighbor = index.vectors[neighbor_offset];

            // 如果邻居向量的出边小于短边下限
            if (neighbor.short_edge_out.size() < index.parameters.short_edge_lower_limit)
            {
                // 邻居向量添加出边
                neighbor.short_edge_out.insert({distance, offset});

                // 新向量添加入边
                new_vector.short_edge_in.insert({neighbor_offset, distance});
            }
            // 如果新向量和邻居的距离小于邻居当前距离最大的出边的距离
            else if (distance < neighbor.short_edge_out.rbegin()->first)
            {
                const auto farest_distance = neighbor.short_edge_out.rbegin()->first;
                auto &neighbor_neighbor = index.vectors[neighbor.short_edge_out.rbegin()->second];

                // neighbor neighbor offset
                const auto &NN_offset = neighbor_neighbor.offset;

                // 邻居向量删除距离最大的出边
                neighbor.short_edge_out.erase(std::prev(neighbor.short_edge_out.end()));
                neighbor_neighbor.short_edge_in.erase(neighbor_offset);

                if (!neighbor.short_edge_in.contains(NN_offset))
                {
                    if (!Connected(index, neighbor_offset, NN_offset))
                    {
                        if (neighbor.short_edge_out.size() < index.parameters.short_edge_upper_limit)
                        {
                            neighbor.short_edge_out.insert({farest_distance, NN_offset});
                            neighbor_neighbor.short_edge_in.insert({neighbor_offset, farest_distance});
                        }
                        else
                        {
                            neighbor.keep_connected.insert(NN_offset);
                            neighbor_neighbor.keep_connected.insert(neighbor_offset);
                        }
                    }
                }

                // 邻居向量添加出边
                neighbor.short_edge_out.insert({distance, offset});

                // 新向量添加入边
                new_vector.short_edge_in.insert({neighbor_offset, distance});
            }
        }
    }

    // 添加
    inline void Add(Index &index, const ID id, const float *const added_vector_data)
    {
        Offset offset = index.vectors.size();
        auto zero_distance = Space::Euclidean2::zero(added_vector_data, index.parameters.dimension);

        ++index.count;

        if (index.empty.empty())
        {
            // 在索引中创建一个新向量
            index.vectors.push_back(Vector(id, offset, added_vector_data, zero_distance));
        }
        else
        {
            offset = index.empty.top();
            index.empty.pop();
            index.vectors[offset].id = id;
            index.vectors[offset].data = added_vector_data;
            index.vectors[offset].zero = zero_distance;
        }

        index.id_to_offset.insert({id, offset});

        auto &new_vector = index.vectors[offset];
        auto nearest_neighbors = std::priority_queue<std::pair<float, Offset>>();
        auto long_path = std::vector<std::pair<float, Offset>>();
        uint64_t short_path_length = 0;
        auto all = std::vector<std::pair<Offset, float>>();

        // 搜索距离新增向量最近的 index.parameters.short_edge_lower_limit 个向量
        // 同时记录搜索路径
        Search_Add(index, offset, long_path, short_path_length, nearest_neighbors, all);
        Neighbor_Optimize(index, offset, all);

        // 添加短边
        while (!nearest_neighbors.empty())
        {
            const auto &distance = nearest_neighbors.top().first;
            const auto &neighbor_offset = nearest_neighbors.top().second;
            auto &neighbor = index.vectors[neighbor_offset];

            nearest_neighbors.pop();

            // 为新向量添加出边
            new_vector.short_edge_out.insert({distance, neighbor_offset});

            // 为邻居向量添加入边
            neighbor.short_edge_in.insert({offset, distance});
        }

        Add_Long_Edges(index, long_path, short_path_length, offset);
    }

    inline void Transfer_LEO(Index &index, const Offset whose_offset, const Offset to_offset)
    {
        auto &whose_V = index.vectors[whose_offset];
        auto &to_V = index.vectors[to_offset];

        for (auto i = whose_V.long_edge_out.begin(); i != whose_V.long_edge_out.end(); ++i)
        {
            const auto &neighbor_O = i->first;
            auto &neighbor_V = index.vectors[neighbor_O];
            auto distance = index.similarity(to_V.data, neighbor_V.data, index.parameters.dimension);

            neighbor_V.long_edge_in.erase(whose_offset);
            to_V.long_edge_out.insert({neighbor_O, distance});
            neighbor_V.long_edge_in.insert({to_offset, distance});
        }

        whose_V.long_edge_out.clear();
    }

    inline void Mark_Erase(const Vector &repaired_vector, std::vector<bool> &visited)
    {
        for (auto iterator = repaired_vector.short_edge_in.begin(); iterator != repaired_vector.short_edge_in.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->first;

            visited[neighbor_offset] = true;
        }

        for (auto iterator = repaired_vector.short_edge_out.begin(); iterator != repaired_vector.short_edge_out.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->second;

            visited[neighbor_offset] = true;
        }

        for (auto iterator = repaired_vector.keep_connected.begin(); iterator != repaired_vector.keep_connected.end();
             ++iterator)
        {
            const auto &neighbor_offset = *iterator;

            visited[neighbor_offset] = true;
        }
    }

    inline void Similarity_Erase(const Index &index, const Vector &repaired_vector, std::vector<bool> &visited,
                                 std::vector<Offset> &pool,
                                 std::priority_queue<std::pair<float, ID>> &nearest_neighbors,
                                 std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                     std::greater<>> &waiting_vectors)
    {
        for (auto iterator = repaired_vector.short_edge_in.begin(); iterator != repaired_vector.short_edge_in.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->first;
            const auto &neighbor_vector = index.vectors[neighbor_offset];

            for (auto iterator = neighbor_vector.short_edge_in.begin(); iterator != neighbor_vector.short_edge_in.end();
                 ++iterator)
            {
                const auto &NNO = iterator->first;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }

            for (auto iterator = neighbor_vector.short_edge_out.begin();
                 iterator != neighbor_vector.short_edge_out.end(); ++iterator)
            {
                const auto &NNO = iterator->second;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }

            for (auto iterator = neighbor_vector.keep_connected.begin();
                 iterator != neighbor_vector.keep_connected.end(); ++iterator)
            {
                const auto &NNO = *iterator;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }
        }

        for (auto iterator = repaired_vector.short_edge_out.begin(); iterator != repaired_vector.short_edge_out.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->second;
            const auto &neighbor_vector = index.vectors[neighbor_offset];

            for (auto iterator = neighbor_vector.short_edge_in.begin(); iterator != neighbor_vector.short_edge_in.end();
                 ++iterator)
            {
                const auto &NNO = iterator->first;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }

            for (auto iterator = neighbor_vector.short_edge_out.begin();
                 iterator != neighbor_vector.short_edge_out.end(); ++iterator)
            {
                const auto &NNO = iterator->second;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }

            for (auto iterator = neighbor_vector.keep_connected.begin();
                 iterator != neighbor_vector.keep_connected.end(); ++iterator)
            {
                const auto &NNO = *iterator;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }
        }

        for (auto iterator = repaired_vector.keep_connected.begin(); iterator != repaired_vector.keep_connected.end();
             ++iterator)
        {
            const auto &neighbor_offset = *iterator;
            const auto &neighbor_vector = index.vectors[neighbor_offset];

            for (auto iterator = neighbor_vector.short_edge_in.begin(); iterator != neighbor_vector.short_edge_in.end();
                 ++iterator)
            {
                const auto &NNO = iterator->first;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }

            for (auto iterator = neighbor_vector.short_edge_out.begin();
                 iterator != neighbor_vector.short_edge_out.end(); ++iterator)
            {
                const auto &NNO = iterator->second;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }

            for (auto iterator = neighbor_vector.keep_connected.begin();
                 iterator != neighbor_vector.keep_connected.end(); ++iterator)
            {
                const auto &NNO = *iterator;

                Get_Pool_From_SE(index, NNO, visited, pool);
                Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
            }
        }
    }

    // 删除索引中的向量
    inline void Erase(Index &index, const ID removed_id)
    {
        auto removed_offset = Get_Offset(index, removed_id);

        index.id_to_offset.erase(removed_id);

        auto &removed_vector = index.vectors[removed_offset];

        // 删除短边的出边
        for (auto iterator = removed_vector.short_edge_out.begin(); iterator != removed_vector.short_edge_out.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->second;
            auto &neighbor_vector = index.vectors[neighbor_offset];

            neighbor_vector.short_edge_in.erase(removed_offset);
        }

        // 删除短边的入边
        for (auto iterator = removed_vector.short_edge_in.begin(); iterator != removed_vector.short_edge_in.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->first;
            const auto &distance = iterator->second;
            auto &vector = index.vectors[neighbor_offset];
            auto temporary_iterator = vector.short_edge_out.find(distance);

            while (temporary_iterator->second != removed_offset)
            {
                ++temporary_iterator;
            }

            vector.short_edge_out.erase(temporary_iterator);
        }

        for (auto iterator = removed_vector.keep_connected.begin(); iterator != removed_vector.keep_connected.end();
             ++iterator)
        {
            const auto &neighbor_offset = *iterator;
            auto &vector = index.vectors[neighbor_offset];

            vector.keep_connected.erase(removed_offset);
        }

        // 删除长边
        for (auto iterator = removed_vector.long_edge_out.begin(); iterator != removed_vector.long_edge_out.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->first;
            auto &neighbor_vector = index.vectors[neighbor_offset];

            neighbor_vector.long_edge_in.erase(removed_offset);
        }

        for (auto iterator = removed_vector.long_edge_in.begin(); iterator != removed_vector.long_edge_in.end();
             ++iterator)
        {
            const auto &neighbor_offset = iterator->first;
            auto &neighbor_vector = index.vectors[neighbor_offset];

            neighbor_vector.long_edge_out.erase(removed_offset);
        }

        // 补边
        for (auto iterator = removed_vector.short_edge_in.begin(); iterator != removed_vector.short_edge_in.end();
             ++iterator)
        {
            const auto &repaired_offset = iterator->first;
            auto &repaired_vector = index.vectors[repaired_offset];

            if (repaired_vector.short_edge_out.size() < index.parameters.short_edge_lower_limit)
            {
                auto visited = std::vector<bool>(index.vectors.size(), false);

                visited[repaired_offset] = true;

                auto nearest_neighbors = std::priority_queue<std::pair<float, Offset>>();
                auto waiting_vectors = std::priority_queue<std::pair<float, Offset>,
                                                           std::vector<std::pair<float, Offset>>, std::greater<>>();
                auto pool = std::vector<Offset>();

                Mark_Erase(repaired_vector, visited);
                Similarity_Erase(index, repaired_vector, visited, pool, nearest_neighbors, waiting_vectors);

                while (!waiting_vectors.empty())
                {
                    const auto processing_offset = waiting_vectors.top().second;

                    waiting_vectors.pop();
                    Get_Pool_From_SE(index, processing_offset, visited, pool);
                    Similarity(index, repaired_vector.data, pool, nearest_neighbors, waiting_vectors);
                }

                while (nearest_neighbors.size() != 1)
                {
                    nearest_neighbors.pop();
                }

                const auto &TD = nearest_neighbors.top().first;
                const auto &TO = nearest_neighbors.top().second;
                auto &TV = index.vectors[TO];

                repaired_vector.short_edge_out.insert(nearest_neighbors.top());
                TV.short_edge_in.insert({repaired_offset, TD});

                if (repaired_vector.long_edge_out.contains(TO))
                {
                    repaired_vector.long_edge_out.erase(TO);
                    TV.long_edge_in.erase(repaired_offset);
                    Transfer_LEO(index, TO, repaired_offset);
                }
                else if (TV.long_edge_out.contains(repaired_offset))
                {
                    TV.long_edge_out.erase(repaired_offset);
                    repaired_vector.long_edge_in.erase(TO);
                    Transfer_LEO(index, repaired_offset, TO);
                }
            }
        }

        if (!removed_vector.long_edge_out.empty())
        {
            Transfer_LEO(index, removed_offset, removed_vector.long_edge_in.begin()->first);
        }

        Delete_Vector(index, removed_offset);
    }

    inline bool Similarity_Search(const Index &index, const float *const target_vector, const uint64_t capacity,
                                  std::vector<Offset> &pool,
                                  std::priority_queue<std::pair<float, ID>> &nearest_neighbors,
                                  std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                      std::greater<>> &waiting_vectors)
    {
        if (!pool.empty())
        {
            auto farest_offset = std::numeric_limits<uint64_t>::max();

            if (nearest_neighbors.size() == capacity)
            {
                farest_offset = nearest_neighbors.top().second;
            }

            const auto number = pool.size() - 1;

            Prefetch(index.vectors[pool.front()].data);

            for (auto i = 0; i < number; ++i)
            {
                const auto &neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto &neighbor_id = neighbor_vector.id;
                const auto &next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];
                const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

                waiting_vectors.push({distance, neighbor_offset});

                if (nearest_neighbors.size() < capacity)
                {
                    nearest_neighbors.push({distance, neighbor_id});
                }
                else if (distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.pop();
                    nearest_neighbors.push({distance, neighbor_id});
                }

                Prefetch(next_vector.data);
            }

            const auto &neighbor_offset = pool.back();
            const auto &neighbor_vector = index.vectors[neighbor_offset];
            const auto &neighbor_id = neighbor_vector.id;
            const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            waiting_vectors.push({distance, neighbor_offset});

            if (nearest_neighbors.size() < capacity)
            {
                nearest_neighbors.push({distance, neighbor_id});
            }
            else if (distance < nearest_neighbors.top().first)
            {
                nearest_neighbors.pop();
                nearest_neighbors.push({distance, neighbor_id});
            }

            pool.clear();

            if (farest_offset == std::numeric_limits<uint64_t>::max() ||
                farest_offset != nearest_neighbors.top().second)
            {
                return false;
            }

            return true;
        }

        return true;
    }

    // 查询距离目标向量最近的top-k个向量
    inline std::priority_queue<std::pair<float, ID>> Search(const Index &index, const float *const target_vector,
                                                            const uint64_t top_k, const uint64_t magnification)
    {
        const auto capacity = top_k + magnification;

        // 优先队列
        auto nearest_neighbors = std::priority_queue<std::pair<float, ID>>();

        // 标记是否被遍历过
        auto visited = std::vector<bool>(index.vectors.size(), false);

        visited[0] = true;

        // 排队队列
        auto waiting_vectors =
            std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();

        waiting_vectors.push({0, 0});

        // 计算池子
        auto pool = std::vector<Offset>();

        while (true)
        {
            auto processing_offset = waiting_vectors.top().second;

            waiting_vectors.pop();
            Get_Pool_From_SE(index, processing_offset, visited, pool);
            Similarity_Search(index, target_vector, capacity, pool, nearest_neighbors, waiting_vectors);

            const auto short_offset = waiting_vectors.top().second;

            Get_Pool_From_LEO(index, processing_offset, visited, pool);
            Similarity_Search(index, target_vector, capacity, pool, nearest_neighbors, waiting_vectors);

            const auto &nearest_offset = waiting_vectors.top().second;

            if (nearest_offset == short_offset)
            {
                break;
            }
        }

        // 阶段二
        // 查找与目标向量相似度最高（距离最近）的top-k个向量
        while (!waiting_vectors.empty())
        {
            auto processing_offset = waiting_vectors.top().second;

            waiting_vectors.pop();
            Get_Pool_From_SE(index, processing_offset, visited, pool);

            auto end = Similarity_Search(index, target_vector, capacity, pool, nearest_neighbors, waiting_vectors);

            if (end)
            {
                break;
            }
        }

        return nearest_neighbors;
    }

    // 查询
    // inline std::priority_queue<std::pair<float, uint64_t>> search(const Index &index, const float *const
    // query_vector,
    //                                                               const uint64_t top_k,
    //                                                               const uint64_t magnification = 0)
    // {
    //     // if (index.count == 0)
    //     // {
    //     //     throw std::logic_error("Index is empty. ");
    //     // }
    //     // if (query_vector.size() != index.parameters.dimension)
    //     // {
    //     //     throw std::invalid_argument("The dimension of target vector is not "
    //     //                                 "equality with vectors in index. ");
    //     // }
    //     return nearest_neighbors(index, query_vector, top_k, magnification);
    // }

    // Breadth First Search through Short Edges.
    inline void BFS_Through_SE(const Index &index, const Vector &start, std::vector<bool> &VC)
    {
        auto visited = std::unordered_set<Offset>();

        visited.insert(start.offset);

        auto last = std::vector<Offset>();

        last.push_back(start.offset);

        auto next = std::vector<Offset>();

        for (auto i = 1; i < index.parameters.cover_range; ++i)
        {
            for (auto j = 0; j < last.size(); ++j)
            {
                const auto &offset = last[j];
                const auto &vector = index.vectors[offset];

                for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
                {
                    const auto &neighbor_offset = iterator->first;

                    if (!visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
                {
                    const auto &neighbor_offset = iterator->second;

                    if (!visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
                {
                    const auto &neighbor_offset = *iterator;

                    if (!visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }
            }

            std::swap(last, next);
            next.clear();
        }

        for (auto i = 0; i < last.size(); ++i)
        {
            const auto &offset = last[i];
            const auto &vector = index.vectors[offset];

            for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
            {
                const auto &neighbor_offset = iterator->first;

                VC[neighbor_offset] = true;
            }

            for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
            {
                const auto &neighbor_offset = iterator->second;

                VC[neighbor_offset] = true;
            }

            for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
            {
                const auto &neighbor_offset = *iterator;

                VC[neighbor_offset] = true;
            }
        }
    }

    // 通过长边进行广度优先遍历
    //
    //  Breadth First Search Through Long Edges Out.
    inline void BFS_Through_LEO(const Index &index, std::unordered_set<Offset> &VR, std::vector<bool> &VC)
    {
        auto last = std::vector<Offset>();

        last.push_back(0);

        auto next = std::vector<Offset>();

        while (!last.empty())
        {
            for (auto i = 0; i < last.size(); ++i)
            {
                const auto &offset = last[i];
                const auto &vector = index.vectors[offset];

                for (auto iterator = vector.long_edge_out.begin(); iterator != vector.long_edge_out.end(); ++iterator)
                {
                    const auto &neighbor_offset = iterator->first;

                    if (!VC[neighbor_offset])
                    {
                        VR.insert(neighbor_offset);
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }
            }

            std::swap(last, next);
            next.clear();
        }
    }

    // 计算覆盖率
    inline float Calculate_Coverage(const Index &index)
    {
        auto VC = std::vector<bool>(index.vectors.size(), false);
        auto VR = std::unordered_set<Offset>();

        VR.insert(0);
        VC[0] = true;
        BFS_Through_LEO(index, VR, VC);

        for (auto i = VR.begin(); i != VR.end(); ++i)
        {
            BFS_Through_SE(index, index.vectors[*i], VC);
        }

        uint64_t number = 0;

        for (auto i : VC)
        {
            if (i)
            {
                ++number;
            }
        }

        return float(number - 1) / (index.count - 1);
    }

    inline bool Calculate_Benefits(const Index &index, const std::unordered_set<Offset> &missed, const Offset start,
                                   uint64_t &benefits)
    {

        auto visited = std::unordered_set<Offset>();

        visited.insert(start);

        auto last = std::vector<Offset>();

        last.push_back(start);

        auto next = std::vector<Offset>();
        uint64_t not_missed = 0;

        for (auto i = 1; i < index.parameters.cover_range; ++i)
        {
            for (auto j = 0; j < last.size(); ++j)
            {
                const auto &offset = last[j];
                const auto &vector = index.vectors[offset];

                for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
                {
                    const auto &neighbor_offset = iterator->first;

                    if (!visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        next.push_back(neighbor_offset);

                        if (missed.contains(neighbor_offset))
                        {
                            ++benefits;
                        }
                        else
                        {
                            ++not_missed;
                        }
                    }
                }

                for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
                {
                    const auto &neighbor_offset = iterator->second;

                    if (!visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        next.push_back(neighbor_offset);

                        if (missed.contains(neighbor_offset))
                        {
                            ++benefits;
                        }
                        else
                        {
                            ++not_missed;
                        }
                    }
                }

                for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
                {
                    const auto &neighbor_offset = *iterator;

                    if (!visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        next.push_back(neighbor_offset);

                        if (missed.contains(neighbor_offset))
                        {
                            ++benefits;
                        }
                        else
                        {
                            ++not_missed;
                        }
                    }
                }
            }

            std::swap(last, next);
            next.clear();
        }

        for (auto i = 0; i < last.size(); ++i)
        {
            const auto &offset = last[i];
            const auto &vector = index.vectors[offset];

            for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
            {
                const auto &neighbor_offset = iterator->first;

                if (!visited.contains(neighbor_offset))
                {
                    visited.insert(neighbor_offset);

                    if (missed.contains(neighbor_offset))
                    {
                        ++benefits;
                    }
                    else
                    {
                        ++not_missed;
                    }
                }
            }

            for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
            {
                const auto &neighbor_offset = iterator->second;

                if (!visited.contains(neighbor_offset))
                {
                    visited.insert(neighbor_offset);

                    if (missed.contains(neighbor_offset))
                    {
                        ++benefits;
                    }
                    else
                    {
                        ++not_missed;
                    }
                }
            }

            for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
            {
                const auto &neighbor_offset = *iterator;

                if (!visited.contains(neighbor_offset))
                {
                    visited.insert(neighbor_offset);

                    if (missed.contains(neighbor_offset))
                    {
                        ++benefits;
                    }
                    else
                    {
                        ++not_missed;
                    }
                }
            }
        }

        if (not_missed * 10 < benefits)
        {
            return true;
        }

        return false;
    }

    // 计算为哪个顶点补长边可以覆盖的顶点最多
    inline void Max_Benefits(const Index &index, const std::unordered_set<Offset> &missed, uint64_t &max_benefits,
                             Offset &max_benefit_offset)
    {
        for (auto iterator = missed.begin(); iterator != missed.end(); ++iterator)
        {
            auto offset = *iterator;
            uint64_t benefits = 1;
            bool end = Calculate_Benefits(index, missed, offset, benefits);

            if (max_benefits < benefits)
            {
                max_benefits = benefits;
                max_benefit_offset = offset;

                if (end)
                {
                    break;
                }
            }
        }
    }

    inline void Similarity_Optimize(const Index &index, const float *const target_vector, std::vector<Offset> &pool,
                                    std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                        std::greater<>> &waiting_vectors)
    {
        if (!pool.empty())
        {
            const auto number = pool.size() - 1;

            Prefetch(index.vectors[pool.front()].data);

            for (auto i = 0; i < number; ++i)
            {
                const auto &neighbor_offset = pool[i];
                const auto &neighbor_vector = index.vectors[neighbor_offset];
                const auto &next_offset = pool[i + 1];
                const auto &next_vector = index.vectors[next_offset];
                const auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

                Prefetch(next_vector.data);
                waiting_vectors.push({distance, neighbor_offset});
            }

            auto &neighbor_offset = pool.back();
            auto &neighbor_vector = index.vectors[neighbor_offset];
            auto distance = index.similarity(target_vector, neighbor_vector.data, index.parameters.dimension);

            waiting_vectors.push({distance, neighbor_offset});
            pool.clear();
        }
    }

    inline void Search_Optimize(const Index &index, const Offset offset,
                                std::vector<std::pair<float, Offset>> &long_path)
    {
        const auto &vector = index.vectors[offset];

        // 等待队列
        auto waiting_vectors =
            std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();

        waiting_vectors.push({vector.zero, 0});
        long_path.push_back({vector.zero, 0});

        // 标记是否被遍历过
        auto visited = std::vector<bool>(index.vectors.size(), false);

        visited[0] = true;

        // 计算池子
        auto pool = std::vector<Offset>();

        Get_Pool_From_SE(index, 0, visited, pool);
        Similarity_Optimize(index, vector.data, pool, waiting_vectors);

        auto short_offset = waiting_vectors.top().second;

        Get_Pool_From_LEO(index, 0, visited, pool);
        Similarity_Optimize(index, vector.data, pool, waiting_vectors);

        auto &nearest_offset = waiting_vectors.top().second;

        if (short_offset != nearest_offset)
        {
            while (true)
            {
                long_path.push_back(waiting_vectors.top());

                const auto &processing_offset = waiting_vectors.top().second;

                Get_Pool_From_SE(index, processing_offset, visited, pool);
                Similarity_Optimize(index, vector.data, pool, waiting_vectors);

                short_offset = waiting_vectors.top().second;

                Get_Pool_From_LEO(index, processing_offset, visited, pool);
                Similarity_Optimize(index, vector.data, pool, waiting_vectors);

                const auto &nearest_offset = waiting_vectors.top().second;

                if (short_offset == nearest_offset)
                {
                    break;
                }
            }
        }
    }

    inline void Add_Long_Edges_Optimize(Index &index, std::vector<std::pair<float, Offset>> &long_path,
                                        const Offset offset)
    {
        auto &vector = index.vectors[offset];
        float added_distance = 0;
        float maximum_cosine = -2;
        Offset added_offset = 0;

        for (int i = long_path.size() - 1; 0 < i; --i)
        {
            const auto &D = long_path[i].first;
            const auto &neighbor_offset = long_path[i].second;
            auto &neighbor_vector = index.vectors[neighbor_offset];

            if (neighbor_vector.zero < vector.zero && !Adjacent(index, offset, neighbor_offset))
            {
                const auto CV = Cosine_Value(added_distance, vector.zero, neighbor_vector.zero);

                if (maximum_cosine < CV)
                {
                    added_distance = D;
                    maximum_cosine = CV;
                    added_offset = neighbor_offset;
                }
            }
        }

        if (1.732 < maximum_cosine)
        {
            auto &neighbor_vector = index.vectors[added_offset];

            neighbor_vector.long_edge_out.insert({offset, added_distance});
            vector.long_edge_in.insert({added_offset, added_distance});
        }
        else
        {
            auto &neighbor_vector = index.vectors.front();

            neighbor_vector.long_edge_out.insert({offset, added_distance});
            vector.long_edge_in.insert({added_offset, added_distance});
        }
    }

    // 优化索引结构
    inline void Optimize(Index &index)
    {
        auto VC = std::vector<bool>(index.vectors.size(), false);
        auto VR = std::unordered_set<Offset>();

        VR.insert(0);
        VC[0] = true;
        BFS_Through_LEO(index, VR, VC);

        for (auto i = VR.begin(); i != VR.end(); ++i)
        {
            BFS_Through_SE(index, index.vectors[*i], VC);
        }

        auto missed = std::unordered_set<Offset>();

        for (auto offset = 0; offset < VC.size(); ++offset)
        {
            if (!VC[offset] && index.vectors[offset].data != nullptr)
            {
                missed.insert(offset);
            }
        }

        std::cout << "The number of vertices that can be reached through long edges: " << VR.size() << std::endl;
        std::cout << "Number of vertices covered: " << index.count - 1 - missed.size() << std::endl;
        std::cout << "Coverage rate: " << (float)(index.count - 1 - missed.size()) / index.count << std::endl;
        std::cout << "The number of vertices not covered: " << missed.size() << std::endl;

        VR.clear();
        VC.clear();

        Offset offset = 0;
        uint64_t benefits = 0;

        Max_Benefits(index, missed, benefits, offset);

        const auto &id = index.vectors[offset].id;
        auto long_path = std::vector<std::pair<float, Offset>>();

        std::cout << std::format("Vertices with added long edges(id, offset): ({0}, {1})", id, offset) << std::endl;
        std::cout << "Add a long edge to this vertex to cover an additional number of vertices: " << benefits
                  << std::endl;

        Search_Optimize(index, offset, long_path);
        Add_Long_Edges_Optimize(index, long_path, offset);
    }

} // namespace HSG
