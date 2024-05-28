#pragma once

#include <map>
#include <queue>
#include <random>
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
    typedef uint64_t Offset;

    // 向量外部的唯一标识符
    typedef uint64_t ID;

    // 向量
    class Vector
    {
      public:
        // 向量的外部id
        ID id;
        //
        Offset offset;
        // 向量的数据
        const float *data;
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

        explicit Vector(const ID id, Offset offset, const float *const data_address)
            : id(id), offset(offset), data(data_address)
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
        // 插入向量时提前终止条件
        //
        // termination_number = short_edge_lower_limit + magnification
        uint64_t termination_number;
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
              termination_number(short_edge_lower_limit + magnification),
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
            this->vectors.push_back(Vector(std::numeric_limits<uint64_t>::max(), 0, this->zero.data()));
            this->id_to_offset.insert({std::numeric_limits<uint64_t>::max(), 0});
        }
    };

    inline Offset Get_Offset(const Index &index, const ID id)
    {
        return index.id_to_offset.find(id)->second;
    }

    inline bool Reachable(const Vector &vector)
    {
        if (vector.offset == 0)
        {
            return true;
        }
        else if (vector.long_edge_in.empty())
        {
            return false;
        }

        return true;
    }

    inline void Delete_Vector(Vector &vector)
    {
        vector.data = nullptr;
        vector.keep_connected.clear();
        vector.long_edge_in.clear();
        vector.long_edge_out.clear();
        vector.short_edge_in.clear();
        vector.short_edge_out.clear();
    }

    inline void Visit_LEO_First_Time(
        const Index &index, const Vector &processing_vector, const float *target_vector, std::vector<bool> &visited,
        std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>
            &waiting_vectors)
    {
        for (auto iterator = processing_vector.long_edge_out.begin(); iterator != processing_vector.long_edge_out.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;
            visited[neighbor_offset] = true;
            waiting_vectors.push(
                {index.similarity(target_vector, index.vectors[neighbor_offset].data, index.parameters.dimension),
                 neighbor_offset});
        }
    }

    inline void Visit_LEO(const Index &index, const Vector &processing_vector, const float *target_vector,
                          std::vector<bool> &visited,
                          std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                              std::greater<>> &waiting_vectors)
    {
        for (auto iterator = processing_vector.long_edge_out.begin(); iterator != processing_vector.long_edge_out.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                waiting_vectors.push(
                    {index.similarity(target_vector, index.vectors[neighbor_offset].data, index.parameters.dimension),
                     neighbor_offset});
            }
        }
    }

    inline void Visit_SEO(const Index &index, const Vector &processing_vector, const float *target_vector,
                          std::vector<bool> &visited,
                          std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                              std::greater<>> &waiting_vectors)
    {
        for (auto iterator = processing_vector.short_edge_out.begin();
             iterator != processing_vector.short_edge_out.end(); ++iterator)
        {
            auto &neighbor_offset = iterator->second;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                waiting_vectors.push(
                    {index.similarity(target_vector, index.vectors[neighbor_offset].data, index.parameters.dimension),
                     neighbor_offset});
            }
        }
    }

    inline void Visit_SEI(const Index &index, const Vector &processing_vector, const float *target_vector,
                          std::vector<bool> &visited,
                          std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                              std::greater<>> &waiting_vectors)
    {
        for (auto iterator = processing_vector.short_edge_in.begin(); iterator != processing_vector.short_edge_in.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                waiting_vectors.push(
                    {index.similarity(target_vector, index.vectors[neighbor_offset].data, index.parameters.dimension),
                     neighbor_offset});
            }
        }
    }

    inline void Visit_KC(const Index &index, const Vector &processing_vector, const float *target_vector,
                         std::vector<bool> &visited,
                         std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                             std::greater<>> &waiting_vectors)
    {
        for (auto iterator = processing_vector.keep_connected.begin();
             iterator != processing_vector.keep_connected.end(); ++iterator)
        {
            auto &neighbor_offset = *iterator;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                waiting_vectors.push(
                    {index.similarity(target_vector, index.vectors[neighbor_offset].data, index.parameters.dimension),
                     neighbor_offset});
            }
        }
    }

    inline bool Adjacent(const Vector &v1, const Vector &v2)
    {
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

    inline void Get_Pool_From_LEO(const Vector &processing_vector, std::vector<bool> &visited,
                                  std::vector<Offset> &pool)
    {
        for (auto iterator = processing_vector.long_edge_out.begin(); iterator != processing_vector.long_edge_out.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;

            // 计算当前向量的出边指向的向量和目标向量的距离
            if (!visited[neighbor_offset])
            {
                visited[neighbor_offset] = true;
                pool.push_back(neighbor_offset);
            }
        }
    }

    inline void Get_Pool_From_SE(const Vector &processing_vector, std::vector<bool> &visited, std::vector<Offset> &pool)
    {
        for (auto iterator = processing_vector.short_edge_out.begin();
             iterator != processing_vector.short_edge_out.end(); ++iterator)
        {
            auto &neighbor_offset = iterator->second;

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
            auto &neighbor_offset = iterator->first;

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
            auto &neighbor_offset = *iterator;

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

    inline void Similarity(const Index &index, const float *const target_vector, std::vector<Offset> &pool,
                           std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                               std::greater<>> &waiting_vectors)
    {
        if (!pool.empty())
        {
            Prefetch(index.vectors[pool.front()].data);

            const auto number = pool.size() - 1;

            for (auto i = 0; i < number; ++i)
            {
                auto &neighbor_offset = pool[i];
                auto &next_offset = pool[i + 1];

                Prefetch(index.vectors[next_offset].data);
                waiting_vectors.push(
                    {index.similarity(target_vector, index.vectors[neighbor_offset].data, index.parameters.dimension),
                     neighbor_offset});
            }

            waiting_vectors.push(
                {index.similarity(target_vector, index.vectors[pool.back()].data, index.parameters.dimension),
                 pool.back()});

            pool.clear();
        }
    }

    // 查询距离目标向量最近的k个向量
    //
    // k = index.parameters.short_edge_lower_limit
    //
    // 返回最近邻和不属于最近邻但是在路径上的顶点
    inline void Search_Add(const Index &index, const float *const target_vector,
                           std::vector<std::pair<float, Offset>> &long_path,
                           std::vector<std::pair<float, Offset>> &short_path,
                           std::priority_queue<std::pair<float, Offset>> &nearest_neighbors)
    {
        // 等待队列
        auto waiting_vectors =
            std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();
        waiting_vectors.push({Space::Euclidean2::zero(target_vector, index.parameters.dimension), 0});

        // 标记是否被遍历过
        auto visited = std::vector<bool>(index.vectors.size(), false);
        visited[0] = true;

        // 计算池子
        auto pool = std::vector<Offset>();

        // 阶段一：
        // 利用长边快速找到定位到处于目标向量附近区域的向量
        while (true)
        {
            auto &processing_offset = waiting_vectors.top().second;
            auto &processing_vector = index.vectors[processing_offset];
            long_path.push_back(waiting_vectors.top());

            Get_Pool_From_LEO(processing_vector, visited, pool);
            Similarity(index, target_vector, pool, waiting_vectors);

            if (processing_offset == waiting_vectors.top().second)
            {
                break;
            }
        }

        // 阶段二：
        // 利用短边找到和目标向量最近的向量
        while (true)
        {
            auto &processing_offset = waiting_vectors.top().second;
            auto &processing_vector = index.vectors[processing_offset];

            Get_Pool_From_SE(processing_vector, visited, pool);
            Similarity(index, target_vector, pool, waiting_vectors);

            if (processing_offset == waiting_vectors.top().second)
            {
                break;
            }

            short_path.push_back(waiting_vectors.top());
        }

        // 阶段三：
        // 查找与目标向量相似度最高（距离最近）的k个向量
        while (!waiting_vectors.empty())
        {
            auto &processing_distance = waiting_vectors.top().first;
            auto &processing_offset = waiting_vectors.top().second;
            auto &processing_vector = index.vectors[processing_offset];
            waiting_vectors.pop();

            // 如果优先队列中的向量的数量小于k
            if (nearest_neighbors.size() < index.parameters.termination_number)
            {
                nearest_neighbors.push({processing_distance, processing_offset});
            }
            else
            {
                // 如果当前的向量和查询向量的距离小于优先队列中的最大值
                if (processing_distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.pop();
                    nearest_neighbors.push({processing_distance, processing_offset});
                }
                else
                {
                    break;
                }
            }

            Get_Pool_From_SE(processing_vector, visited, pool);
            Similarity(index, target_vector, pool, waiting_vectors);
        }

        while (index.parameters.short_edge_lower_limit < nearest_neighbors.size())
        {
            nearest_neighbors.pop();
        }

        auto &farest_neighbor_distance = nearest_neighbors.top().first;

        // 去重
        // long_path 和 short_path 中可能有 nearest_neighbors 中的顶点，将它们删除
        while (!short_path.empty() && short_path.back().first <= farest_neighbor_distance)
        {
            short_path.pop_back();
        }

        while (!long_path.empty() && long_path.back().first <= farest_neighbor_distance)
        {
            long_path.pop_back();
        }
    }

    inline bool Connected(const Index &index, const Vector &vector, Offset offset)
    {
        auto visited = std::unordered_set<Offset>();
        visited.insert(vector.offset);
        auto last = std::vector<Offset>();
        last.push_back(vector.offset);
        auto next = std::vector<Offset>();

        for (auto round = 0; round < 4; ++round)
        {
            for (auto iterator = last.begin(); iterator != last.end(); ++iterator)
            {
                auto &t = index.vectors[*iterator];

                for (auto iterator = t.short_edge_in.begin(); iterator != t.short_edge_in.end(); ++iterator)
                {
                    auto &t1 = iterator->first;

                    if (!visited.contains(t1))
                    {
                        visited.insert(t1);
                        next.push_back(t1);
                    }
                }

                for (auto iterator = t.short_edge_out.begin(); iterator != t.short_edge_out.end(); ++iterator)
                {
                    auto &t1 = iterator->second;

                    if (!visited.contains(t1))
                    {
                        visited.insert(t1);
                        next.push_back(t1);
                    }
                }

                for (auto iterator = t.keep_connected.begin(); iterator != t.keep_connected.end(); ++iterator)
                {
                    auto &t1 = *iterator;

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

        return false;
    }

    // 添加长边
    //
    // 简单分为三种情况
    //
    // 只实现了第一种情况，后两种情况没有单独处理
    inline void Add_Long_Edges(Index &index, std::vector<std::pair<float, Offset>> &long_path,
                               std::vector<std::pair<float, Offset>> &short_path, Vector &vector)
    {
        auto &offset = vector.offset;

        if (short_path.size() == index.parameters.cover_range)
        {
            auto &neighbor_distance = long_path.back().first;
            auto &neighbor_offset = long_path.back().second;
            auto &neighbor = index.vectors[neighbor_offset];

            neighbor.long_edge_out.insert({offset, neighbor_distance});
            vector.long_edge_in.insert({neighbor_offset, neighbor_distance});
        }
        else if (index.parameters.cover_range < short_path.size())
        {
            auto last_offset = long_path.back().second;

            for (auto i = index.parameters.cover_range; i < short_path.size(); i += index.parameters.cover_range + 1)
            {
                auto &last = index.vectors[last_offset];
                auto &next_offset = short_path[i].second;
                auto &next = index.vectors[next_offset];
                auto distance = index.similarity(last.data, next.data, index.parameters.dimension);

                last.long_edge_out.insert({next_offset, distance});
                next.long_edge_in.insert({last_offset, distance});

                last_offset = next_offset;
            }
        }
    }

    // 添加
    inline void Add(Index &index, const ID id, const float *const added_vector_data)
    {
        Offset offset = index.vectors.size();
        ++index.count;

        if (index.empty.empty())
        {
            // 在索引中创建一个新向量
            index.vectors.push_back(Vector(id, offset, added_vector_data));
        }
        else
        {
            offset = index.empty.top();
            index.empty.pop();
            index.vectors[offset].id = id;
            index.vectors[offset].data = added_vector_data;
        }

        index.id_to_offset.insert({id, offset});
        auto &new_vector = index.vectors[offset];

        auto nearest_neighbors = std::priority_queue<std::pair<float, Offset>>();
        auto long_path = std::vector<std::pair<float, Offset>>();
        auto short_path = std::vector<std::pair<float, Offset>>();

        // 搜索距离新增向量最近的 index.parameters.short_edge_lower_limit 个向量
        // 同时记录搜索路径
        Search_Add(index, added_vector_data, long_path, short_path, nearest_neighbors);

        // 添加短边
        while (!nearest_neighbors.empty())
        {
            auto distance = nearest_neighbors.top().first;
            auto &neighbor = index.vectors[nearest_neighbors.top().second];
            nearest_neighbors.pop();

            // 为新向量添加出边
            new_vector.short_edge_out.insert({distance, neighbor.offset});

            // 为邻居向量添加入边
            neighbor.short_edge_in.insert({offset, distance});

            // 如果邻居向量的出边小于短边下限
            if (neighbor.short_edge_out.size() < index.parameters.short_edge_lower_limit)
            {
                // 邻居向量添加出边
                neighbor.short_edge_out.insert({distance, offset});

                // 新向量添加入边
                new_vector.short_edge_in.insert({neighbor.offset, distance});
            }
            // 如果新向量和邻居的距离小于邻居当前距离最大的出边的距离
            else if (distance < neighbor.short_edge_out.rbegin()->first)
            {
                auto farest_distance = neighbor.short_edge_out.rbegin()->first;
                auto &neighbor_neighbor = index.vectors[neighbor.short_edge_out.rbegin()->second];
                // neighbor neighbor offset
                auto &NN_offset = neighbor_neighbor.offset;

                // 邻居向量删除距离最大的出边
                neighbor.short_edge_out.erase(std::prev(neighbor.short_edge_out.end()));
                neighbor_neighbor.short_edge_in.erase(neighbor.offset);

                if (!neighbor.short_edge_in.contains(NN_offset))
                {
                    // 添加一条长边
                    if (Connected(index, neighbor, NN_offset))
                    {
                        // neighbor neighbor reachable
                        auto NNR = Reachable(neighbor_neighbor);

                        // neighbor reachable
                        auto NR = Reachable(neighbor);

                        if (NR && !NNR)
                        {
                            neighbor.long_edge_out.insert({NN_offset, farest_distance});
                            neighbor_neighbor.long_edge_in.insert({neighbor.offset, farest_distance});
                        }
                        else if (NNR && !NR)
                        {
                            neighbor_neighbor.long_edge_out.insert({neighbor.offset, farest_distance});
                            neighbor.long_edge_in.insert({NN_offset, farest_distance});
                        }
                    }
                    else
                    {
                        if (neighbor.short_edge_out.size() < index.parameters.short_edge_upper_limit)
                        {
                            neighbor.short_edge_out.insert({farest_distance, NN_offset});
                            neighbor_neighbor.short_edge_in.insert({neighbor.offset, farest_distance});
                        }
                        else
                        {
                            neighbor.keep_connected.insert(NN_offset);
                            neighbor_neighbor.keep_connected.insert(neighbor.offset);
                        }
                    }
                }

                // 邻居向量添加出边
                neighbor.short_edge_out.insert({distance, offset});
                // 新向量添加入边
                new_vector.short_edge_in.insert({neighbor.offset, distance});
            }
        }

        Add_Long_Edges(index, long_path, short_path, new_vector);
    }

    // 基于权重的长边修补方法
    inline void Repair_LE_Weight(Index &index, Vector &removed_vector)
    {
        auto &removed_vector_offset = removed_vector.offset;

        // 删除出边
        for (auto iterator = removed_vector.long_edge_out.begin(); iterator != removed_vector.long_edge_out.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;
            index.vectors[neighbor_offset].long_edge_in.erase(removed_vector_offset);
        }

        // 计算分母
        float sum = 0;

        for (auto iterator = removed_vector.long_edge_in.begin(); iterator != removed_vector.long_edge_in.end();
             ++iterator)
        {
            auto &distance = iterator->second;
            sum += distance;
        }

        // 删除入边并补边
        // 真随机数生成器
        std::random_device random_device_generator;

        // 伪随机数生成器
        // 用真随机数生成器初始化
        std::mt19937 mt19937_generator(random_device_generator());

        // 均匀分布
        std::uniform_real_distribution<float> distribution(0, sum);

        for (auto iterator = removed_vector.long_edge_in.begin(); iterator != removed_vector.long_edge_in.end();
             ++iterator)
        {
            auto &repaired_vector_offset = iterator->first;
            auto &in_edge_distance = iterator->second;
            auto &repaired_vector = index.vectors[repaired_vector_offset];

            // 补边
            for (auto iterator = removed_vector.long_edge_out.begin(); iterator != removed_vector.long_edge_out.end();
                 ++iterator)
            {
                auto &new_long_edge_offset = iterator->first;

                if (distribution(mt19937_generator) < in_edge_distance)
                {
                    auto &new_long_edge = index.vectors[new_long_edge_offset];
                    auto distance =
                        index.similarity(repaired_vector.data, new_long_edge.data, index.parameters.dimension);
                    repaired_vector.long_edge_out.insert({new_long_edge_offset, distance});
                    new_long_edge.long_edge_in.insert({repaired_vector_offset, distance});
                }
            }
        }
    }

    inline void Transfer_LEI(Index &index, Vector &from, Vector &to)
    {
        for (auto iterator = from.long_edge_in.begin(); iterator != from.long_edge_in.end(); ++iterator)
        {
            auto &neighbor_offset = iterator->first;
            auto &neighbor_vector = index.vectors[neighbor_offset];
            auto distance = index.similarity(neighbor_vector.data, to.data, index.parameters.dimension);

            neighbor_vector.long_edge_out.erase(from.offset);
            neighbor_vector.long_edge_out.insert({to.offset, distance});
            to.long_edge_in.insert({neighbor_offset, distance});
        }
    }

    inline void Transfer_LEO(Index &index, Vector &from, Vector &to)
    {
        for (auto iterator = from.long_edge_out.begin(); iterator != from.long_edge_out.end(); ++iterator)
        {
            auto &neighbor_offset = iterator->first;
            auto &neighbor_vector = index.vectors[neighbor_offset];
            auto distance = index.similarity(neighbor_vector.data, to.data, index.parameters.dimension);

            neighbor_vector.long_edge_in.erase(from.offset);
            neighbor_vector.long_edge_in.insert({to.offset, distance});
            to.long_edge_out.insert({neighbor_offset, distance});
        }
    }

    inline void Transfer_LE(Index &index, Vector &from, Vector &to)
    {
        Transfer_LEI(index, from, to);
        Transfer_LEO(index, from, to);
        // from.long_edge_in.clear();
        // from.long_edge_out.clear();
    }

    // 通过转移长边到距离被删除向量最近的向量修补长边
    inline void Repair_LE_Transfer(Index &index, Vector &removed_vector)
    {
        auto &RV = removed_vector;
        auto &substitute_offset = removed_vector.short_edge_out.begin()->second;
        auto &substitute_vector = index.vectors[substitute_offset];

        for (auto iterator = RV.long_edge_out.begin(); iterator != RV.long_edge_out.end(); ++iterator)
        {
            auto &repaired_offset = iterator->first;
            auto &repaired_vector = index.vectors[repaired_offset];

            if (Adjacent(repaired_vector, substitute_vector))
            {
                Transfer_LE(index, repaired_vector, substitute_vector);
            }
            else
            {
                auto distance =
                    index.similarity(substitute_vector.data, repaired_vector.data, index.parameters.dimension);
                substitute_vector.long_edge_out.insert({repaired_offset, distance});
                repaired_vector.long_edge_in.insert({substitute_offset, distance});
            }
        }

        for (auto iterator = RV.long_edge_in.begin(); iterator != RV.long_edge_in.end(); ++iterator)
        {
            auto &repaired_offset = iterator->first;
            auto &repaired_vector = index.vectors[repaired_offset];

            if (Adjacent(repaired_vector, substitute_vector))
            {
                Transfer_LE(index, repaired_vector, substitute_vector);
            }
            else
            {
                auto distance =
                    index.similarity(substitute_vector.data, repaired_vector.data, index.parameters.dimension);
                repaired_vector.long_edge_out.insert({substitute_offset, distance});
                substitute_vector.long_edge_in.insert({repaired_offset, distance});
            }
        }
    }

    // 删除索引中的向量
    inline void Erase(Index &index, const ID removed_vector_id)
    {
        auto removed_vector_offset = Get_Offset(index, removed_vector_id);
        auto &removed_vector = index.vectors[removed_vector_offset];

        // 删除短边的出边
        for (auto iterator = removed_vector.short_edge_out.begin(); iterator != removed_vector.short_edge_out.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->second;
            index.vectors[neighbor_offset].short_edge_in.erase(removed_vector_offset);
        }

        // 删除短边的入边
        for (auto iterator = removed_vector.short_edge_in.begin(); iterator != removed_vector.short_edge_in.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;
            auto &distance = iterator->second;
            auto &vector = index.vectors[neighbor_offset];
            auto temporary_iterator = vector.short_edge_out.find(distance);

            while (temporary_iterator->second != removed_vector_id)
            {
                ++temporary_iterator;
            }

            vector.short_edge_out.erase(temporary_iterator);
        }

        // 补边
        for (auto iterator = removed_vector.short_edge_in.begin(); iterator != removed_vector.short_edge_in.end();
             ++iterator)
        {
            auto &repaired_offset = iterator->first;

            // 需要补边的向量
            auto &repaired_vector = index.vectors[repaired_offset];

            if (repaired_vector.short_edge_out.size() < index.parameters.short_edge_lower_limit)
            {
                auto visited = std::vector<bool>(index.vectors.size(), false);
                visited[repaired_offset] = true;

                // 优先队列
                auto nearest_neighbors = std::priority_queue<std::pair<float, Offset>>();

                // 排队队列
                auto waiting_vectors = std::priority_queue<std::pair<float, Offset>,
                                                           std::vector<std::pair<float, Offset>>, std::greater<>>();

                auto pool = std::vector<Offset>();

                for (auto iterator = repaired_vector.short_edge_in.begin();
                     iterator != repaired_vector.short_edge_in.end(); ++iterator)
                {
                    auto &neighbor_offset = iterator->first;
                    auto &neighbor_vector = index.vectors[neighbor_offset];

                    Get_Pool_From_SE(neighbor_vector, visited, pool);
                    Similarity(index, repaired_vector.data, pool, waiting_vectors);
                }

                for (auto iterator = repaired_vector.short_edge_out.begin();
                     iterator != repaired_vector.short_edge_out.end(); ++iterator)
                {
                    auto &neighbor_offset = iterator->second;
                    auto &neighbor_vector = index.vectors[neighbor_offset];

                    Get_Pool_From_SE(neighbor_vector, visited, pool);
                    Similarity(index, repaired_vector.data, pool, waiting_vectors);
                }

                for (auto iterator = repaired_vector.keep_connected.begin();
                     iterator != repaired_vector.keep_connected.end(); ++iterator)
                {
                    auto &neighbor_offset = *iterator;
                    auto &neighbor_vector = index.vectors[neighbor_offset];

                    Get_Pool_From_SE(neighbor_vector, visited, pool);
                    Similarity(index, repaired_vector.data, pool, waiting_vectors);
                }

                while (!waiting_vectors.empty())
                {
                    auto processing_distance = waiting_vectors.top().first;
                    auto processing_vector_offset = waiting_vectors.top().second;
                    auto &processing_vector = index.vectors[processing_vector_offset];
                    waiting_vectors.pop();

                    // 如果已遍历的向量小于候选数量
                    if (nearest_neighbors.size() < index.parameters.magnification)
                    {
                        nearest_neighbors.push({processing_distance, processing_vector_offset});
                    }
                    else
                    {
                        // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
                        if (processing_distance < nearest_neighbors.top().first)
                        {
                            nearest_neighbors.pop();
                            nearest_neighbors.push({processing_distance, processing_vector_offset});
                        }
                        else
                        {
                            break;
                        }
                    }

                    Get_Pool_From_SE(processing_vector, visited, pool);
                    Similarity(index, repaired_vector.data, pool, waiting_vectors);
                }

                while (nearest_neighbors.size() != 1)
                {
                    nearest_neighbors.pop();
                }

                repaired_vector.short_edge_out.insert(nearest_neighbors.top());
                index.vectors[nearest_neighbors.top().second].short_edge_in.insert(
                    {repaired_offset, nearest_neighbors.top().first});
            }
        }

        // 删除长边
        for (auto iterator = removed_vector.long_edge_out.begin(); iterator != removed_vector.long_edge_out.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;
            auto &neighbor_vector = index.vectors[neighbor_offset];
            neighbor_vector.long_edge_in.erase(removed_vector_offset);
        }

        for (auto iterator = removed_vector.long_edge_in.begin(); iterator != removed_vector.long_edge_in.end();
             ++iterator)
        {
            auto &neighbor_offset = iterator->first;
            auto &neighbor_vector = index.vectors[neighbor_offset];
            neighbor_vector.long_edge_out.erase(removed_vector_offset);
        }

        // 如果长边不多
        // repair_LE_weight(index, removed_vector);
        // 否则
        Repair_LE_Transfer(index, removed_vector);

        Delete_Vector(removed_vector);
    }

    // 查询距离目标向量最近的top-k个向量
    inline std::priority_queue<std::pair<float, ID>> search(const Index &index, const float *const target_vector,
                                                            const uint64_t top_k, const uint64_t magnification)
    {
        // 优先队列
        auto nearest_neighbors = std::priority_queue<std::pair<float, ID>>();

        // 标记是否被遍历过
        auto visited = std::vector<bool>(index.vectors.size(), false);
        visited[0] = true;

        // 排队队列
        auto waiting_vectors =
            std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>, std::greater<>>();

        // 计算池子
        auto pool = std::vector<Offset>();

        Get_Pool_From_LEO(index.vectors[0], visited, pool);
        Similarity(index, target_vector, pool, waiting_vectors);

        // 阶段一
        // 利用长边快速找到定位到处于目标向量附件区域的向量
        while (true)
        {
            auto &processing_offset = waiting_vectors.top().second;
            auto &processing_vector = index.vectors[processing_offset];

            Get_Pool_From_LEO(processing_vector, visited, pool);
            Similarity(index, target_vector, pool, waiting_vectors);

            if (waiting_vectors.top().second == processing_offset)
            {
                break;
            }
        }

        // 阶段二
        // 查找与目标向量相似度最高（距离最近）的top-k个向量
        while (!waiting_vectors.empty())
        {
            auto processing_distance = waiting_vectors.top().first;
            auto processing_vector_offset = waiting_vectors.top().second;
            auto &processing_vector = index.vectors[processing_vector_offset];
            waiting_vectors.pop();

            // 如果已遍历的向量小于候选数量
            if (nearest_neighbors.size() < top_k + magnification)
            {
                nearest_neighbors.push({processing_distance, processing_vector.id});
            }
            else
            {
                // 如果当前的向量和查询向量的距离小于已优先队列中的最大值
                if (processing_distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.pop();
                    nearest_neighbors.push({processing_distance, processing_vector.id});
                }
                else
                {
                    break;
                }
            }

            Get_Pool_From_SE(processing_vector, visited, pool);
            Similarity(index, target_vector, pool, waiting_vectors);
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
        auto last = std::vector<Offset>();
        last.push_back(start.offset);
        auto next = std::vector<Offset>();

        for (auto i = 1; i < index.parameters.cover_range; ++i)
        {
            for (auto j = 0; j < last.size(); ++j)
            {
                auto &offset = last[j];
                auto &vector = index.vectors[offset];

                for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
                {
                    auto &neighbor_offset = iterator->first;

                    if (!VC[neighbor_offset])
                    {
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
                {
                    auto &neighbor_offset = iterator->second;

                    if (!VC[neighbor_offset])
                    {
                        VC[neighbor_offset] = true;
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
                {
                    auto &neighbor_offset = *iterator;

                    if (!VC[neighbor_offset])
                    {
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
            auto &offset = last[i];
            auto &vector = index.vectors[offset];

            for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
            {
                auto &neighbor_offset = iterator->first;

                if (!VC[neighbor_offset])
                {
                    VC[neighbor_offset] = true;
                }
            }

            for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
            {
                auto &neighbor_offset = iterator->second;

                if (!VC[neighbor_offset])
                {
                    VC[neighbor_offset] = true;
                }
            }

            for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
            {
                auto &neighbor_offset = *iterator;

                if (!VC[neighbor_offset])
                {
                    VC[neighbor_offset] = true;
                }
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
                auto &offset = last[i];
                auto &vector = index.vectors[offset];

                for (auto iterator = vector.long_edge_out.begin(); iterator != vector.long_edge_out.end(); ++iterator)
                {
                    auto &neighbor_offset = iterator->first;

                    if (!VR.contains(neighbor_offset))
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

        return float(number - 1) / index.count;
    }

    inline uint64_t Calculate_Benefits(const Index &index, const std::unordered_set<Offset> &missed, const Offset start)
    {
        auto visited = std::unordered_set<Offset>();
        visited.insert(start);
        auto last = std::vector<Offset>();
        last.push_back(start);
        auto next = std::vector<Offset>();

        for (auto i = 1; i < index.parameters.cover_range; ++i)
        {
            for (auto j = 0; j < last.size(); ++j)
            {
                auto &offset = last[j];
                auto &vector = index.vectors[offset];

                for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
                {
                    auto &neighbor_offset = iterator->first;

                    if (missed.contains(neighbor_offset) && !visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
                {
                    auto &neighbor_offset = iterator->second;

                    if (missed.contains(neighbor_offset) && !visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        next.push_back(neighbor_offset);
                    }
                }

                for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
                {
                    auto &neighbor_offset = *iterator;

                    if (missed.contains(neighbor_offset) && !visited.contains(neighbor_offset))
                    {
                        visited.insert(neighbor_offset);
                        next.push_back(neighbor_offset);
                    }
                }
            }

            std::swap(last, next);
            next.clear();
        }

        for (auto i = 0; i < last.size(); ++i)
        {
            auto &offset = last[i];
            auto &vector = index.vectors[offset];

            for (auto iterator = vector.short_edge_in.begin(); iterator != vector.short_edge_in.end(); ++iterator)
            {
                auto &neighbor_offset = iterator->first;

                if (missed.contains(neighbor_offset) && !visited.contains(neighbor_offset))
                {
                    visited.insert(neighbor_offset);
                }
            }

            for (auto iterator = vector.short_edge_out.begin(); iterator != vector.short_edge_out.end(); ++iterator)
            {
                auto &neighbor_offset = iterator->second;

                if (missed.contains(neighbor_offset) && !visited.contains(neighbor_offset))
                {
                    visited.insert(neighbor_offset);
                }
            }

            for (auto iterator = vector.keep_connected.begin(); iterator != vector.keep_connected.end(); ++iterator)
            {
                auto &neighbor_offset = *iterator;

                if (missed.contains(neighbor_offset) && !visited.contains(neighbor_offset))
                {
                    visited.insert(neighbor_offset);
                }
            }
        }

        return visited.size();
    }

    // 计算为哪个顶点补长边可以覆盖的顶点最多
    inline Offset Max_Benefits(const Index &index, std::unordered_set<Offset> &missed)
    {
        uint64_t max_benefits = 0;
        uint64_t max_benefit_offset = 0;

        for (auto iterator = missed.begin(); iterator != missed.end(); ++iterator)
        {
            auto offset = *iterator;
            auto benefits = Calculate_Benefits(index, missed, offset);

            if (max_benefits < benefits)
            {
                max_benefits = benefits;
                max_benefit_offset = offset;
            }
        }

        return max_benefit_offset;
    }

    // 优化索引结构
    inline void Optimize(Index &index)
    {
        auto VC = std::vector<bool>(index.vectors.size(), false);
        auto VR = std::unordered_set<Offset>();
        VR.insert(0);
        BFS_Through_LEO(index, VR, VC);

        for (auto i = VR.begin(); i != VR.end(); ++i)
        {
            BFS_Through_SE(index, index.vectors[*i], VC);
        }

        VR.clear();
        auto missed = std::unordered_set<Offset>();

        for (auto offset = 0; offset < VC.size(); ++offset)
        {
            if (!VC[offset] && index.vectors[offset].data != nullptr)
            {
                missed.insert(offset);
            }
        }

        auto offset = Max_Benefits(index, missed);
        auto &vector = index.vectors[offset];
        auto nearest_neighbors = std::priority_queue<std::pair<float, Offset>>();
        auto long_path = std::vector<std::pair<float, Offset>>();
        auto short_path = std::vector<std::pair<float, Offset>>();

        Search_Add(index, vector.data, long_path, short_path, nearest_neighbors);
        Add_Long_Edges(index, long_path, short_path, vector);
    }

} // namespace HSG
