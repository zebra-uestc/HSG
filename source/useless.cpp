inline void Visit_LEO_First_Time(const Index &index, const Vector &processing_vector, const float *target_vector,
                                 std::vector<bool> &visited,
                                 std::priority_queue<std::pair<float, Offset>, std::vector<std::pair<float, Offset>>,
                                                     std::greater<>> &waiting_vectors)
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
    for (auto iterator = processing_vector.short_edge_out.begin(); iterator != processing_vector.short_edge_out.end();
         ++iterator)
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
    for (auto iterator = processing_vector.keep_connected.begin(); iterator != processing_vector.keep_connected.end();
         ++iterator)
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

// Long Edge Reachacble
//
// 通过长边可达
inline bool LE_Reachable(const Index &index, const Offset offset)
{
    auto &vector = index.vectors[offset];

    if (offset == 0)
    {
        return true;
    }
    else if (vector.long_edge_in.empty())
    {
        return false;
    }

    return true;
}

inline void Neighbor_Optimize(Index &index, const Offset offset, std::vector<std::pair<Offset, float>> &all)
{
    auto &new_vector = index.vectors[offset];

    for (auto i = 0; i < all.size(); ++i)
    {
        const auto &neighbor_offset = all[i].first;
        const auto &distance = all[i].second;
        auto &neighbor_vector = index.vectors[neighbor_offset];

        // 如果邻居向量的出边小于短边下限
        if (neighbor_vector.short_edge_out.size() < index.parameters.short_edge_lower_limit)
        {
            // 邻居向量添加出边
            neighbor_vector.short_edge_out.insert({distance, offset});

            // 新向量添加入边
            new_vector.short_edge_in.insert({neighbor_offset, distance});
        }
        // 如果新向量和邻居的距离小于邻居当前距离最大的出边的距离
        else if (distance < neighbor_vector.short_edge_out.rbegin()->first)
        {
            neighbor_vector.short_edge_out.insert({distance, offset});
            new_vector.short_edge_in.insert({neighbor_offset, distance});

            const auto distance = neighbor_vector.short_edge_out.rbegin()->first;
            const auto NN_offset = neighbor_vector.short_edge_out.rbegin()->second;
            auto &NN_vector = index.vectors[NN_offset];

            // 邻居向量删除距离最大的出边
            neighbor_vector.short_edge_out.erase(std::prev(neighbor_vector.short_edge_out.end()));
            NN_vector.short_edge_in.erase(neighbor_offset);

            if (!neighbor_vector.short_edge_in.contains(NN_offset))
            {
                if (!Connected(index, neighbor_offset, NN_offset))
                {
                    if (neighbor_vector.short_edge_out.size() < index.parameters.short_edge_upper_limit)
                    {
                        neighbor_vector.short_edge_out.insert({distance, NN_offset});
                        NN_vector.short_edge_in.insert({neighbor_offset, distance});
                    }
                    else
                    {
                        neighbor_vector.keep_connected.insert(NN_offset);
                        NN_vector.keep_connected.insert(neighbor_offset);
                    }
                }
            }
        }
    }
}
