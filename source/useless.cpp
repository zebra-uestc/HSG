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
