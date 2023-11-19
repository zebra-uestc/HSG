#pragma once

#include <cinttypes>
#include <concepts>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Cluster;
class Layer;

// 向量
template <typename Dimension_Type> class Vector
{
  private:
  public:
    std::vector<Dimension_Type> data;

    explicit Vector(const std::vector<Dimension_Type> &data)
    {
        this->data = data;
    }
};

// 簇中的向量
class Vector_In_Cluster
{
  private:
  public:
    // 向量在原始数据的偏移量
    uint64_t global_offset;
    // 向量对应的下一层中的向量
    std::weak_ptr<Vector_In_Cluster> lower_layer;
    // 指向自己所在的簇
    std::weak_ptr<Cluster> cluster;
    // 指向的邻居向量
    std::multimap<float, std::weak_ptr<Vector_In_Cluster>> out;
    // 指向该向量的邻居向量
    std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>> in;

    explicit Vector_In_Cluster(const uint64_t global_offset)
    {
        this->global_offset = global_offset;
        this->out = std::multimap<float, std::weak_ptr<Vector_In_Cluster>>();
        this->in = std::unordered_map<uint64_t, std::weak_ptr<Vector_In_Cluster>>();
    }
};

// 每层中的簇
class Cluster
{
  private:
  public:
    // 该簇所在的层
    std::weak_ptr<Layer> layer;
    // 簇中的向量
    std::unordered_map<uint64_t, std::shared_ptr<Vector_In_Cluster>> vectors;
    // 该簇中被选出的向量在原始数据的偏移量
    std::unordered_set<uint64_t> selected_vectors;

    explicit Cluster(const std::weak_ptr<Layer> &layer)
    {
        this->layer = layer;
        this->vectors = std::unordered_map<uint64_t, std::shared_ptr<Vector_In_Cluster>>();
        this->selected_vectors = std::unordered_set<uint64_t>();
    }

    // 返回当前簇中所有的簇
    std::vector<std::shared_ptr<Cluster>> calculate_clusters()
    {
        uint64_t hit = 0;
        std::vector<std::shared_ptr<Cluster>> new_clusters;
        // key：向量的全局偏移量
        // value：向量在第几个新的簇中
        std::unordered_map<uint64_t, uint64_t> flag;
        uint64_t new_cluster_number = 0;
        for (auto &vector_iterator : this->vectors)
        {
            if (!flag.contains(vector_iterator.first))
            {
                ++hit;
                flag.emplace(vector_iterator.first, new_cluster_number);
                std::shared_ptr<Cluster> temporary_cluster = std::make_shared<Cluster>(this->layer);
                vector_iterator.second->cluster = temporary_cluster;
                temporary_cluster->vectors.emplace(vector_iterator.first, vector_iterator.second);
                std::queue<uint64_t> waiting_vectors;
                waiting_vectors.push(vector_iterator.first);
                while (!waiting_vectors.empty())
                {
                    uint64_t neighbor_iteration = waiting_vectors.front();
                    waiting_vectors.pop();
                    for (auto &out_iterator : this->vectors.find(neighbor_iteration)->second->out)
                    {
                        std::shared_ptr<Vector_In_Cluster> temporary_vector_pointer = out_iterator.second.lock();
                        if (flag.emplace(temporary_vector_pointer->global_offset, new_cluster_number).second)
                        {
                            ++hit;
                            temporary_vector_pointer->cluster = temporary_cluster;
                            temporary_cluster->vectors.emplace(temporary_vector_pointer->global_offset,
                                                               temporary_vector_pointer);
                        }
                    }
                    for (auto &in_iteration : this->vectors.find(neighbor_iteration)->second->in)
                    {
                        if (flag.emplace(in_iteration.first, new_cluster_number).second)
                        {
                            ++hit;
                            in_iteration.second.lock()->cluster = temporary_cluster;
                            temporary_cluster->vectors.emplace(in_iteration.first, in_iteration.second);
                        }
                    }
                }
                new_clusters.push_back(std::move(temporary_cluster));
                ++new_cluster_number;
            }
            if (hit == this->vectors.size())
            {
                break;
            }
        }
        for (auto &selected_vector : this->selected_vectors)
        {
            new_clusters[flag.find(selected_vector)->second]->selected_vectors.insert(selected_vector);
        }
        return new_clusters;
    }
};

// 索引中的一层
class Layer
{
  private:
  public:
    // 每层中的多个簇
    std::vector<std::shared_ptr<Cluster>> clusters;
    // 上一层
    std::weak_ptr<Layer> upper_layer;
    // 下一层
    std::weak_ptr<Layer> lower_layer;

    Layer()
    {
        this->clusters = std::vector<std::shared_ptr<Cluster>>();
    }

    // 合并两个簇
    void merge_two_clusters(const std::shared_ptr<Cluster> &target_cluster,
                            const std::shared_ptr<Cluster> &merged_cluster)
    {
        // 移动被合并的簇中的向量到目标簇中
        for (auto &merged_cluster_vector : merged_cluster->vectors)
        {
            merged_cluster_vector.second->cluster = target_cluster;
            target_cluster->vectors.insert(merged_cluster_vector);
        }
        // 移动被合并的簇中被选出的代表向量到目标簇中
        for (auto &merged_cluster_selected_vector : merged_cluster->selected_vectors)
        {
            target_cluster->selected_vectors.insert(merged_cluster_selected_vector);
        }
        // 删除簇
        for (auto cluster_iterator = this->clusters.begin(); cluster_iterator != this->clusters.end();
             ++cluster_iterator)
        {
            if (*cluster_iterator == merged_cluster)
            {
                this->clusters.erase(cluster_iterator);
                break;
            }
        }
    }

    // 分裂一个簇
    // 新的簇中可能没有被选出的向量在上一层中
    // 返回需要插入到上一层中的向量
    std::vector<std::shared_ptr<Vector_In_Cluster>> divide_a_cluster(const std::shared_ptr<Cluster> &cluster)
    {
        auto new_clusters = cluster->calculate_clusters();
        std::vector<std::shared_ptr<Vector_In_Cluster>> selected_vectors;
        if (cluster->layer.lock()->upper_layer.expired())
        {
            for (auto &cluster_iterator : this->clusters)
            {
                if (cluster_iterator == cluster)
                {
                    cluster_iterator = new_clusters[0];
                    break;
                }
            }
            for (auto new_cluster_offset = 1; new_cluster_offset < new_clusters.size(); ++new_cluster_offset)
            {
                this->clusters.push_back(new_clusters[new_cluster_offset]);
            }
            return selected_vectors;
        }
        if (new_clusters[0]->selected_vectors.empty())
        {
            selected_vectors.push_back(std::make_shared<Vector_In_Cluster>(new_clusters[0]->vectors[0]->global_offset));
            selected_vectors[selected_vectors.size() - 1]->lower_layer = new_clusters[0]->vectors[0];
        }
        for (auto &cluster_iterator : this->clusters)
        {
            if (cluster_iterator == cluster)
            {
                cluster_iterator = new_clusters[0];
                break;
            }
        }
        for (auto new_cluster_offset = 1; new_cluster_offset < new_clusters.size(); ++new_cluster_offset)
        {
            if (new_clusters[new_cluster_offset]->selected_vectors.empty())
            {
                selected_vectors.push_back(
                    std::make_shared<Vector_In_Cluster>(new_clusters[new_cluster_offset]->vectors[0]->global_offset));
                selected_vectors[selected_vectors.size() - 1]->lower_layer =
                    new_clusters[new_cluster_offset]->vectors[0];
            }
            this->clusters.push_back(new_clusters[new_cluster_offset]);
        }
        return selected_vectors;
    }
};
