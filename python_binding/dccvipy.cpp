#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../source/dccvi.h"
#include "../source/distance.h"

namespace py = pybind11;

std::vector<uint64_t> wrapped_query(const dccvi::Index<float> &index, const std::vector<float> &query_vector,
                                    uint64_t top_k, uint64_t relaxed_monotonicity = 0)
{
    auto result = std::vector<uint64_t>();
    result.reserve(top_k);
    for (const auto &i : dccvi::query(index, query_vector, top_k, relaxed_monotonicity))
    {
        result.push_back(i.second);
    }
    return result;
}

PYBIND11_MODULE(dccvipy, index)
{
    py::class_<dccvi::Index<float>>(index, "Index")
        .def(py::init<const Distance_Type, const uint64_t, const uint64_t, const uint64_t, const uint64_t,
                      const uint64_t>());
    py::enum_<Distance_Type>(index, "Distance_Type")
        .value("Euclidean2", Distance_Type::Euclidean2)
        .value("Inner_Product", Distance_Type::Inner_Product)
        .value("Cosine_Similarity", Distance_Type::Cosine_Similarity);
    index.def("insert", &dccvi::insert<float>, "Insert a vector into index.");
    index.def("query", &wrapped_query, "query the most k-th nearest neighbors of a vector.");
}
