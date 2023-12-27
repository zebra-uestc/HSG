#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../source/dehnsw.h"
#include "../source/distance.h"

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

py::array_t<uint64_t> wrapped_query(const dehnsw::Index &index, const py::object &query_vector, uint64_t top_k,
                                    uint64_t relaxed_monotonicity = 0)
{
    py::array_t<float, py::array::c_style | py::array::forcecast> items(query_vector);
    auto return_result = py::array_t<uint64_t>(top_k);
    auto return_buffer = (float *)return_result.request().ptr;
    auto result = dehnsw::query(index, items.data(0), top_k, relaxed_monotonicity);
    for (int i = top_k - 1; i >= 0; i--)
    {
        return_buffer[i] = result.top().second;
        result.pop();
    }
    return return_result;
}

PYBIND11_MODULE(dehnswpy, index)
{
    py::class_<dehnsw::Index>(index, "Index")
        .def(py::init<const Distance_Type, const uint64_t, const uint64_t, const uint64_t, const uint64_t,
                      const uint64_t>());
    py::enum_<Distance_Type>(index, "Distance_Type")
        .value("Euclidean2", Distance_Type::Euclidean2)
        .value("Inner_Product", Distance_Type::Inner_Product)
        .value("Cosine_Similarity", Distance_Type::Cosine_Similarity);
    index.def("insert", &dehnsw::insert, "Insert a vector into index.");
    index.def("query", &wrapped_query, "query the most k-th nearest neighbors of a vector.");
}
