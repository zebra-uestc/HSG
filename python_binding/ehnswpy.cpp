#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../source/distance.h"
#include "../source/ehnsw.h"

namespace py = pybind11;

PYBIND11_MODULE(ehnswpy, index)
{
    py::class_<ehnsw::Index<float>>(index, "Index").def(py::init<const Distance_Type>());
    py::enum_<Distance_Type>(index, "Distance_Type")
        .value("Euclidean2", Distance_Type::Euclidean2)
        .value("Inner_Product", Distance_Type::Inner_Product)
        .value("Cosine_Similarity", Distance_Type::Cosine_Similarity);
    index.def("insert", &ehnsw::insert<float>, "Insert a vector into index.");
    index.def("query", &ehnsw::query<float>, "query the most k-th nearest neighbors of a vector.");
}
