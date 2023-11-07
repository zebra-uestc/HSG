#pragma once

#include "../distance/cosine_similarity.h"
#include "../distance/euclidean2.h"
#include "../distance/inner_product.h"
#include "../struct/data_struct.h"

typedef float (*distance)(const std::vector<float> &vector1, const std::vector<float> &vector2);

distance get_distance_calculation_function(Distance_Type distance)
{
    switch (distance)
    {
    case Distance_Type::Euclidean2:
        return euclidean2::distance;
        break;
    case Distance_Type::Inner_Product:
        return inner_product::distance;
        break;
    case Distance_Type::Cosine_Similarity:
        return cosine_similarity::distance;
        break;
    default:
        break;
    }
    return euclidean2::distance;
}
