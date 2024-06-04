#include "universal.h"

std::vector<std::vector<float>> train;
std::vector<std::vector<float>> test;
std::vector<std::vector<uint64_t>> neighbors;
std::vector<std::vector<float>> reference_answer;
std::string name;

int main(int argc, char **argv)
{
    name = std::string(argv[5]);

    if (name == "sift10M")
    {
        bvecs_vectors(argv[1], train, 10000000);
        bvecs_vectors(argv[2], test);
        ivecs(argv[3], neighbors);
    }
    else
    {
        train = load_vector(argv[1]);
        test = load_vector(argv[2]);
        neighbors = load_neighbors(argv[3]);
    }

    reference_answer = get_reference_answer(train, test, neighbors);
    write_reference_answer(argv[4], reference_answer);

    return 0;
}
