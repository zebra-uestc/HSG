cmake --build build/clang --config Release --target WRA
# cmake --build build/gcc --config Debug --target WRA

data=fashion-mnist
./binary/release/WRA \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data}

data=sift
./binary/release/WRA \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data}

data=gist
./binary/release/WRA \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data}

data=sift10M
./binary/release/WRA \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data}
