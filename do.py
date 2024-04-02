import argparse
import os

parser = argparse.ArgumentParser(
    description="build, generate, debug, performance test, clear"
)
group = parser.add_mutually_exclusive_group()
group.add_argument("-b", "--build", action="store_true")
group.add_argument("-g", "--generate", action="store_true")
group.add_argument("-d", "--debug", action="store_true")
group.add_argument("-p", "--performance_test", action="store_true")
group.add_argument("-c", "--clear", action="store_true")
parser.add_argument(
    "-a", "--arguments", default="", type=str, help="arguments to be used"
)
parser.add_argument("-data", "--data_set", default="", type=str, help="choose dataset")
args = parser.parse_args()
if args.build:
    print("---")
    print("build project with command:")
    print('cmake -S . -B build -G"Ninja Multi-Config" {0}'.format(args.arguments))
    print()
    os.system('cmake -S . -B build -G"Ninja Multi-Config" {0}'.format(args.arguments))
    print()
    print("build done")
    print("---")
elif args.generate:
    if str(args.arguments).lower() == "release":
        print("---")
        print("generating(release)")
        print()
        os.system("cmake --build build --config Release")
        print()
        print("generating(release) done")
        print("---")
    elif str(args.arguments).lower() == "debug":
        print("---")
        print("generating(debug)")
        print()
        os.system("cmake --build build --config Debug")
        print()
        print("generating(debug) done")
        print("---")
    elif args.arguments == "":
        print("---")
        print("generating(release)")
        print()
        os.system("cmake --build build --config Release")
        print()
        print("generating(release) done")
        print("---")
        print("---")
        print("generating(debug)")
        print()
        os.system("cmake --build build --config Debug")
        print()
        print("generating(debug) done")
        print("---")
    else:
        print("build only has two arguments: release, debug")
elif args.debug:
    data_set = "fashion-mnist"
    if args.data_set == "":
        print("use default data set: fashion-mnist")
    else:
        data_set = args.data_set
    if (
        os.path.isfile(os.path.join("./data/", data_set, "train"))
        and os.path.isfile(os.path.join("./data/", data_set, "test"))
        and os.path.isfile(os.path.join("./data/", data_set, "neighbors"))
    ):
        os.system(
            "lldb ./binary/debug/miluann_example ./data/{0}/train ./data/{0}/test ./data/{0}/neighbors".format(
                data_set
            )
        )
    else:
        print("please download the data set and extract data.")
elif args.performance_test:
    data_set = "fashion-mnist"
    if args.data_set == "":
        print("use default data set: fashion-mnist")
    else:
        data_set = args.data_set
    if (
        os.path.isfile(os.path.join("./data/", data_set, "train"))
        and os.path.isfile(os.path.join("./data/", data_set, "test"))
        and os.path.isfile(os.path.join("./data/", data_set, "neighbors"))
    ):
        os.system(
            "./binary/release/miluann_example ./data/{0}/train ./data/{0}/test ./data/{0}/neighbors".format(
                data_set
            )
        )
    else:
        print("please download the data set and extract data.")
elif args.clear:
    os.system("rm -rf build")
    os.system("rm -rf binary")
