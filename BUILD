load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

# cc_import(
#   name = "gflags",
#   hdrs = [
#     # "//third_party/gflags",
#     # "/usr/include/gflags/gflags_completions.h",
#     # "/usr/include/gflags/gflags_declare.h",
#     # "/usr/include/gflags/gflags_gflags.h",
#     # "/usr/include/gflags/gflags.h",
#     "//third_party/gflags/gflags.h",
#   ],
#   # mylib.lib is an import library for mylib.dll which will be passed to linker
#   # interface_library = "mylib.lib",
#   # mylib.dll is provided by system environment, for example it can be found in PATH.
#   # This indicates that Bazel is not responsible for making mylib.dll available.
#   system_provided = 1,
# )

tf_cc_binary(
    name = "model",
    srcs = [
        "model.cc",
        "data_set.h",
        "data_set.cc"
    ],
    deps = [
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:grad_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/core:tensorflow",
        # ":gflags",
        "//third_party/gflags:gflags",
        "//tensorflow/c:c_api",
        # Import some boiler plate for interacting with the C-API.
        "//tensorflow/c:c_test_util_lite",
    ],
    data = ["normalized_car_features.csv"]
)

# Arcade-Learning-Environment uses cmake.
# Hack to call cmake from bazel (to avoid maintaining bazel build config for it)
# https://blog.envoyproxy.io/external-c-dependency-management-in-bazel-dd37477422f5
# FOO_GENRULE_BUILD = """
# genrule(
#     name = "bar_genrule",
#     srcs = glob(["src/**.cc", "src/**.h"]),
#     outs = ["bar.a", "bar_0.h", "bar_1.h", ..., "bar_N.h"],
#     cmd = "./configure && make",
# )
# cc_library(
#     name = "bar",
#     srcs = ["bar.a"],
#     hdrs = ["bar_0.h", "bar_1.h", ..., "bar_N.h"],
# )
# """

# FOO_GENRULE_BUILD = """
# genrule(
#     name = "ale_genrule",
#     srcs = glob(["src/**.cc", "src/**.h"]),
#     outs = ["bar.a", "bar_0.h", "bar_1.h", ..., "bar_N.h"],
#     cmd = "./configure && make",
# )
# cc_library(
#     name = "ale",
#     srcs = ["bar.a"],
#     hdrs = ["bar_0.h", "bar_1.h", ..., "bar_N.h"],
# )
# """
#
# new_git_repository(
#     # name = "foo",
#     name = "Arcade-Learning-Environment",
#     remote = "git@github.com:mgbellemare/Arcade-Learning-Environment.git",
#     # remote = "https://github.com/something/foo.git",
#     commit = "4374be38e9a75ff5957c3922adb155d32086fe14",
#     build_file_content = FOO_GENRULE_BUILD,
# )
