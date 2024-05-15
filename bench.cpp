#include <hnswlib/hnswalg.h>
#include <hnswlib/space_l2.h>
#include <memory>
#include <ostream>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "hnswlib/hnswlib.h"
#include "usearch/index.hpp"
#include "usearch/index_dense.hpp"
#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"

static const int kDim = 1000;
static const int kNumVector = 10000;
static const int kTopK = 10;

static std::vector<std::vector<float>> GenerateRandomVectors() {
  static std::vector<std::vector<float>> data;
  if (!data.empty()) {
    return data;
  }

  data.reserve(kNumVector);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0f, 100.0f);

  for (int i = 0; i < kNumVector; ++i) {
    std::vector<float> vec;
    vec.reserve(kDim);
    for (int j = 0; j < kDim; ++j) {
      vec.push_back(dis(gen));
    }
    data.push_back(vec);
  }

  std::cout << "random vectors generated" << std::endl;

  return data;
}

static void BM_HNSWLib(benchmark::State &state) {
  // Perform setup here
  hnswlib::L2Space space(kDim);
  auto hnsw =
      std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, kNumVector);
  const auto &data = GenerateRandomVectors();
  for (int i = 0; i < data.size(); i++) {
    hnsw->addPoint(data[i].data(), i);
  }
  std::cout << "hnswlib construction done." << std::endl;

  for (auto _ : state) {
    for (int i = 0; i < 10; i++) {
      std::vector<std::pair<float, hnswlib::labeltype>> result =
          hnsw->searchKnnCloserFirst(data[i].data(), kTopK);
    }
  }
}

using namespace unum;
using namespace unum::usearch;

static void BM_Usearch(benchmark::State &state) {
  const auto &data = GenerateRandomVectors();
  metric_punned_t metric(kDim, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);

  // If you plan to store more than 4 Billion entries - use `index_dense_big_t`.
  // Or directly instantiate the template variant you need -
  // `index_dense_gt<vector_key_t, internal_id_t>`.
  index_dense_config_t config;
  config.connectivity = 16; // M
  config.expansion_add = 200; // efConstruction
  config.expansion_search = 10; // ef

  index_dense_t index = index_dense_t::make(metric, config);

  index.reserve(kNumVector);

  for (int i = 0; i < data.size(); i++) {
    index.add(i, data[i].data());
  }
  std::cout << "usearch construction done." << std::endl;
  for (auto _ : state) {
    for (int i = 0; i < 10; i++) {
      auto results = index.search(data[i].data(), kTopK);
    }
  }
}

using namespace Annoy;

static void BM_Annoy(benchmark::State& state) {
  const auto &data = GenerateRandomVectors();
  auto index = AnnoyIndex<int, float, Annoy::Angular, Annoy::Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>(kDim);

  for (int i = 0; i < data.size(); i++) {
    index.add_item(i, data[i].data());
  }
  for (auto _ : state) {
    for (int i = 0; i < 10; i++) {
      std::vector<int> result;
      index.get_nns_by_vector(data[i].data(), kTopK, -1, &result, nullptr);
    }
  }
}

// Register the function as a benchmark
BENCHMARK(BM_HNSWLib);
BENCHMARK(BM_Usearch);
BENCHMARK(BM_Annoy);