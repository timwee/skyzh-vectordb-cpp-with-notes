#include "storage/index/ivfflat_index.h"
#include <algorithm>
#include <optional>
#include <random>
#include "common/exception.h"
#include "execution/expressions/vector_expression.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {
using Vector = std::vector<double>;

IVFFlatIndex::IVFFlatIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                           VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn) {
  std::optional<size_t> lists;
  std::optional<size_t> probe_lists;
  for (const auto &[k, v] : options) {
    if (k == "lists") {
      lists = v;
    } else if (k == "probe_lists") {
      probe_lists = v;
    }
  }
  if (!lists.has_value() || !probe_lists.has_value()) {
    throw Exception("missing options: lists / probe_lists for ivfflat index");
  }
  lists_ = *lists;
  probe_lists_ = *probe_lists;
}

void VectorAdd(Vector &a, const Vector &b) {
  for (size_t i = 0; i < a.size(); i++) {
    a[i] += b[i];
  }
}

void VectorScalarDiv(Vector &a, double x) {
  for (auto &y : a) {
    y /= x;
  }
}

// Find the nearest centroid to the base vector in all centroids
auto FindCentroid(const Vector &vec, const std::vector<Vector> &centroids, VectorExpressionType dist_fn) -> size_t {
  double min_distance = ComputeDistance(vec, centroids[0], dist_fn);
  size_t centroid = 0;
  for (size_t i = 1; i < centroids.size(); i++) {
    double dist = ComputeDistance(vec, centroids[i], dist_fn);
    if (dist < min_distance) {
      centroid = i;
    }
  }
  return centroid;
}

// Compute new centroids based on the original centroids.
// for index building step
auto FindCentroids(const std::vector<std::pair<Vector, RID>> &data, const std::vector<Vector> &centroids,
                   VectorExpressionType dist_fn) -> std::vector<Vector> {
  std::vector<std::pair<Vector, size_t>> new_centroids;
  int dim = centroids[0].size();
  new_centroids.resize(centroids.size());
  for (size_t i = 0; i < centroids.size(); i++) {
    new_centroids[i].first.resize(dim);
  }
  for (const auto &[vec, rid] : data) {
    auto centroid = FindCentroid(vec, centroids, dist_fn);
    // adjust based on mean of all points in that current cluster
    VectorAdd(new_centroids[centroid].first, vec);
    new_centroids[centroid].second += 1;
  }
  std::vector<Vector> final_centroids;
  for (const auto &[vec, sz] : new_centroids) {
    auto v = vec;
    VectorScalarDiv(v, sz);
    final_centroids.emplace_back(v);
  }
  return final_centroids;
}

void IVFFlatIndex::BuildIndex(std::vector<std::pair<Vector, RID>> initial_data) {
    if (initial_data.empty()) {
      return;
    }
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::shuffle(initial_data.begin(), initial_data.end(), generator);
    std::vector<Vector> centroids;
    // just take first num_lists_ data for initializing k-means clusters
    for (size_t i = 0; i < lists_ && i < initial_data.size(); i++) {
      centroids.push_back(initial_data[i].first);
    }
    for (size_t iter = 0; iter < 500; iter++) {
      centroids = FindCentroids(initial_data, centroids, distance_fn_);
    }
    centroids_.clear();
    centroids_buckets_.clear();
    for (const auto &centroid : centroids) {
      centroids_.emplace_back(centroid);
      centroids_buckets_.emplace_back();
    }
    for (const auto &[vec, rid] : initial_data) {
      auto centroid = FindCentroid(vec, centroids, distance_fn_);
      centroids_buckets_[centroid].emplace_back(vec, rid);
    }
}

void IVFFlatIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  auto centroid = FindCentroid(key, centroids_, distance_fn_);
  centroids_buckets_[centroid].emplace_back(key, rid);
}

auto IVFFlatIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  std::vector<size_t> centroids;
  for (size_t i = 0; i < centroids_.size(); i++) {
    centroids.push_back(i);
  }
  // sort centroids according to distance to vector to search
  std::sort(centroids.begin(), centroids.end(), [&](const auto &a, const auto &b) {
    auto dist_a = ComputeDistance(base_vector, centroids_[a], distance_fn_);
    auto dist_b = ComputeDistance(base_vector, centroids_[b], distance_fn_);
    return dist_a < dist_b;
  });

  // search up to probe_lists centroids/clusters
  std::vector<std::pair<Vector, RID>> selections;
  for (size_t i = 0; i < probe_lists_; i++) {
    for (const auto &[vec, rid] : centroids_buckets_[centroids[i]]) {
      selections.emplace_back(vec, rid);
    }
  }

  // take up toe limit of the closest vectors
  std::sort(selections.begin(), selections.end(), [&](const auto &a, const auto &b) {
    auto dist_a = ComputeDistance(base_vector, a.first, distance_fn_);
    auto dist_b = ComputeDistance(base_vector, b.first, distance_fn_);
    return dist_a < dist_b;
  });
  std::vector<RID> result;
  for (size_t i = 0; i < limit && i < selections.size(); i++) {
    result.push_back(selections[i].second);
  }
  return result;
}

}  // namespace bustub
