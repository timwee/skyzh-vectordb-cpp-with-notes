#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"
#include "execution/expressions/vector_expression.h"
#include "fmt/format.h"
#include "fmt/std.h"
#include "storage/index/hnsw_index.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

// #define NSW_ONLY  // only enable NSW code path without HNSW

namespace bustub {
HNSWIndex::HNSWIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                     VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn),
      vertices_(std::make_unique<std::vector<Vector>>()),
      layers_{{*vertices_, distance_fn}} {
  std::optional<size_t> m;
  std::optional<size_t> ef_construction;
  std::optional<size_t> ef_search;
  for (const auto &[k, v] : options) {
    if (k == "m") {
      m = v;
    } else if (k == "ef_construction") {
      ef_construction = v;
    } else if (k == "ef_search") {
      ef_search = v;
    }
  }
  if (!m.has_value() || !ef_construction.has_value() || !ef_search.has_value()) {
    throw Exception("missing options: m / ef_construction / ef_search for hnsw index");
  }
  ef_construction_ = *ef_construction;
  m_ = *m;
  ef_search_ = *ef_search;
  m_max_ = m_;
  m_max_0_ = m_ * m_;
  layers_[0].m_max_ = m_max_0_;
  m_l_ = 1.0 / std::log(m_);
  std::random_device rand_dev;
  generator_ = std::mt19937(rand_dev());
}

auto SelectNeighbors(const std::vector<double> &vec, const std::vector<size_t> &vertex_ids,
                     const std::vector<std::vector<double>> &vertices, size_t m, VectorExpressionType dist_fn)
    -> std::vector<size_t> {
  std::vector<std::pair<double, size_t>> distances;
  distances.reserve(vertex_ids.size());
  for (const auto vert : vertex_ids) {
    distances.emplace_back(ComputeDistance(vertices[vert], vec, dist_fn), vert);
  }
  std::sort(distances.begin(), distances.end());
  std::vector<size_t> selected_vs;
  selected_vs.reserve(vertex_ids.size());
  for (size_t i = 0; i < m && i < distances.size(); i++) {
    selected_vs.emplace_back(distances[i].second);
  }
  return selected_vs;
}

/**
 C <- entry_points as min heap on distance
 W <- entry_points as max heap on distance
 visited <- entry_points as unordered set
 while not C.empty():
     node <- pop C (nearest element in C)
     if dist(node) > dist(top W): # top W is the furthest element in W
         break
     for neighbor in neighbors of node:
         if not visited neighbor:
             visited += neighbor
             C += neighbor
             W += neighbor
             retain k-nearest elements in W
 return W
 */
auto NSW::SearchLayer(const std::vector<double> &base_vector, size_t limit, const std::vector<size_t> &entry_points)
    -> std::vector<size_t> {
  BUSTUB_ASSERT(limit > 0, "limit > 0");
  std::vector<size_t> candidates;
  std::unordered_set<size_t> visited;
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::greater<>> explore_q;
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::less<>> result_set;
  for (const auto entry_point : entry_points) {
    auto dist = ComputeDistance(vertices_[entry_point], base_vector, dist_fn_);
    explore_q.emplace(dist, entry_point);
    result_set.emplace(dist, entry_point);
    visited.emplace(entry_point);
  }
  while (!explore_q.empty()) {
    auto [dist, vertex] = explore_q.top();
    explore_q.pop();
    if (dist > result_set.top().first) {
      break;
    }
    BUSTUB_ASSERT(in_vertices_.size() <= 1 || !edges_[vertex].empty(), "not in the layer");
    for (const auto &neighbor : edges_[vertex]) {
      if (visited.find(neighbor) == visited.end()) {
        visited.emplace(neighbor);
        auto dist = ComputeDistance(vertices_[neighbor], base_vector, dist_fn_);
        explore_q.emplace(dist, neighbor);
        result_set.emplace(dist, neighbor);
        while (result_set.size() > limit) {
          result_set.pop();
        }
      }
    }
  }
  while (!result_set.empty()) {
    candidates.push_back(result_set.top().second);
    result_set.pop();
  }
  std::reverse(candidates.begin(), candidates.end());
  return candidates;
}

auto NSW::AddVertex(size_t vertex_id) { in_vertices_.push_back(vertex_id); }

auto NSW::Insert(const std::vector<double> &vec, size_t vertex_id, size_t ef_construction, size_t m) {
  if (!in_vertices_.empty()) {
    auto ef_con_nodes = SearchLayer(vec, ef_construction, {DefaultEntryPoint()});
    auto neighbors = SelectNeighbors(vec, ef_con_nodes, vertices_, m, dist_fn_);
    for (const auto neighbor : neighbors) {
      Connect(vertex_id, neighbor);
    }

    // prune neighbors edges if > max
    for (const auto neighbor : neighbors) {
      auto &edges = edges_[neighbor];
      if (edges.size() > m_max_) {
        auto new_neighbors = SelectNeighbors(vertices_[neighbor], edges, vertices_, m_max_, dist_fn_);
        edges = new_neighbors;
      }
    }
  }
  AddVertex(vertex_id);
}

void NSW::Connect(size_t vertex_a, size_t vertex_b) {
  edges_[vertex_a].push_back(vertex_b);
  edges_[vertex_b].push_back(vertex_a);
}

auto HNSWIndex::AddVertex(const std::vector<double> &vec, RID rid) -> size_t {
  auto id = vertices_->size();
  vertices_->emplace_back(vec);
  rids_.emplace_back(rid);
  return id;
}

void HNSWIndex::BuildIndex(std::vector<std::pair<std::vector<double>, RID>> initial_data) {
  std::shuffle(initial_data.begin(), initial_data.end(), generator_);

  for (const auto &[vec, rid] : initial_data) {
    InsertVectorEntry(vec, rid);
  }
}

#ifdef NSW_ONLY

auto HNSWIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  auto vertex_ids = layers_[0].SearchLayer(base_vector, limit, {layers_[0].DefaultEntryPoint()});
  std::vector<RID> result;
  result.reserve(vertex_ids.size());
  for (const auto &id : vertex_ids) {
    result.push_back(rids_[id]);
  }
  return result;
}

void HNSWIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  auto id = AddVertex(key, rid);
  layers_[0].Insert(key, id, ef_construction_, m_);
}

#else

/*
ep = upper-most level entry point
target_level = generate random level based on m_L
for go down one level until target_level + 1
    ep <- layers[level].search(ep=ep, limit=1, search_target)
for go down one level until level 0
    ep <- layers[level].search(ep=ep, limit=ef_construction, search_target)
    neighbors <- m-nearest neighbor in ep
    connect neighbors with search_target
    purge edges of neighbors if larger than m_max of that layer
 */
auto HNSWIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  std::vector<size_t> entry_points{layers_[layers_.size() - 1].DefaultEntryPoint()};
  for (int level = layers_.size() - 1; level >= 1; level--) {
    auto nearest_elements = layers_[level].SearchLayer(base_vector, ef_search_, entry_points);
    nearest_elements = SelectNeighbors(base_vector, nearest_elements, *vertices_, 1, distance_fn_);
    entry_points = {nearest_elements[0]};
  }
  auto neighbors = layers_[0].SearchLayer(base_vector, limit > ef_search_ ? limit : ef_search_, entry_points);
  neighbors = SelectNeighbors(base_vector, neighbors, *vertices_, limit, distance_fn_);
  std::vector<RID> result;
  result.reserve(neighbors.size());
  for (auto id : neighbors) {
    result.push_back(rids_[id]);
  }
  return result;
}

void HNSWIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  std::uniform_real_distribution<double> level_dist(0.0, 1.0);
  auto vertex_id = AddVertex(key, rid);
  int target_level = static_cast<int>(std::floor(-std::log(level_dist(generator_)) * m_l_));
  BUSTUB_ASSERT(target_level >= 0, "invalid target level");
  std::vector<size_t> nearest_elements;
  if (!layers_[0].in_vertices_.empty()) {
    std::vector<size_t> entry_points{layers_[layers_.size() - 1].DefaultEntryPoint()};
    int level = layers_.size() - 1;
    for (; level > target_level; level--) {
      nearest_elements = layers_[level].SearchLayer(key, ef_search_, entry_points);
      nearest_elements = SelectNeighbors(key, nearest_elements, *vertices_, 1, distance_fn_);
      entry_points = {nearest_elements[0]};
    }
    for (; level >= 0; level--) {
      auto &layer = layers_[level];
      nearest_elements = layer.SearchLayer(key, ef_construction_, entry_points);
      auto neighbors = SelectNeighbors(key, nearest_elements, *vertices_, m_, distance_fn_);
      layer.AddVertex(vertex_id);
      for (const auto neighbor : neighbors) {
        layer.Connect(vertex_id, neighbor);
      }
      for (const auto neighbor : neighbors) {
        auto &edges = layer.edges_[neighbor];
        if (edges.size() > m_max_) {
          auto new_neighbors = SelectNeighbors((*vertices_)[neighbor], edges, *vertices_, layer.m_max_, distance_fn_);
          edges = new_neighbors;
        }
      }
      entry_points = nearest_elements;
    }
  } else {
    layers_[0].AddVertex(vertex_id);
  }
  while (static_cast<int>(layers_.size()) <= target_level) {
    auto layer = NSW{*vertices_, distance_fn_, m_max_};
    layer.AddVertex(vertex_id);
    layers_.emplace_back(std::move(layer));
  }
}

#endif

}  // namespace bustub
