#include <memory>
#include <optional>
#include "binder/bound_order_by.h"
#include "catalog/catalog.h"
#include "catalog/column.h"
#include "concurrency/transaction.h"
#include "execution/expressions/array_expression.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/vector_expression.h"
#include "execution/plans/abstract_plan.h"
#include "execution/plans/index_scan_plan.h"
#include "execution/plans/limit_plan.h"
#include "execution/plans/projection_plan.h"
#include "execution/plans/seq_scan_plan.h"
#include "execution/plans/sort_plan.h"
#include "execution/plans/vector_index_scan_plan.h"
#include "fmt/core.h"
#include "optimizer/optimizer.h"
#include "type/type.h"
#include "type/type_id.h"

namespace bustub {

auto MatchVectorIndex(const Catalog &catalog, table_oid_t table_oid, uint32_t col_idx, VectorExpressionType dist_fn,
                      const std::string &vector_index_match_method) -> const IndexInfo * {
  if (vector_index_match_method == "none") {
    // do not match any index if it is set to none
    return nullptr;
  }
  auto table_indexes = catalog.GetTableIndexes(catalog.GetTable(table_oid)->name_);
  for (const auto *index : table_indexes) {
    if (index->index_type_ == IndexType::VectorIVFFlatIndex || index->index_type_ == IndexType::VectorHNSWIndex) {
      if (vector_index_match_method == "ivfflat" && index->index_type_ != IndexType::VectorIVFFlatIndex) {
        continue;
      }
      if (vector_index_match_method == "hnsw" && index->index_type_ != IndexType::VectorHNSWIndex) {
        continue;
      }
      auto *vector_index = dynamic_cast<const VectorIndex *>(index->index_.get());
      BUSTUB_ASSERT(vector_index != nullptr, "??");
      if (vector_index->GetKeyAttrs().size() == 1 && vector_index->GetKeyAttrs()[0] == col_idx &&
          vector_index->distance_fn_ == dist_fn) {
        return index;
      }
    }
  }
  return nullptr;
}

auto Optimizer::OptimizeAsVectorIndexScan(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  std::vector<AbstractPlanNodeRef> children;
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeAsVectorIndexScan(child));
  }
  auto optimized_plan = plan->CloneWithChildren(std::move(children));

  /**
   * You may choose to run the conversion rule before or after the top-n optimization rule. If you choose to invoke this
   rule before converting to top-n, you will need to match sort and limit plan nodes. Otherwise, you may want to match
   the top-n plan node. The rule order can be found in optimizer_custom_rules.cpp.

   There are three cases you should consider:

   Case 1: TopN directly followed by SeqScan

   TopN { n=2, order_bys=[("Default", "l2_dist([1.000000,1.000000,1.000000], #0.0)")]}
     SeqScan { table=t1 }
   Case 2: TopN followed by Projection

   TopN { n=2, order_bys=[("Default", "l2_dist([1.000000,1.000000,1.000000], #0.0)")]}
     Projection { exprs=["#0.0", "l2_dist([1.000000,1.000000,1.000000], #0.0)"] }
       SeqScan { table=t1 }
   Case 3: TopN followed by Projection with column reference shuffled


   TopN { n=2, order_bys=[("Default", "l2_dist([1.000000,1.000000,1.000000], #0.1)")]}
     Projection { exprs=["#0.1", "#0.0"] }
       SeqScan { table=t1 }
   */


  // match the pattern: (topn is sort then limit?)
  //    All cases have TopN first then SeqScan last.
  // Limit
  //   Sort vector (constant) <-> vector
  //     Projection+SeqScan or SeqScan

  if (optimized_plan->GetType() != PlanType::Limit) {
    return optimized_plan;
  }

  auto limit_node = dynamic_cast<const LimitPlanNode *>(optimized_plan.get());
  auto limit_n = limit_node->limit_;

  auto limit_child = limit_node->GetChildAt(0);
  if (limit_child->GetType() != PlanType::Sort) {
    return optimized_plan;
  }

  auto sort_node = dynamic_cast<const SortPlanNode *>(limit_child.get());

  // only support single order bys
  if (sort_node->order_bys_.size() != 1) {
    return optimized_plan;
  }

  auto [orderby_order, orderby_expr] = sort_node->order_bys_[0];
  if (orderby_order == OrderByType::DESC) {
    return optimized_plan;
  }

  // if not distance expr - exit
  auto *distance_expr = dynamic_cast<const VectorExpression *>(orderby_expr.get());
  if (distance_expr == nullptr) {
    return optimized_plan;
  }
  std::shared_ptr<const ArrayExpression> base_vector;
  std::shared_ptr<const ColumnValueExpression> column_vector;
  VectorExpressionType vty = distance_expr->expr_type_;

  // handle distance_expr params. Example queries:
  // EXPLAIN (o) SELECT v1 FROM t1 ORDER BY ARRAY [1.0, 1.0, 1.0] <-> v1 LIMIT 2;
  // EXPLAIN (o) SELECT * FROM t1 ORDER BY ARRAY [1.0, 1.0, 1.0] <-> v1 LIMIT 2;
  // EXPLAIN (o) SELECT v1, ARRAY [1.0, 1.0, 1.0] <-> v1 FROM t1 ORDER BY ARRAY [1.0, 1.0, 1.0] <-> v1 LIMIT 2;
  // EXPLAIN (o) SELECT v2, v1 FROM t1 ORDER BY ARRAY [1.0, 1.0, 1.0] <-> v1 LIMIT 2;
  base_vector = std::dynamic_pointer_cast<const ArrayExpression>(distance_expr->GetChildAt(0));
  column_vector = std::dynamic_pointer_cast<const ColumnValueExpression>(distance_expr->GetChildAt(1));
  if (base_vector == nullptr || column_vector == nullptr) {
    base_vector = std::dynamic_pointer_cast<const ArrayExpression>(distance_expr->GetChildAt(1));
    column_vector = std::dynamic_pointer_cast<const ColumnValueExpression>(distance_expr->GetChildAt(0));
  }
  if (base_vector == nullptr || column_vector == nullptr) {
    return optimized_plan;
  }

  // Case 1: TopN directly followed by SeqScan
  // TopN { n=2, order_bys=[("Default", "l2_dist([1.000000,1.000000,1.000000], #0.0)")]}
  //   SeqScan { table=t1 }
  if (auto *seqscan = dynamic_cast<const SeqScanPlanNode *>(sort_node->GetChildPlan().get()); seqscan != nullptr) {
    auto index_info =
        MatchVectorIndex(catalog_, seqscan->table_oid_, column_vector->GetColIdx(), vty, vector_index_match_method_);
    if (index_info == nullptr) {
      return optimized_plan;
    }
    return std::make_shared<VectorIndexScanPlanNode>(seqscan->output_schema_, seqscan->table_oid_, seqscan->table_name_,
                                                     index_info->index_oid_, index_info->name_, base_vector, limit_n);
  }

  // Case 2: TopN followed by Projection
  // TopN { n=2, order_bys=[("Default", "l2_dist([1.000000,1.000000,1.000000], #0.0)")]}
  //   Projection { exprs=["#0.0", "l2_dist([1.000000,1.000000,1.000000], #0.0)"] }
  //     SeqScan { table=t1 }
  // Case 3: TopN followed by Projection with column reference shuffled
  // TopN { n=2, order_bys=[("Default", "l2_dist([1.000000,1.000000,1.000000], #0.1)")]}
  //   Projection { exprs=["#0.1", "#0.0"] }
  //     SeqScan { table=t1 }
  if (auto *projection_node = dynamic_cast<const ProjectionPlanNode *>(sort_node->GetChildPlan().get());
      projection_node != nullptr) {
    if (auto *seqscan = dynamic_cast<const SeqScanPlanNode *>(projection_node->GetChildPlan().get());
        seqscan != nullptr) {
      if (auto *projection_expr = dynamic_cast<const ColumnValueExpression *>(
              projection_node->expressions_[column_vector->GetColIdx()].get());
          projection_expr != nullptr) {
        auto index_info = MatchVectorIndex(catalog_, seqscan->table_oid_, projection_expr->GetColIdx(), vty,
                                           vector_index_match_method_);
        if (index_info == nullptr) {
          return optimized_plan;
        }
        auto index_scan = std::make_shared<VectorIndexScanPlanNode>(seqscan->output_schema_, seqscan->table_oid_,
                                                                    seqscan->table_name_, index_info->index_oid_,
                                                                    index_info->name_, base_vector, limit_n);
        return std::make_shared<ProjectionPlanNode>(projection_node->output_schema_, projection_node->expressions_,
                                                    index_scan);
      }
    }
  }

  return optimized_plan;
}

}  // namespace bustub
