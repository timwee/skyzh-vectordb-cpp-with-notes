//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// insert_executor.cpp
//
// Identification: src/execution/insert_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "common/macros.h"
#include "execution/executors/insert_executor.h"
#include "storage/index/vector_index.h"
#include "storage/table/tuple.h"
#include "type/type_id.h"
#include "type/value_factory.h"

namespace bustub {

InsertExecutor::InsertExecutor(ExecutorContext *exec_ctx, const InsertPlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan), child_executor_(std::move(child_executor)) {
  auto table = exec_ctx->GetCatalog()->GetTable(plan->table_oid_);
  table_heap_ = table->table_.get();
  for (const auto &index_info : exec_ctx->GetCatalog()->GetTableIndexes(table->name_)) {
    auto index = dynamic_cast<VectorIndex *>(index_info->index_.get());
    BUSTUB_ASSERT(index != nullptr, "only vector index is supported");
    indexes_.push_back(index);
  }
}

void InsertExecutor::Init() {
  child_executor_->Init();
  Tuple tuple;
  RID rid;
  while (child_executor_->Next(&tuple, &rid)) {
    auto inserted_rid = table_heap_->InsertTuple(TupleMeta{0, false}, tuple);
    for (auto *index : indexes_) {
      BUSTUB_ASSERT(index->GetKeyAttrs().size() == 1, "only support vector index with one vector");
      auto index_key = index->GetKeyAttrs()[0];
      auto val = tuple.GetValue(&child_executor_->GetOutputSchema(), index_key);
      BUSTUB_ASSERT(val.GetTypeId() == TypeId::VECTOR, "only support vector type for vector index");
      auto vec = val.GetVector();
      index->InsertVectorEntry(vec, *inserted_rid);
    }
  }
  emitted_ = false;
}

auto InsertExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  if (emitted_) {
    return false;
  }
  emitted_ = true;
  *tuple = {{ValueFactory::GetIntegerValue(0)}, &GetOutputSchema()};
  return true;
}

}  // namespace bustub
