#include <isl/interface/cpp.h>

//
// High-level interface.
//

// Static control part.
// Names of fields are self-explanatory.
// Access relations are tagged, that is live in the spaces
//   [S1[...] -> __ref_ID[]] -> arrayID[...]
// where __ref_ID are unique identifiers.
// 
// External users have statement-specific information stored either
// (a) separately in a map<isl::id, CustomStmtClass> or
// (b) as user-pointers of statement ids using C interface (discouraged).
struct Scop {
  isl::set context;
  isl::union_set domain;
  isl::union_map mayReads;
  isl::union_map mayWrites;
  isl::union_map mustWrites;
  isl::schedule schedule;
};

// Map "scop" to a grid of GPU thread blocks described by "grid" and "block"
// starting from schedule node "node" and using the end-to-end mapping
// strategy.  When tiling is performed, use sizes provided in "tiles".
// The return value indicates whether the mapping was performed.
// Mapping strategy is allowed to decrease the block and grid size to avoid
// launching empty blocks or threads, in which case the values in "block" and
// "grid" are updated with the new sizes.
//
// The underlying strategy may change at any time as long as the same types of
// trees can be mapped.
//
// Node is not necessarily a band node.
//
// Trailing ones in "block" and "grid" may be interpreted as not mapping to the
// corresponding thread or block dimensions.  All values are strictly positive.
bool mapToGPU(
  Scop& scop,
  isl::schedule_node node,
  const std::vector<long>& tiles,
  std::array<long, 3>& block,
  std::array<long, 3>& grid);


