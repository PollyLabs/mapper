#include <isl/interface/cpp.h>

//
// Low-level interface.
//

class Scop;

// Syntactic identifiers for threads.
enum class Thread {
  x = 0,
  y,
  z
};

// Syntactic identifiers for blocks.
enum class Block {
  x = 0,
  y,
  z
};

// Mark the schedule node "node" as the first node in a GPU kernel by inserting
// a mark node with a unique identifier, followed by an eventual guard node and
// a context node (referred to as kernel context below).  The kernel is
// expected to be launched on a grid of thread blocks described by "grid" and
// "block" sizes.  The grid configuration is stored in the kernel context.  It
// will be used when mapping band members to blocks/threads.
isl::schedule_node_mark initKernel(Scop& scop, isl::schedule_node node,
  const std::array<long, 3>& block, const std::array<long, 3>& grid);

// Check whether the schedule node "node" is inside a kernel.
bool isInKernel(isl::schedule_node node);

// Return the id of the kernel to which "node" belongs.
isl::id kernelId(isl::schedule_node node);

// Return the grid of the kernel to which "node" belongs.
std::array<long, 3> kernelGrid(isl::schedule_node node);

// Return the block size of the kernel to which "node" belongs.
std::array<long, 3> kernelBlock(isl::schedule_node node);

// Check whether it is valid to map "dim"-th dimension of the band node "node"
// to thread "t" (block "b") in the context of schedule to which "node"
// belongs.  For example, if children or ancestors of "node" already contain a
// mapping to thread "t" (block "b"), if the dimension is parallel, etc.
bool canMapBandDim(isl::schedule_node_band node, int dim, Thread t);
bool canMapBandDim(isl::schedule_node_band node, int dim, Block b);

// Map "dim"-th dimension of the band node "node" to thread (block) dimension
// "t" ("b").  This mapping is assumed to be valid, in particular that the band
// is situated inside a kernel.  The number of threads (blocks) to map to is
// taken from the kernel context. Return the updated band node, its parents may
// have changed due to mapping.
//
// Note: while it may be possible to map subtrees to a different number of
// blocks (threads), it is often dangerous in practice.  For example, inserting
// a __syncthreads call under a condition involving thread identifiers leads to
// undefined behavior.  Therefore, we prefer to have a fixed number of threads
// (blocks) per kernel.  If it is strictly necessary to use less threads, the
// caller can first strip-mine the band so as to have the required number of
// threads and then map the inner band.
isl::schedule_node_band mapBandDim(isl::schedule_node_band node, int dim,
    Thread t);
isl::schedule_node_band mapBandDim(isl::schedule_node_band node, int dim,
    Block b);

// Map the subtree rooted at "node" to a single thread (block) along the thread
// (block) dimension "t" ("b").  Return the updated node, its parents may have
// changed due to mapping.
isl::schedule_node mapFixed(isl::schedule_node node, Thread t);
isl::schedule_node mapFixed(isl::schedule_node node, Block b);

// Get the mapping active at the subtree rooted at "node".  In particular,
// intersect all filters on the path from "node" to the root of schedule tree.
// Some of these filters perform the mapping and the corresponding constraints
// will appear in the resulting union set.
// Ignores any mapping below "node".
isl::union_set mapping(isl::schedule_node node);

// Check if the subtree rooted at "node" is mapped to thread (block) dimension
// "t" ("b").  Only the mapping above "node" is taken into account.
bool isMapped(isl::schedule_node node, Thread t);
bool isMapped(isl::schedule_node node, Block b);

// Tile the band node "band" with sizes "tiles" and return the updated node.
isl::schedule_node_band tile(isl::schedule_node_band band, const std::vector<long>& tiles);
isl::schedule_node_band unroll(isl::schedule_node_band band, long limit);

// Finalize the mapping of a subtree rooted at "node" that represents a kernel.
// In particular, ensure that all branches are mapped to the same number of
// thread and block dimensions and that the appropriate synchronizations are
// inserted.  Optionally, change the kernel-level context to only include the
// actually used threads (blocks).
isl::schedule_node finalizeKernel(isl::schedule_node_mark node);

// An identifier of a group of references that must be promoted together
// for validity reasons.  All references are to the array identifiable by
// "arrayId".  Individual reference IDs are stored in "refIds".
// The promotion is be scoped under a mark node identified by "markId", that
// is, the extension subtree for copying to/from the promoted memory space is
// inserted below the mark node.
//
// It is impossible to store an isl::schedule_node because a new schedule may
// be created on each operation due to CoW.
// It is impossible to store the schedule depth if we want the promotion to be
// scoped at a sequence node, that is keep the promoted elements in memory for
// a group of sibling subtrees.
// Note: we may need to artificially separate a sequence (set) node into a
// nested structure of sequence (set) nodes to restrict the promotion scope.
struct ScopedReferenceGroup {
  isl::id markId;
  isl::id arrayId;
  isl::id_list refIds;

  // Extra functions can be provided here, e.g. the size of the footprint in
  // promoted memory space.
};

// Insert a mark node with a unique id that identifies a potential promotion
// scope.  The insertion takes place immediately above "node".  Return the
// inserted mark node.
isl::schedule_node_mark definePromotionScope(isl::schedule_node node);

// For all promotion scopes defined in "scop", return the list of
// ScopedReferenceGroup for which promotion to shared (private) memory is legal
// given the current mapping and that the copies are inserted below the scoping
// mark node.
std::unordered_set<ScopedReferenceGroup> promotableToShared(const Scop& scop);
std::unordered_set<ScopedReferenceGroup> promotableToPrivate(const Scop& scop);

// Promote the scoped reference group "group" to shared (private) memory in its
// scope by modifying the schedule of "scop".  May additionally keep track of
// required declarations in each kernel inside "scop".
void promoteToShared(Scop& scop, ScopedReferenceGroup group);
void promoteToPrivate(Scop& scop, ScopedReferenceGroup group);
