//
// Created by jgleeson on 2019-12-17.
//

#include <sys/types.h>

#include <vector>
#include <memory>
#include <assert.h>

namespace rlscope {

class ILRUEntry {
public:
  ILRUEntry* _next;
  ILRUEntry* _prev;
  bool _in_memory;
  ILRUEntry() :
      _next(nullptr),
      _prev(nullptr),
      _in_memory(false) {
  }

  virtual void PageIn() = 0;
  virtual void Evict() = 0;
  virtual size_t SizeBytes() const = 0;

  inline bool InMemory() const {
    return _in_memory;
  }

};

class LRUCache {
public:
  // std::vector<std::shared_ptr<ILRUEntry>> _cache;
  ILRUEntry* _cache;
  // ILRUEntry* _cache_front;
  ILRUEntry* _cache_back;

  size_t _capacity_bytes;
  size_t _cached_bytes;

  LRUCache(size_t capacity_bytes) :
      _cache(nullptr),
      _cache_back(nullptr),
      _capacity_bytes(capacity_bytes),
      _cached_bytes(0)
  {

  }

  void _AddFront(ILRUEntry* entry) {
    if (!_cache_back) {
      _cache_back = entry;
    }

    entry->_next = _cache;
    entry->_prev = nullptr;
    if (entry->_next) {
      entry->_next->_prev = entry;
    }

    _cache = entry;
  }

  void _AddBack(ILRUEntry* entry) {
    if (!_cache) {
      _cache = entry;
    }

    entry->_next = nullptr;
    entry->_prev = _cache_back;
    if (entry->_prev) {
      entry->_prev->_next = entry;
    }

    _cache_back = entry;
  }

  void _Remove(ILRUEntry* entry) {
    if (entry == _cache_back) {
      _cache_back = entry->_prev;
    }
    if (entry == _cache) {
      _cache = entry->_next;
    }

    if (entry->_prev) {
      entry->_prev->_next = entry->_next;
    }
    if (entry->_next) {
      entry->_next->_prev = entry->_prev;
    }
    entry->_prev = nullptr;
    entry->_next = nullptr;
  }

  void _MoveToBack(ILRUEntry* entry) {
    _Remove(entry);
    _AddBack(entry);
  }

  void _MoveToFront(ILRUEntry* entry) {
    _Remove(entry);
    _AddFront(entry);
  }

  void Evict(ILRUEntry* entry) {
    entry->Evict();
    _MoveToBack(entry);
  }

  template <typename LRUEntryKlass, typename ...Args>
  void Register(Args && ...args)
  {
    // std::shared_ptr<LRUEntryKlass> entry = new LRUEntryKlass(std::forward<Args>(args)...);
    ILRUEntry* entry = new LRUEntryKlass(std::forward<Args>(args)...);
    _AddBack(entry);
  }

  void NotifyUse(ILRUEntry* entry) {
    _MoveToFront(entry);
    if (!entry->InMemory()) {
      entry->PageIn();
      auto entry_bytes = entry->SizeBytes();
      // An individual entry must fit into the cache,
      // otherwise we will get into a page-in/evict loop.
      assert(entry_bytes <= _capacity_bytes);
      _cached_bytes += entry_bytes;
      _MaybePageOut();
    }
  }

  void _MaybePageOut() {

    // Keep the first <= _capacity_bytes (e.g. 1GB)
    size_t keep_bytes = 0;
    ILRUEntry* cur = _cache;
    while (cur && keep_bytes + cur->SizeBytes() < _capacity_bytes && cur->InMemory()) {
      keep_bytes += cur->SizeBytes();
      cur = cur->_next;
    }

    // Evict anything that remains in-memory that would put us above _capacity_bytes.
    while (cur && cur->InMemory()) {
      // keep_bytes + cur->SizeBytes() >= _capacity_bytes
      //   Evict any remaining entries.
      auto evicted_bytes = cur->SizeBytes();
      // An individual entry must fit into the cache,
      // otherwise we will get into a page-in/evict loop.
      assert(evicted_bytes <= _capacity_bytes);
      cur->Evict();
      _cached_bytes -= evicted_bytes;
      cur = cur->_next;
    }

    // Sanity: anything at the tail must NOT be in-memory.
    while (cur) {
      assert(!cur->InMemory());
      cur = cur->_next;
    }

  }

};

}
