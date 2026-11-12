#include "d3d12_resource_registry.h"

#include "d3d12_device_proxy.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

namespace remix {

  namespace {

    struct ResourceEntry {
      ID3D12Resource* resource = nullptr;
      D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
      UINT64 size = 0;
      D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
      D3D12_RESOURCE_DESC description {};
      D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    };

    // The records the shader reads are 416 bytes and the game writes them with a
    // copy into a default heap, so only those addresses are mirrored and the
    // amount of host memory stays at a few kilobytes.
    constexpr size_t kRecordSize = 416;
    // Records are re-noted every frame and their addresses move, so the table has
    // to be deep enough that a copy arriving a few draws later still matches.
    constexpr size_t kMaxRecords = 64;
    struct RecordSlot {
      D3D12_GPU_VIRTUAL_ADDRESS address = 0;
      std::array<uint8_t, kRecordSize> bytes {};
      bool valid = false;
    };
    RecordSlot g_records[kMaxRecords];
    size_t g_recordCount = 0;

    std::mutex g_resourceMutex;
    std::map<D3D12_GPU_VIRTUAL_ADDRESS, ResourceEntry> g_resources;

    ResourceEntry* findResourceLocked(D3D12_GPU_VIRTUAL_ADDRESS address, size_t size) {
      auto it = g_resources.upper_bound(address);
      if (it == g_resources.begin())
        return nullptr;
      --it;

      const ResourceEntry& entry = it->second;
      if (address < entry.gpuAddress)
        return nullptr;
      const UINT64 offset = address - entry.gpuAddress;
      if (offset > entry.size || size > entry.size - offset)
        return nullptr;
      return &it->second;
    }

  }

  void registerResource(
      ID3D12Resource* resource,
      const D3D12_HEAP_PROPERTIES* heapProperties,
      const D3D12_RESOURCE_DESC* description) {
    if (!resource || !heapProperties || !description
     || description->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
     || description->Width == 0)
      return;

    const D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = resource->GetGPUVirtualAddress();
    if (!gpuAddress)
      return;

    resource->AddRef();
    ResourceEntry entry;
    entry.resource = resource;
    entry.gpuAddress = gpuAddress;
    entry.size = description->Width;
    entry.heapType = heapProperties->Type;
    entry.description = *description;

    std::lock_guard lock(g_resourceMutex);
    auto old = g_resources.find(gpuAddress);
    if (old != g_resources.end()) {
      old->second.resource->Release();
      old->second = entry;
    } else {
      g_resources.emplace(gpuAddress, entry);
    }
  }

  static float g_searchOrigin[3] = { 0.0f, 0.0f, 0.0f };

  bool readGpuBuffer(
      D3D12_GPU_VIRTUAL_ADDRESS address,
      size_t size,
      std::vector<uint8_t>& output) {
    if (!address || size == 0 || size > static_cast<size_t>(UINT_MAX))

      return false;

    ResourceEntry entry;
    UINT64 offset = 0;
    {
      std::lock_guard lock(g_resourceMutex);
      ResourceEntry* found = findResourceLocked(address, size);
      if (!found)
        return false;
      offset = address - found->gpuAddress;
      entry = *found;
      entry.resource->AddRef();
    }

    void* mapped = nullptr;
    D3D12_RANGE readRange {
      static_cast<SIZE_T>(offset),
      static_cast<SIZE_T>(offset + size),
    };
    HRESULT result = entry.resource->Map(0, &readRange, &mapped);
    if (SUCCEEDED(result) && mapped) {
      output.resize(size);
      std::memcpy(output.data(), static_cast<const uint8_t*>(mapped) + offset, size);
      D3D12_RANGE writtenRange { 0, 0 };
      entry.resource->Unmap(0, &writtenRange);
    }
    entry.resource->Release();
    return SUCCEEDED(result) && mapped != nullptr;
  }

  int probeGpuBuffer(
      D3D12_GPU_VIRTUAL_ADDRESS address,
      size_t size,
      uint32_t& heapType,
      UINT64& bufferSize) {
    heapType = 0;
    bufferSize = 0;
    if (!address || size == 0)
      return 0;

    ResourceEntry entry;
    UINT64 offset = 0;
    {
      std::lock_guard lock(g_resourceMutex);
      ResourceEntry* found = findResourceLocked(address, size);
      if (!found)
        return 0;
      offset = address - found->gpuAddress;
      entry = *found;
      entry.resource->AddRef();
    }

    heapType = static_cast<uint32_t>(entry.heapType);
    bufferSize = entry.size;
    void* mapped = nullptr;
    D3D12_RANGE readRange {
      static_cast<SIZE_T>(offset),
      static_cast<SIZE_T>(offset + size),
    };
    const HRESULT result = entry.resource->Map(0, &readRange, &mapped);
    if (SUCCEEDED(result) && mapped) {
      D3D12_RANGE writtenRange { 0, 0 };
      entry.resource->Unmap(0, &writtenRange);
    }
    entry.resource->Release();
    return (SUCCEEDED(result) && mapped) ? 1 : 2;
  }

  bool getRegisteredBuffer(
      size_t index,
      D3D12_GPU_VIRTUAL_ADDRESS& base,
      UINT64& size,
      uint32_t& heapType) {
    std::lock_guard lock(g_resourceMutex);
    if (index >= g_resources.size())
      return false;
    size_t current = 0;
    for (auto it = g_resources.begin(); it != g_resources.end(); ++it) {
      if (current++ != index)
        continue;
      base = it->second.gpuAddress;
      size = it->second.size;
      heapType = static_cast<uint32_t>(it->second.heapType);
      return true;
    }
    return false;
  }

  RegistryCoverage describeRegistry(D3D12_GPU_VIRTUAL_ADDRESS address, size_t size) {
    RegistryCoverage coverage;
    std::lock_guard lock(g_resourceMutex);
    coverage.resourceCount = g_resources.size();
    if (g_resources.empty())
      return coverage;
    auto it = g_resources.upper_bound(address);
    if (it == g_resources.begin())
      return coverage;
    --it;
    coverage.nearestBase = it->second.gpuAddress;
    coverage.nearestSize = it->second.size;
    if (address < it->second.gpuAddress)
      return coverage;
    coverage.offset = address - it->second.gpuAddress;
    coverage.covered = coverage.offset <= it->second.size
      && size <= it->second.size - coverage.offset;
    return coverage;
  }

  void noteRecordAddress(D3D12_GPU_VIRTUAL_ADDRESS address) {
    if (!address)
      return;
    std::lock_guard lock(g_resourceMutex);
    // The ring wraps, so the scan must stop at the slots actually filled rather
    // than at the call count.
    const size_t used = g_recordCount < kMaxRecords ? g_recordCount : kMaxRecords;
    for (size_t i = 0; i < used; ++i) {
      if (g_records[i].address == address)
        return;
    }
    const size_t slot = g_recordCount % kMaxRecords;
    ++g_recordCount;
    g_records[slot].address = address;
    g_records[slot].valid = false;
  }

  bool readRecordShadow(
      D3D12_GPU_VIRTUAL_ADDRESS address,
      size_t size,
      std::vector<uint8_t>& output) {
    if (!address || size > kRecordSize)
      return false;
    std::lock_guard lock(g_resourceMutex);
    for (size_t i = 0; i < kMaxRecords; ++i) {
      const RecordSlot& record = g_records[i];
      if (record.address != address || !record.valid)
        continue;
      output.assign(record.bytes.begin(), record.bytes.begin() + size);
      return true;
    }
    return false;
  }

  void recordCopyBufferRegion(
      ID3D12Resource* destination,
      UINT64 destinationOffset,
      ID3D12Resource* source,
      UINT64 sourceOffset,
      UINT64 numBytes) {
    if (!destination || !source || !numBytes)
      return;
    const D3D12_GPU_VIRTUAL_ADDRESS destinationBase = destination->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS sourceBase = source->GetGPUVirtualAddress();
    if (!destinationBase || !sourceBase)
      return;

    // Establishes whether the game copies buffers at all, which the record mirror
    // depends on. The first calls are reported in full, then only the running total
    // at each power of ten, so volume stays readable without hiding activity.
    static uint64_t copyCalls = 0;
    static uint64_t nextMilestone = 10;
    ++copyCalls;
    if (copyCalls <= 8) {
      log("ACZ copy: #%llu dst=0x%llx+%llu src=0x%llx+%llu bytes=%llu",
        static_cast<unsigned long long>(copyCalls),
        static_cast<unsigned long long>(destinationBase),
        static_cast<unsigned long long>(destinationOffset),
        static_cast<unsigned long long>(sourceBase),
        static_cast<unsigned long long>(sourceOffset),
        static_cast<unsigned long long>(numBytes));
    } else if (copyCalls == nextMilestone) {
      log("ACZ copy: total=%llu", static_cast<unsigned long long>(copyCalls));
      nextMilestone *= 10;
    }

    const UINT64 destinationFirst = destinationBase + destinationOffset;
    const UINT64 destinationLast = destinationFirst + numBytes;
    D3D12_GPU_VIRTUAL_ADDRESS matched[kMaxRecords] = {};
    size_t matchedCount = 0;
    {
      std::lock_guard lock(g_resourceMutex);
      for (size_t i = 0; i < kMaxRecords && matchedCount < kMaxRecords; ++i) {
        const RecordSlot& record = g_records[i];
        if (!record.address || record.address < destinationFirst)
          continue;
        if (record.address + kRecordSize > destinationLast)
          continue;
        matched[matchedCount++] = record.address;
      }
    }

    // One line per matched record reports whether the mirroring copy could be read
    // back, which separates "no copy ever lands here" from "the copy source is
    // itself unreadable". Both cases otherwise look identical from the probe.
    static D3D12_GPU_VIRTUAL_ADDRESS reportedMirror = 0;

    for (size_t i = 0; i < matchedCount; ++i) {
      const D3D12_GPU_VIRTUAL_ADDRESS sourceAddress =
        sourceBase + sourceOffset + (matched[i] - destinationFirst);
      std::vector<uint8_t> bytes;
      const bool mirrored =
        readGpuBuffer(sourceAddress, kRecordSize, bytes) && bytes.size() >= kRecordSize;
      if (matched[i] != reportedMirror) {
        reportedMirror = matched[i];
        log("ACZ mirror: dst=0x%llx src=0x%llx copyBytes=%llu srcRead=%d",
          static_cast<unsigned long long>(matched[i]),
          static_cast<unsigned long long>(sourceAddress),
          static_cast<unsigned long long>(numBytes), mirrored ? 1 : 0);
      }
      if (!mirrored)
        continue;
      std::lock_guard lock(g_resourceMutex);
      for (size_t slot = 0; slot < kMaxRecords; ++slot) {
        if (g_records[slot].address != matched[i])
          continue;
        std::memcpy(g_records[slot].bytes.data(), bytes.data(), kRecordSize);
        g_records[slot].valid = true;
      }
    }
  }
}
