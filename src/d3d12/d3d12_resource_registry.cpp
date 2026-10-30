#include "d3d12_resource_registry.h"

#include "d3d12_device_proxy.h"

#include <windows.h>

#include <algorithm>
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
      std::vector<uint8_t> shadow;
      bool shadowValid = false;
    };

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

    log("D3D12 resource registry: resource=%p gpu=0x%llx size=%llu heapType=%u map=%s",
      resource,
      static_cast<unsigned long long>(gpuAddress),
      static_cast<unsigned long long>(description->Width),
      static_cast<unsigned>(heapProperties->Type),
      heapProperties->Type == D3D12_HEAP_TYPE_UPLOAD
       || heapProperties->Type == D3D12_HEAP_TYPE_READBACK ? "likely" : "no");
  }

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
      if (found->shadowValid && offset <= found->shadow.size()
       && size <= found->shadow.size() - offset) {
        output.resize(size);
        std::memcpy(output.data(), found->shadow.data() + offset, size);
        return true;
      }
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

  void recordCopyBufferRegion(
      ID3D12Resource* destination,
      UINT64 destinationOffset,
      ID3D12Resource* source,
      UINT64 sourceOffset,
      UINT64 numBytes) {
    if (!destination || !source || !numBytes || numBytes > static_cast<UINT64>(SIZE_MAX))
      return;

    const auto destinationAddress = destination->GetGPUVirtualAddress();
    const auto sourceAddress = source->GetGPUVirtualAddress();
    if (!destinationAddress || !sourceAddress)
      return;

    std::vector<uint8_t> bytes;
    if (!readGpuBuffer(sourceAddress + sourceOffset, static_cast<size_t>(numBytes), bytes)) {
      log("D3D12 resource registry: CopyBufferRegion source unavailable src=%p offset=%llu bytes=%llu",
        source,
        static_cast<unsigned long long>(sourceOffset),
        static_cast<unsigned long long>(numBytes));
      return;
    }

    std::lock_guard lock(g_resourceMutex);
    ResourceEntry* found = findResourceLocked(destinationAddress + destinationOffset, bytes.size());
    if (!found)
      return;
    const UINT64 offset = destinationAddress + destinationOffset - found->gpuAddress;
    // Keep the shadow bounded. Large render targets and staging allocations
    // are not useful to the geometry/material mapper and should remain lazy.
    constexpr UINT64 kMaxShadowBytes = 64ull * 1024ull * 1024ull;
    if (found->size > kMaxShadowBytes)
      return;
    if (found->shadow.size() != found->size)
      found->shadow.resize(static_cast<size_t>(found->size));
    std::memcpy(found->shadow.data() + offset, bytes.data(), bytes.size());
    found->shadowValid = true;
  }

  void recordResourceBarrier(
      ID3D12Resource* resource,
      D3D12_RESOURCE_STATES before,
      D3D12_RESOURCE_STATES after,
      UINT subresource) {
    if (!resource)
      return;
    std::lock_guard lock(g_resourceMutex);
    const auto gpuAddress = resource->GetGPUVirtualAddress();
    if (!gpuAddress)
      return;
    auto* found = findResourceLocked(gpuAddress, 1);
    if (!found)
      return;
    found->state = after;
    log("D3D12 resource registry: barrier resource=%p subresource=%u before=0x%x after=0x%x",
      resource, subresource, static_cast<unsigned>(before), static_cast<unsigned>(after));
  }

}
