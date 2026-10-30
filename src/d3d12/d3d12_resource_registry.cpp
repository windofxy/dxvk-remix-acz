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

}
