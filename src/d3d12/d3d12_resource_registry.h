#pragma once

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace remix {

  // Registers buffer resources by their GPU virtual address. Upload
  // heaps carry much of the dynamic geometry, so those buffers can be read back
  // without changing the command stream.
  void registerResource(
      ID3D12Resource* resource,
      const D3D12_HEAP_PROPERTIES* heapProperties,
      const D3D12_RESOURCE_DESC* description);

  bool readGpuBuffer(
      D3D12_GPU_VIRTUAL_ADDRESS address,
      size_t size,
      std::vector<uint8_t>& output);

  // Mirrors a D3D12 buffer copy into the registry. This is required for
  // default-heap resources, which cannot be CPU-mapped after the game uploads data
  // from an upload heap.
  void recordCopyBufferRegion(
      ID3D12Resource* destination,
      UINT64 destinationOffset,
      ID3D12Resource* source,
      UINT64 sourceOffset,
      UINT64 numBytes);
  void recordResourceBarrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
      D3D12_RESOURCE_STATES after, UINT subresource);

}
