#pragma once

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace remix {

  // Remembers a GPU virtual address the vertex shader reads its transform record
  // from, so a copy into device memory can be mirrored for that address only.
  void noteRecordAddress(D3D12_GPU_VIRTUAL_ADDRESS address);

  // Reads a record that was mirrored from a copy into device memory.
  bool readRecordShadow(
      D3D12_GPU_VIRTUAL_ADDRESS address,
      size_t size,
      std::vector<uint8_t>& output);

  // Mirrors the record-sized part of a device-to-device copy, which is the only
  // way a record that lives in a default heap becomes readable on the host.
  void recordCopyBufferRegion(
      ID3D12Resource* destination,
      UINT64 destinationOffset,
      ID3D12Resource* source,
      UINT64 sourceOffset,
      UINT64 numBytes);

  // Registers buffer resources by their GPU virtual address. Upload
  // heaps carry much of the dynamic geometry, so those buffers can be read back
  // without changing the command stream.
  void registerResource(
      ID3D12Resource* resource,
      const D3D12_HEAP_PROPERTIES* heapProperties,
      const D3D12_RESOURCE_DESC* description);

  // Mirrors a copied region into host memory so data the game uploaded into a
  // default heap stays readable. Only small, record-sized destinations are
  // mirrored and the total is capped.
  bool readGpuBuffer(
      D3D12_GPU_VIRTUAL_ADDRESS address,
      size_t size,
      std::vector<uint8_t>& output);

  // Reports why a read of a GPU address fails, which separates "the buffer was
  // never registered" from "the buffer lives in a default heap".
  // Returns 0 when no registered buffer covers the address.
  // Returns 1 when the buffer was found and the host mapping succeeded.
  // Returns 2 when the buffer was found but the host mapping failed.
  int probeGpuBuffer(
      D3D12_GPU_VIRTUAL_ADDRESS address,
      size_t size,
      uint32_t& heapType,
      UINT64& bufferSize);

  // Reports how the registered buffers cover an address: how many were
  // registered, the nearest one at or below the address, and whether its range
  // contains the address. Separates a registration gap from a lookup miss.
  struct RegistryCoverage {
    size_t resourceCount = 0;
    D3D12_GPU_VIRTUAL_ADDRESS nearestBase = 0;
    UINT64 nearestSize = 0;
    UINT64 offset = 0;
    bool covered = false;
  };

  RegistryCoverage describeRegistry(D3D12_GPU_VIRTUAL_ADDRESS address, size_t size);

  // Returns the registered buffer at a given position, so a scan can walk every
  // buffer without already knowing an address.
  bool getRegisteredBuffer(
      size_t index,
      D3D12_GPU_VIRTUAL_ADDRESS& base,
      UINT64& size,
      uint32_t& heapType);

}
