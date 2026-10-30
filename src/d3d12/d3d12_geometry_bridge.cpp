#include "d3d12_geometry_bridge.h"

#include "d3d12_device_proxy.h"
#include "d3d12_resource_registry.h"
#include "d3d12_runtime_adapter.h"

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace remix {

  namespace {

    constexpr uint32_t kCaptureMagic = 0x58444D52; // "RMDX"
    constexpr uint32_t kCaptureVersion = 1;
    constexpr size_t kMaxCaptureBytes = 4 * 1024 * 1024;
    constexpr uint32_t kMaxCaptures = 512;

    #pragma pack(push, 1)
    struct CaptureHeader {
      uint32_t magic;
      uint32_t version;
      uint64_t vertexGpuAddress;
      uint32_t vertexBufferSize;
      uint32_t vertexStride;
      uint64_t indexGpuAddress;
      uint32_t indexBufferSize;
      uint32_t indexFormat;
      uint32_t indexCount;
      uint32_t instanceCount;
      uint32_t startIndexLocation;
      int32_t baseVertexLocation;
      uint32_t startInstanceLocation;
      uint32_t topology;
      uint64_t pipelineState;
      uint32_t vertexDataSize;
      uint32_t indexDataSize;
    };
    #pragma pack(pop)

    struct CaptureKey {
      uint64_t vertexAddress;
      uint64_t indexAddress;
      uint32_t vertexSize;
      uint32_t indexSize;
      uint32_t stride;
      uint32_t indexCount;
      uint32_t topology;
      uint32_t startIndexLocation;
      int32_t baseVertexLocation;
      uint32_t startInstanceLocation;

      bool operator==(const CaptureKey& other) const {
        return vertexAddress == other.vertexAddress
          && indexAddress == other.indexAddress
          && vertexSize == other.vertexSize
          && indexSize == other.indexSize
          && stride == other.stride
          && indexCount == other.indexCount
          && topology == other.topology
          && startIndexLocation == other.startIndexLocation
          && baseVertexLocation == other.baseVertexLocation
          && startInstanceLocation == other.startInstanceLocation;
      }
    };

    struct CaptureKeyHash {
      size_t operator()(const CaptureKey& key) const {
        uint64_t value = key.vertexAddress;
        value ^= key.indexAddress + 0x9E3779B97F4A7C15ull + (value << 6) + (value >> 2);
        value ^= static_cast<uint64_t>(key.vertexSize) << 17;
        value ^= static_cast<uint64_t>(key.indexSize) << 29;
        value ^= static_cast<uint64_t>(key.stride) << 37;
        value ^= static_cast<uint64_t>(key.indexCount) << 43;
        value ^= static_cast<uint64_t>(key.topology) << 53;
        value ^= static_cast<uint64_t>(key.startIndexLocation) << 7;
        value ^= static_cast<uint64_t>(static_cast<uint32_t>(key.baseVertexLocation)) << 23;
        value ^= static_cast<uint64_t>(key.startInstanceLocation) << 31;
        return static_cast<size_t>(value ^ (value >> 32));
      }
    };

    std::mutex g_captureMutex;
    std::unordered_set<CaptureKey, CaptureKeyHash> g_captured;
    uint32_t g_captureNumber = 0;

    std::filesystem::path captureDirectory() {
      wchar_t modulePath[MAX_PATH] = {};
      DWORD length = GetModuleFileNameW(nullptr, modulePath, ARRAYSIZE(modulePath));
      if (!length || length >= ARRAYSIZE(modulePath))
        return {};
      std::filesystem::path directory(modulePath);
      directory = directory.parent_path() / L"remix-d3d12-geometry";
      std::error_code error;
      std::filesystem::create_directories(directory, error);
      return directory;
    }

    uint32_t indexStride(DXGI_FORMAT format) {
      switch (format) {
      case DXGI_FORMAT_R16_UINT: return 2;
      case DXGI_FORMAT_R32_UINT: return 4;
      default: return 0;
      }
    }

    uint32_t readIndex(const std::vector<uint8_t>& data, uint32_t stride, size_t index) {
      const size_t offset = index * stride;
      if (offset + stride > data.size())
        return 0;
      if (stride == 2)
        return *reinterpret_cast<const uint16_t*>(data.data() + offset);
      return *reinterpret_cast<const uint32_t*>(data.data() + offset);
    }

  }

  void captureIndexedGeometry(
      ID3D12GraphicsCommandList* commandList,
      const D3D12_VERTEX_BUFFER_VIEW* vertexBuffer,
      const D3D12_INDEX_BUFFER_VIEW* indexBuffer,
      D3D12_PRIMITIVE_TOPOLOGY topology,
      ID3D12PipelineState* pipelineState,
      UINT indexCount,
      UINT instanceCount,
      UINT startIndexLocation,
      INT baseVertexLocation,
      UINT startInstanceLocation) {
    if (!commandList || !vertexBuffer || !indexBuffer || !indexCount
     || !vertexBuffer->BufferLocation || !vertexBuffer->StrideInBytes
     || !indexBuffer->BufferLocation)
      return;

    const uint32_t stride = indexStride(indexBuffer->Format);
    if (!stride)
      return;

    LARGE_INTEGER captureStart {};
    QueryPerformanceCounter(&captureStart);

    CaptureKey key {
      vertexBuffer->BufferLocation,
      indexBuffer->BufferLocation,
      vertexBuffer->SizeInBytes,
      indexBuffer->SizeInBytes,
      vertexBuffer->StrideInBytes,
      indexCount,
      static_cast<uint32_t>(topology),
      startIndexLocation,
      baseVertexLocation,
      startInstanceLocation,
    };

    uint32_t captureNumber = 0;
    {
      std::lock_guard lock(g_captureMutex);
      if (g_captured.find(key) != g_captured.end())
        return;
      if (g_captureNumber >= kMaxCaptures)
        return;
      g_captured.emplace(key);
      captureNumber = ++g_captureNumber;
    }

    const size_t indexBytes = static_cast<size_t>(indexCount) * stride;
    std::vector<uint8_t> indexData;
    const bool indexMapped = readGpuBuffer(
      indexBuffer->BufferLocation + static_cast<UINT64>(startIndexLocation) * stride,
      indexBytes,
      indexData);

    uint32_t maxIndex = 0;
    bool hasReferencedIndex = false;
    if (indexMapped) {
      const uint32_t restartIndex = stride == 2 ? 0xFFFFu : 0xFFFFFFFFu;
      for (UINT i = 0; i < indexCount; ++i) {
        const uint32_t index = readIndex(indexData, stride, i);
        if (index == restartIndex)
          continue;
        hasReferencedIndex = true;
        maxIndex = std::max(maxIndex, index);
      }
    }

    uint64_t vertexCount = indexMapped && hasReferencedIndex
      ? static_cast<uint64_t>(maxIndex) + (baseVertexLocation >= 0 ? static_cast<uint64_t>(baseVertexLocation) + 1 : 1)
      : vertexBuffer->SizeInBytes / vertexBuffer->StrideInBytes;
    vertexCount = std::min<uint64_t>(vertexCount, vertexBuffer->SizeInBytes / vertexBuffer->StrideInBytes);
    size_t vertexBytes = static_cast<size_t>(vertexCount) * vertexBuffer->StrideInBytes;
    vertexBytes = std::min(vertexBytes, kMaxCaptureBytes);
    std::vector<uint8_t> vertexData;
    const bool vertexMapped = readGpuBuffer(vertexBuffer->BufferLocation, vertexBytes, vertexData);

    CaptureHeader header {
      kCaptureMagic,
      kCaptureVersion,
      vertexBuffer->BufferLocation,
      vertexBuffer->SizeInBytes,
      vertexBuffer->StrideInBytes,
      indexBuffer->BufferLocation,
      indexBuffer->SizeInBytes,
      static_cast<uint32_t>(indexBuffer->Format),
      indexCount,
      instanceCount,
      startIndexLocation,
      baseVertexLocation,
      startInstanceLocation,
      static_cast<uint32_t>(topology),
      reinterpret_cast<uint64_t>(pipelineState),
      vertexMapped ? static_cast<uint32_t>(vertexData.size()) : 0,
      indexMapped ? static_cast<uint32_t>(indexData.size()) : 0,
    };

    const std::filesystem::path directory = captureDirectory();
    if (!directory.empty()) {
      wchar_t filename[64] = {};
      swprintf_s(filename, L"batch_%06u.bin", captureNumber);
      const auto path = directory / filename;
      std::ofstream file(path, std::ios::binary);
      if (file) {
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        if (vertexMapped)
          file.write(reinterpret_cast<const char*>(vertexData.data()), vertexData.size());
        if (indexMapped)
          file.write(reinterpret_cast<const char*>(indexData.data()), indexData.size());
      }
    }

    log("D3D12 geometry bridge: batch=%u list=%p indices=%u instances=%u topology=%u vertex(gpu=0x%llx,size=%u,stride=%u,mapped=%u,bytes=%zu) index(gpu=0x%llx,size=%u,format=%u,mapped=%u,bytes=%zu) maxIndex=%u",
      captureNumber,
      commandList,
      indexCount,
      instanceCount,
      static_cast<unsigned>(topology),
      static_cast<unsigned long long>(vertexBuffer->BufferLocation),
      vertexBuffer->SizeInBytes,
      vertexBuffer->StrideInBytes,
      vertexMapped ? 1u : 0u,
      vertexData.size(),
      static_cast<unsigned long long>(indexBuffer->BufferLocation),
      indexBuffer->SizeInBytes,
      static_cast<unsigned>(indexBuffer->Format),
      indexMapped ? 1u : 0u,
      indexData.size(),
      maxIndex);
    LARGE_INTEGER captureEnd {};
    QueryPerformanceCounter(&captureEnd);
    recordD3D12GeometryCaptureDuration(
      static_cast<uint64_t>(captureEnd.QuadPart - captureStart.QuadPart));
  }

}
