#include "d3d12_runtime_adapter.h"

#include "acz_camera_hook.h"
#include "d3d12_command_list_trace.h"
#include "d3d12_device_proxy.h"
#include "d3d12_native_backend.h"
#include "d3d12_resource_registry.h"

#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace remix {

  namespace {
    std::once_flag g_initialized;
    std::atomic<uint64_t> g_deviceCount = 0;
    std::atomic<uint64_t> g_queueCount = 0;
    std::atomic<uint64_t> g_commandListCount = 0;
    std::atomic<uint64_t> g_drawCount = 0;
    std::atomic<uint64_t> g_submitCount = 0;
    std::atomic<uint64_t> g_presentCount = 0;
    std::mutex g_frameMutex;

    struct FrameLedger {
      uint64_t frame = 0;
      uint64_t draws = 0;
      uint64_t submits = 0;
      uint64_t pipelines = 0;
      uint64_t resources = 0;
      uint64_t indexedDraws = 0;
      uint64_t instancedDraws = 0;
      uint64_t fenceSignals = 0;
      uint64_t fenceWaits = 0;
      UINT backBufferIndex = 0;
      bool presentBegin = false;
    };

    FrameLedger g_frame;

    struct PipelineMetadata {
      uint64_t vertexShaderHash = 0;
      uint64_t pixelShaderHash = 0;
      D3D12NativePipelineInfo nativeInfo;
      UINT inputElementCount = 0;
      D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
      DXGI_FORMAT renderTargetFormat = DXGI_FORMAT_UNKNOWN;
      DXGI_FORMAT depthStencilFormat = DXGI_FORMAT_UNKNOWN;
    };

    std::unordered_map<ID3D12PipelineState*, PipelineMetadata> g_pipelines;

    struct ResourceMetadata {
      D3D12_RESOURCE_DESC description {};
      D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
      D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
    };

    std::unordered_map<ID3D12Resource*, ResourceMetadata> g_resourceMetadata;

    struct DescriptorHeapMetadata {
      D3D12_CPU_DESCRIPTOR_HANDLE cpuStart {};
      D3D12_GPU_DESCRIPTOR_HANDLE gpuStart {};
      UINT increment = 0;
      UINT count = 0;
      D3D12_DESCRIPTOR_HEAP_TYPE type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    };

    struct DescriptorMetadata {
      ID3D12Resource* resource = nullptr;
      D3D12_RESOURCE_DESC resourceDescription {};
      D3D12_GPU_VIRTUAL_ADDRESS resourceGpuAddress = 0;
      DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
      const char* viewType = nullptr;
      D3D12_GPU_VIRTUAL_ADDRESS bufferLocation = 0;
      UINT bufferSize = 0;
      std::array<uint8_t, 64> viewDescription {};
      UINT viewDescriptionSize = 0;
      bool sampler = false;
    };

    struct TextureSubresourceSnapshot {
      UINT width = 0;
      UINT height = 0;
      UINT depth = 0;
      UINT rowCount = 0;
      UINT rowPitch = 0;
      UINT64 rowSize = 0;
      DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
      std::vector<uint8_t> bytes;
    };

    struct TextureResourceSnapshot {
      D3D12_RESOURCE_DESC description {};
      std::unordered_map<UINT, TextureSubresourceSnapshot> subresources;
      uint64_t bytes = 0;
    };

    std::unordered_map<ID3D12DescriptorHeap*, DescriptorHeapMetadata> g_descriptorHeaps;
    std::unordered_map<uint64_t, DescriptorMetadata> g_descriptors;
    std::unordered_map<uint64_t, TextureResourceSnapshot> g_textureSnapshots;
    uint64_t g_textureSnapshotBytes = 0;
    constexpr uint64_t kMaxTextureSnapshotBytes = 128ull * 1024ull * 1024ull;
    // Capture only the small, frequently used textures that can be uploaded
    // synchronously into the native material path. Large streaming resources
    // stay on the game's original material until an asynchronous uploader is
    // available; capturing those here stalls the render thread.
    constexpr bool kEnableTextureSnapshotCapture = true;
    constexpr UINT kMaxTextureCaptureWidth = 128;
    constexpr UINT kMaxTextureCaptureHeight = 128;
    constexpr UINT64 kMaxTextureCaptureReadBytes = 64ull * 1024ull;
    constexpr size_t kMaxTextureCaptureResources = 64;
    constexpr bool kEnableTextureCopyDiagnostics = false;
    constexpr bool kEnableNativeBindingSnapshots = true;
    LARGE_INTEGER g_perfFrequency {};
    std::once_flag g_perfFrequencyOnce;
    std::atomic<uint64_t> g_perfDrawTicks { 0 };
    std::atomic<uint64_t> g_perfDrawCalls { 0 };
    std::atomic<uint64_t> g_perfTransformTicks { 0 };
    std::atomic<uint64_t> g_perfBindingTicks { 0 };
    std::atomic<uint64_t> g_perfTextureTicks { 0 };
    std::atomic<uint64_t> g_perfTextureCalls { 0 };
    std::atomic<uint64_t> g_perfGeometryTicks { 0 };
    std::atomic<uint64_t> g_perfGeometryCalls { 0 };
    std::atomic<uint32_t> g_pipelineInputDiagnostics { 0 };

    uint64_t perfTicks() {
      std::call_once(g_perfFrequencyOnce, [] {
        QueryPerformanceFrequency(&g_perfFrequency);
      });
      LARGE_INTEGER value {};
      QueryPerformanceCounter(&value);
      return static_cast<uint64_t>(value.QuadPart);
    }
    constexpr uint64_t kMaxTextureSubresourceBytes = 16ull * 1024ull * 1024ull;

    std::unordered_set<uint64_t> g_loggedNativeBindings;
    constexpr size_t kMaxNativeBindingSnapshots = 8192;

    // Row-vector 4x4 product, so the second operand is applied after the first.
    void multiplyRowVectorMatrices(const float* left, const float* right, float* out) {
      for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t column = 0; column < 4; ++column) {
          float sum = 0.0f;
          for (uint32_t k = 0; k < 4; ++k)
            sum += left[row * 4 + k] * right[k * 4 + column];
          out[row * 4 + column] = sum;
        }
      }
    }

    // General 4x4 inverse by Gauss-Jordan elimination with partial pivoting.
    // Returns false for a singular or non-finite matrix.
    bool invertMatrix4(const float* source, float* out) {
      double augmented[4][8] = {};
      for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t column = 0; column < 4; ++column) {
          if (!std::isfinite(source[row * 4 + column]))
            return false;
          augmented[row][column] = static_cast<double>(source[row * 4 + column]);
        }
        augmented[row][4 + row] = 1.0;
      }

      for (uint32_t pivot = 0; pivot < 4; ++pivot) {
        uint32_t best = pivot;
        for (uint32_t row = pivot + 1; row < 4; ++row) {
          if (std::abs(augmented[row][pivot]) > std::abs(augmented[best][pivot]))
            best = row;
        }
        if (std::abs(augmented[best][pivot]) < 1.0e-12)
          return false;

        if (best != pivot) {
          for (uint32_t column = 0; column < 8; ++column) {
            const double value = augmented[pivot][column];
            augmented[pivot][column] = augmented[best][column];
            augmented[best][column] = value;
          }
        }

        const double scale = 1.0 / augmented[pivot][pivot];
        for (uint32_t column = 0; column < 8; ++column)
          augmented[pivot][column] *= scale;

        for (uint32_t row = 0; row < 4; ++row) {
          if (row == pivot)
            continue;
          const double factor = augmented[row][pivot];
          if (factor == 0.0)
            continue;
          for (uint32_t column = 0; column < 8; ++column)
            augmented[row][column] -= factor * augmented[pivot][column];
        }
      }

      for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t column = 0; column < 4; ++column) {
          const double value = augmented[row][4 + column];
          if (!std::isfinite(value) || std::abs(value) > 1.0e12)
            return false;
          out[row * 4 + column] = static_cast<float>(value);
        }
      }
      return true;
    }

    bool readObjectTransform(
        D3D12_GPU_VIRTUAL_ADDRESS cbv,
        std::array<float, 16>& result,
        uint32_t matrixOffset = 0x100) {
      if (!cbv || cbv > UINT64_MAX - matrixOffset - sizeof(float) * 16)
        return false;

      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(cbv + matrixOffset, sizeof(float) * 16, bytes)
       || bytes.size() < sizeof(float) * 16)
        return false;

      const auto* matrix = reinterpret_cast<const float*>(bytes.data());
      for (unsigned i = 0; i < 16; ++i) {
        if (!std::isfinite(matrix[i]) || std::abs(matrix[i]) > 1.0e7f)
          return false;
      }
      if (std::abs(matrix[3]) > 0.05f || std::abs(matrix[7]) > 0.05f
       || std::abs(matrix[11]) > 0.05f || std::abs(matrix[15] - 1.0f) > 0.05f)
        return false;

      const float n0 = std::sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2]);
      const float n1 = std::sqrt(matrix[4] * matrix[4] + matrix[5] * matrix[5] + matrix[6] * matrix[6]);
      const float n2 = std::sqrt(matrix[8] * matrix[8] + matrix[9] * matrix[9] + matrix[10] * matrix[10]);
      const float d01 = matrix[0] * matrix[4] + matrix[1] * matrix[5] + matrix[2] * matrix[6];
      const float d02 = matrix[0] * matrix[8] + matrix[1] * matrix[9] + matrix[2] * matrix[10];
      const float d12 = matrix[4] * matrix[8] + matrix[5] * matrix[9] + matrix[6] * matrix[10];
      if (n0 < 0.5f || n0 > 2.0f || n1 < 0.5f || n1 > 2.0f || n2 < 0.5f || n2 > 2.0f
       || std::abs(d01) > 0.08f || std::abs(d02) > 0.08f || std::abs(d12) > 0.08f)
        return false;

      std::copy(matrix, matrix + 16, result.begin());
      return true;
    }

    bool isIdentityObjectTransform(const std::array<float, 16>& matrix) {
      return std::abs(matrix[0] - 1.0f) < 0.001f
        && std::abs(matrix[5] - 1.0f) < 0.001f
        && std::abs(matrix[10] - 1.0f) < 0.001f
        && std::abs(matrix[15] - 1.0f) < 0.001f
        && std::abs(matrix[1]) < 0.001f && std::abs(matrix[2]) < 0.001f
        && std::abs(matrix[3]) < 0.001f && std::abs(matrix[4]) < 0.001f
        && std::abs(matrix[6]) < 0.001f && std::abs(matrix[7]) < 0.001f
        && std::abs(matrix[8]) < 0.001f && std::abs(matrix[9]) < 0.001f
        && std::abs(matrix[11]) < 0.001f && std::abs(matrix[12]) < 0.001f
        && std::abs(matrix[13]) < 0.001f && std::abs(matrix[14]) < 0.001f;
    }

    // Resolves the per-draw structured buffer bound at t10, falling back to the
    // table base, and reports the GPU address and readable width of the backing
    // buffer. Model vertex shaders read all of their transform data from it.
    bool resolveDrawShaderBuffer(
        const D3D12_GPU_DESCRIPTOR_HANDLE* tables,
        D3D12_GPU_VIRTUAL_ADDRESS& gpuAddress,
        UINT64& resourceWidth) {
      static std::atomic<uint32_t> lookupDiagnostics { 0 };
      const uint32_t diagnosticIndex = lookupDiagnostics.fetch_add(1, std::memory_order_relaxed);
      const bool emitDiagnostic = diagnosticIndex < 8 || (diagnosticIndex % 1024) == 0;


      if (!tables)
        return false;

      // One-shot dump of the table handles and heap ranges; the pair is enough
      // to tell offline whether a table lands inside a tracked heap.
      if (diagnosticIndex < 3) {
        log("D3D12 draw buffer diag: tables=(0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx) heaps=%u",
          static_cast<unsigned long long>(tables[0].ptr),
          static_cast<unsigned long long>(tables[1].ptr),
          static_cast<unsigned long long>(tables[2].ptr),
          static_cast<unsigned long long>(tables[3].ptr),
          static_cast<unsigned long long>(tables[4].ptr),
          static_cast<unsigned long long>(tables[5].ptr),
          static_cast<unsigned long long>(tables[6].ptr),
          static_cast<unsigned long long>(tables[7].ptr),
          static_cast<unsigned>(g_descriptorHeaps.size()));

        std::lock_guard lock(g_frameMutex);
        unsigned reported = 0;
        for (const auto& entry : g_descriptorHeaps) {
          if (reported >= 8)
            break;
          const auto& metadata = entry.second;
          log("D3D12 draw buffer diag heap: gpu=0x%llx cpu=0x%llx count=%u inc=%u",
            static_cast<unsigned long long>(metadata.gpuStart.ptr),
            static_cast<unsigned long long>(metadata.cpuStart.ptr),
            metadata.count, metadata.increment);
          ++reported;
        }
      }

      std::array<size_t, 16> offsets { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
      for (size_t tableIndex = 0; tableIndex < 16; ++tableIndex) {
        const uint64_t tableGpu = tables[tableIndex].ptr;
        if (!tableGpu)
          continue;

        for (const size_t descriptorOffset : offsets) {
          std::lock_guard lock(g_frameMutex);
          const DescriptorHeapMetadata* heap = nullptr;
          for (const auto& entry : g_descriptorHeaps) {
            const auto& metadata = entry.second;
            if (!metadata.gpuStart.ptr || !metadata.increment
             || tableGpu < metadata.gpuStart.ptr)
              continue;
            const uint64_t delta = tableGpu - metadata.gpuStart.ptr;
            if (delta % metadata.increment)
              continue;
            const uint64_t tableIndexInHeap = delta / metadata.increment;
            if (tableIndexInHeap >= metadata.count)
              continue;
            heap = &metadata;
            break;
          }
          if (!heap) {
            if (emitDiagnostic)
              log("D3D12 transform lookup: sample=%u table=%zu gpu=0x%llx offset=%zu heap=none",
                diagnosticIndex, tableIndex, static_cast<unsigned long long>(tableGpu), descriptorOffset);
            continue;
          }

          const uint64_t descriptorGpu = tableGpu
            + static_cast<uint64_t>(descriptorOffset) * heap->increment;
          const uint64_t descriptorDelta = descriptorGpu - heap->gpuStart.ptr;
          if (descriptorDelta % heap->increment)
            continue;
          const uint64_t descriptorIndex = descriptorDelta / heap->increment;
          if (descriptorIndex >= heap->count)
            continue;

          const uint64_t cpu = heap->cpuStart.ptr + descriptorIndex * heap->increment;
          const auto descriptor = g_descriptors.find(cpu);

          // Decode the descriptor straight out of the heap. Parts of the engine
          // build descriptor memory without calling a device method, so the
          // creation hooks never see those views. A constant-buffer or buffer
          // view starts with a GPU virtual address followed by its byte size.
          if (descriptor == g_descriptors.end() && cpu) {
            uint64_t bufferGpu = 0;
            uint32_t bufferSize = 0;
            std::memcpy(&bufferGpu, reinterpret_cast<const void*>(cpu), sizeof(bufferGpu));
            std::memcpy(&bufferSize, reinterpret_cast<const void*>(cpu + 8), sizeof(bufferSize));
            if (bufferGpu > 0x100000000ull && bufferGpu < 0x7fffffffffffull
             && bufferSize > 0 && bufferSize < (1u << 28)) {
              gpuAddress = bufferGpu;
              resourceWidth = bufferSize;
              if (diagnosticIndex < 2) {
                log("D3D12 draw buffer diag raw: table=%zu off=%zu idx=%llu gpu=0x%llx size=%u",
                  tableIndex, descriptorOffset, static_cast<unsigned long long>(descriptorIndex),
                  static_cast<unsigned long long>(bufferGpu), bufferSize);
              }
              return true;
            }
          }
          if (diagnosticIndex < 2) {
            const DescriptorMetadata* md = descriptor != g_descriptors.end() ? &descriptor->second : nullptr;
            log("D3D12 draw buffer diag desc: table=%zu off=%zu idx=%llu found=%u view=%s dim=%u width=%llu gpu=0x%llx",
              tableIndex, descriptorOffset, static_cast<unsigned long long>(descriptorIndex),
              md ? 1u : 0u, md && md->viewType ? md->viewType : "none",
              md ? static_cast<unsigned>(md->resourceDescription.Dimension) : 0u,
              md ? static_cast<unsigned long long>(md->resourceDescription.Width) : 0ull,
              md ? static_cast<unsigned long long>(md->resourceGpuAddress) : 0ull);
          }
          if (diagnosticIndex < 3
           && (descriptor == g_descriptors.end() || !descriptor->second.viewType)) {
            log("D3D12 draw buffer diag missing: table=%zu offset=%zu descIndex=%llu cpu=0x%llx tracked=%zu",
              tableIndex, descriptorOffset, static_cast<unsigned long long>(descriptorIndex),
              static_cast<unsigned long long>(cpu), g_descriptors.size());
          }
          if (descriptor == g_descriptors.end()
           || !descriptor->second.resource
           || !descriptor->second.viewType
           || (std::strcmp(descriptor->second.viewType, "SRV") != 0
            && std::strcmp(descriptor->second.viewType, "CBV") != 0)) {
            if (emitDiagnostic)
              log("D3D12 transform lookup: sample=%u table=%zu gpu=0x%llx offset=%zu heapStart=0x%llx inc=%u descriptor=missing-or-non-srv",
                diagnosticIndex, tableIndex, static_cast<unsigned long long>(tableGpu), descriptorOffset,
                static_cast<unsigned long long>(heap->gpuStart.ptr), heap->increment);
            continue;
          }

          // CBVs inside a descriptor table carry their buffer range directly.
          if (std::strcmp(descriptor->second.viewType, "CBV") == 0) {
            const D3D12_GPU_VIRTUAL_ADDRESS cbvAddress = descriptor->second.bufferLocation;
            if (!cbvAddress)
              continue;
            gpuAddress = cbvAddress;
            resourceWidth = descriptor->second.bufferSize ? descriptor->second.bufferSize : 0x1000;
            return true;
          }

          const D3D12_RESOURCE_DESC& resourceDescription = descriptor->second.resourceDescription;
          if (resourceDescription.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
            if (emitDiagnostic)
              log("D3D12 transform lookup: sample=%u table=%zu offset=%zu descriptor=srv resource=%p dimension=%u",
                diagnosticIndex, tableIndex, descriptorOffset, descriptor->second.resource,
                static_cast<unsigned>(resourceDescription.Dimension));
            continue;
          }

          D3D12_GPU_VIRTUAL_ADDRESS resourceGpuAddress = descriptor->second.resourceGpuAddress;
          UINT64 firstElementOffset = 0;
          if (descriptor->second.viewDescriptionSize >= sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) {
            const auto* srv = reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(
              descriptor->second.viewDescription.data());
            if (srv->ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
              firstElementOffset = static_cast<UINT64>(srv->Buffer.FirstElement)
                * srv->Buffer.StructureByteStride;
            }
          }
          resourceGpuAddress += firstElementOffset;
          if (!resourceGpuAddress || resourceDescription.Width <= firstElementOffset) {
            if (emitDiagnostic)
              log("D3D12 transform lookup: sample=%u table=%zu offset=%zu descriptor=srv resource=%p buffer-invalid gpu=0x%llx width=%llu",
                diagnosticIndex, tableIndex, descriptorOffset, descriptor->second.resource,
                static_cast<unsigned long long>(resourceGpuAddress),
                static_cast<unsigned long long>(resourceDescription.Width));
            continue;
          }

          gpuAddress = resourceGpuAddress;
          resourceWidth = resourceDescription.Width - firstElementOffset;
          if (emitDiagnostic)
            log("D3D12 transform lookup: sample=%u table=%zu offset=%zu descriptor=srv resource=%p gpu=0x%llx width=%llu",
              diagnosticIndex, tableIndex, descriptorOffset, descriptor->second.resource,
              static_cast<unsigned long long>(gpuAddress), static_cast<unsigned long long>(resourceWidth));
          if (diagnosticIndex < 3) {
            std::vector<uint8_t> probe;
            const bool readOk = readGpuBuffer(gpuAddress, sizeof(float) * 8, probe);
            log("D3D12 draw buffer diag read: ok=%u bytes=%zu first=%g,%g,%g,%g",
              readOk ? 1u : 0u, probe.size(),
              probe.size() >= 16 ? static_cast<double>(reinterpret_cast<const float*>(probe.data())[0]) : 0.0,
              probe.size() >= 16 ? static_cast<double>(reinterpret_cast<const float*>(probe.data())[1]) : 0.0,
              probe.size() >= 16 ? static_cast<double>(reinterpret_cast<const float*>(probe.data())[2]) : 0.0,
              probe.size() >= 16 ? static_cast<double>(reinterpret_cast<const float*>(probe.data())[3]) : 0.0);
          }
          return true;
        }
      }
      return false;
    }

    // Diagnostic for the model vertex shader: dumps every root descriptor table
    // and the first descriptors of each, including the raw heap bytes. Used to
    // locate the draw buffer that carries the camera.
    void probeDrawBufferBindings(
        const D3D12_GPU_DESCRIPTOR_HANDLE* tables,
        const D3D12_GPU_VIRTUAL_ADDRESS* rootSrvs) {
      static std::atomic<uint32_t> probes { 0 };
      if (!tables || probes.fetch_add(1, std::memory_order_relaxed) >= 3)
        return;

      std::lock_guard lock(g_frameMutex);
      log("D3D12 probe start: heaps=%zu", g_descriptorHeaps.size());
      log("D3D12 probe roots: srv=(0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx) cbv2=0x%llx cbvA=0x%llx",
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[0]) : 0ull,
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[1]) : 0ull,
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[2]) : 0ull,
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[3]) : 0ull,
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[4]) : 0ull,
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[5]) : 0ull,
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[6]) : 0ull,
        rootSrvs ? static_cast<unsigned long long>(rootSrvs[7]) : 0ull,
        static_cast<unsigned long long>(currentGraphicsRootCbv(2)),
        static_cast<unsigned long long>(currentGraphicsRootCbv(0)));
      for (size_t tableIndex = 0; tableIndex < 16; ++tableIndex) {
        const uint64_t tableGpu = tables[tableIndex].ptr;
        if (!tableGpu)
          continue;

        const DescriptorHeapMetadata* heap = nullptr;
        for (const auto& entry : g_descriptorHeaps) {
          const auto& metadata = entry.second;
          if (!metadata.gpuStart.ptr || !metadata.increment || tableGpu < metadata.gpuStart.ptr)
            continue;
          const uint64_t delta = tableGpu - metadata.gpuStart.ptr;
          if (delta % metadata.increment)
            continue;
          if (delta / metadata.increment >= metadata.count)
            continue;
          heap = &metadata;
          break;
        }

        log("D3D12 probe table: slot=%zu gpu=0x%llx heap=%s", tableIndex,
          static_cast<unsigned long long>(tableGpu), heap ? "yes" : "no");
        if (!heap)
          continue;

        const uint64_t baseIndex = (tableGpu - heap->gpuStart.ptr) / heap->increment;
        for (uint32_t offset = 0; offset < 16; ++offset) {
          const uint64_t index = baseIndex + offset;
          if (index >= heap->count)
            break;
          const uint64_t cpu = heap->cpuStart.ptr + index * heap->increment;
          const auto stored = g_descriptors.find(cpu);
          uint64_t rawGpu = 0;
          uint32_t rawSize = 0;
          std::memcpy(&rawGpu, reinterpret_cast<const void*>(cpu), sizeof(rawGpu));
          std::memcpy(&rawSize, reinterpret_cast<const void*>(cpu + 8), sizeof(rawSize));
          log("D3D12 probe desc: slot=%zu off=%u idx=%llu found=%u view=%s dim=%u width=%llu gpu=0x%llx rawGpu=0x%llx rawSize=%u",
            tableIndex, offset, static_cast<unsigned long long>(index),
            stored != g_descriptors.end() ? 1u : 0u,
            stored != g_descriptors.end() && stored->second.viewType ? stored->second.viewType : "none",
            stored != g_descriptors.end()
              ? static_cast<unsigned>(stored->second.resourceDescription.Dimension) : 0u,
            stored != g_descriptors.end()
              ? static_cast<unsigned long long>(stored->second.resourceDescription.Width) : 0ull,
            stored != g_descriptors.end()
              ? static_cast<unsigned long long>(stored->second.resourceGpuAddress) : 0ull,
            static_cast<unsigned long long>(rawGpu), rawSize);
        }
      }
    }

    bool readMatrixFromShaderBuffer(
        D3D12_GPU_VIRTUAL_ADDRESS gpuAddress,
        UINT64 resourceWidth,
        uint32_t offset,
        std::array<float, 16>& result) {
      if (!gpuAddress || resourceWidth < offset + sizeof(float) * 16)
        return false;

      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(gpuAddress + offset, sizeof(float) * 16, bytes)
       || bytes.size() < sizeof(float) * 16)
        return false;

      const auto* matrix = reinterpret_cast<const float*>(bytes.data());
      for (unsigned i = 0; i < 16; ++i) {
        if (!std::isfinite(matrix[i]) || std::abs(matrix[i]) > 1.0e7f)
          return false;
      }
      std::copy(matrix, matrix + 16, result.begin());
      return true;
    }

    // Object-to-world matrix, stored at byte offset 0x100 of the draw buffer.
    // The buffer address is reported so the leading matrix at 0x000 can be read
    // from the same binding.
    bool readObjectTransformFromDescriptorTables(
        const D3D12_GPU_DESCRIPTOR_HANDLE* tables,
        std::array<float, 16>& result,
        D3D12_GPU_VIRTUAL_ADDRESS* bufferOut = nullptr,
        UINT64* bufferWidthOut = nullptr) {
      D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
      UINT64 resourceWidth = 0;
      if (!resolveDrawShaderBuffer(tables, gpuAddress, resourceWidth))
        return false;

      std::array<float, 16> matrix {};
      if (!readMatrixFromShaderBuffer(gpuAddress, resourceWidth, 0x100, matrix))
        return false;

      if (std::abs(matrix[3]) > 0.05f || std::abs(matrix[7]) > 0.05f
       || std::abs(matrix[11]) > 0.05f || std::abs(matrix[15] - 1.0f) > 0.05f)
        return false;

      const float n0 = std::sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2]);
      const float n1 = std::sqrt(matrix[4] * matrix[4] + matrix[5] * matrix[5] + matrix[6] * matrix[6]);
      const float n2 = std::sqrt(matrix[8] * matrix[8] + matrix[9] * matrix[9] + matrix[10] * matrix[10]);
      const float d01 = matrix[0] * matrix[4] + matrix[1] * matrix[5] + matrix[2] * matrix[6];
      const float d02 = matrix[0] * matrix[8] + matrix[1] * matrix[9] + matrix[2] * matrix[10];
      const float d12 = matrix[4] * matrix[8] + matrix[5] * matrix[9] + matrix[6] * matrix[10];
      if (n0 < 0.5f || n0 > 2.0f || n1 < 0.5f || n1 > 2.0f || n2 < 0.5f || n2 > 2.0f
       || std::abs(d01) > 0.08f || std::abs(d02) > 0.08f || std::abs(d12) > 0.08f)
        return false;

      result = matrix;
      if (bufferOut)
        *bufferOut = gpuAddress;
      if (bufferWidthOut)
        *bufferWidthOut = resourceWidth;
      return true;
    }

    // Camera world-to-clip matrix, stored at byte offset 0 of the draw buffer.
    // The model vertex shader applies it after the object matrix, so it is the
    // per-frame view-projection.
    bool isCameraViewProjection(const std::array<float, 16>& matrix) {
      const float fx = matrix[3];
      const float fy = matrix[7];
      const float fz = matrix[11];
      const float forwardLength = std::sqrt(fx * fx + fy * fy + fz * fz);
      if (std::abs(forwardLength - 1.0f) > 0.05f)
        return false;

      const float p00 = std::sqrt(matrix[0] * matrix[0] + matrix[4] * matrix[4] + matrix[8] * matrix[8]);
      const float p11 = std::sqrt(matrix[1] * matrix[1] + matrix[5] * matrix[5] + matrix[9] * matrix[9]);
      const float p22 = matrix[2] * fx + matrix[6] * fy + matrix[10] * fz;
      return p00 >= 0.1f && p11 >= 0.1f && p22 >= 0.1f && p22 <= 10.0f;
    }

    bool readCameraFromDescriptorTables(
        const D3D12_GPU_DESCRIPTOR_HANDLE* tables,
        std::array<float, 16>& result) {
      D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
      UINT64 resourceWidth = 0;
      if (!resolveDrawShaderBuffer(tables, gpuAddress, resourceWidth))
        return false;

      std::array<float, 16> matrix {};
      if (!readMatrixFromShaderBuffer(gpuAddress, resourceWidth, 0x0, matrix))
        return false;
      if (!isCameraViewProjection(matrix))
        return false;

      result = matrix;
      return true;
    }

    std::unordered_set<uint64_t> g_loggedDrawBindings;
    std::unordered_set<uint64_t> g_loggedSrvBuffers;
    std::once_flag g_bindingFileOnce;
    std::ofstream g_bindingFile;
    // Descriptor-table inspection maps and GPU reads are useful during reverse
    // engineering, but are far too expensive to run on the first 4096 draws.
    constexpr bool kEnableDrawDiagnostics = false;

    void openBindingFile() {
      wchar_t modulePath[MAX_PATH] = {};
      DWORD length = GetModuleFileNameW(nullptr, modulePath, ARRAYSIZE(modulePath));
      if (!length || length >= ARRAYSIZE(modulePath))
        return;
      std::filesystem::path path(modulePath);
      path = path.parent_path() / L"remix-d3d12-bindings.csv";
      g_bindingFile.open(path, std::ios::out | std::ios::trunc);
      if (g_bindingFile) {
        g_bindingFile << "frame,pipeline,vs_hash,ps_hash,topology,index_count,instance_count,vb_gpu,vb_size,vb_stride,ib_gpu,ib_size,ib_format,table0_gpu,table0_resource,table0_format,table0_width,table0_height,table1_gpu,table1_resource,table1_format,table1_width,table1_height,table2_gpu,table2_resource,table2_format,table2_width,table2_height,table3_gpu,table3_resource,table3_format,table3_width,table3_height\n";
      }
    }

    const ResourceMetadata* findResourceMetadata(ID3D12Resource* resource) {
      const auto entry = g_resourceMetadata.find(resource);
      return entry == g_resourceMetadata.end() ? nullptr : &entry->second;
    }

    uint64_t bytecodeHash(const D3D12_SHADER_BYTECODE& bytecode) {
      if (!bytecode.pShaderBytecode || !bytecode.BytecodeLength)
        return 0;
      const auto* bytes = static_cast<const uint8_t*>(bytecode.pShaderBytecode);
      uint64_t hash = 1469598103934665603ull;
      for (SIZE_T i = 0; i < bytecode.BytecodeLength; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
      }
      return hash;
    }

    uint64_t drawBindingHash(
        ID3D12PipelineState* pipelineState,
        const D3D12_VERTEX_BUFFER_VIEW& vertexBuffer,
        const D3D12_INDEX_BUFFER_VIEW& indexBuffer,
        const D3D12_GPU_DESCRIPTOR_HANDLE* tables) {
      uint64_t hash = reinterpret_cast<uint64_t>(pipelineState);
      hash ^= vertexBuffer.BufferLocation + 0x9E3779B97F4A7C15ull + (hash << 6) + (hash >> 2);
      hash ^= indexBuffer.BufferLocation + 0x9E3779B97F4A7C15ull + (hash << 6) + (hash >> 2);
      hash ^= static_cast<uint64_t>(vertexBuffer.StrideInBytes) << 17;
      hash ^= static_cast<uint64_t>(indexBuffer.Format) << 29;
      for (size_t i = 0; i < 4; ++i) {
        const uint64_t table = tables ? tables[i].ptr : 0;
        hash ^= table + 0x9E3779B97F4A7C15ull + (hash << 6) + (hash >> 2);
      }
      return hash;
    }

    bool shouldLog(uint64_t count) {
      return count <= 8 || (count % 1024) == 0;
    }

    void logSrvBufferData(uint64_t descriptorGpu, const ResourceMetadata& resource) {
      if (resource.description.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
       || !resource.gpuAddress || !resource.description.Width)
        return;

      if (g_loggedSrvBuffers.size() >= 128
       || !g_loggedSrvBuffers.emplace(resource.gpuAddress).second)
        return;

      const size_t bytesToRead = static_cast<size_t>(std::min<UINT64>(resource.description.Width, 4096));
      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(resource.gpuAddress, bytesToRead, bytes)) {
        log("D3D12 SRV buffer data unavailable: descriptorGpu=0x%llx gpu=0x%llx size=%llu",
          static_cast<unsigned long long>(descriptorGpu),
          static_cast<unsigned long long>(resource.gpuAddress),
          static_cast<unsigned long long>(resource.description.Width));
        return;
      }

      log("D3D12 SRV buffer data: descriptorGpu=0x%llx gpu=0x%llx size=%llu bytes=%zu",
        static_cast<unsigned long long>(descriptorGpu),
        static_cast<unsigned long long>(resource.gpuAddress),
        static_cast<unsigned long long>(resource.description.Width), bytes.size());
      const size_t rowCount = bytes.size() / (sizeof(float) * 4);
      const auto* values = reinterpret_cast<const float*>(bytes.data());
      for (size_t row = 0; row < rowCount; ++row) {
        log("D3D12 SRV buffer row%zu: %.9g %.9g %.9g %.9g",
          row,
          static_cast<double>(values[row * 4 + 0]),
          static_cast<double>(values[row * 4 + 1]),
          static_cast<double>(values[row * 4 + 2]),
          static_cast<double>(values[row * 4 + 3]));
      }
    }

    void captureTextureUpload(
        const D3D12_TEXTURE_COPY_LOCATION& destination,
        const D3D12_TEXTURE_COPY_LOCATION& source,
        UINT destinationX,
        UINT destinationY,
        UINT destinationZ,
        const D3D12_BOX* sourceBox) {
      if (destination.Type != D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX
       || source.Type != D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT
       || !destination.pResource || !source.pResource
       || destinationX || destinationY || destinationZ || sourceBox)
        return;

      const D3D12_RESOURCE_DESC description = destination.pResource->GetDesc();
      if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
       || !description.Width || !description.Height)
        return;

      const UINT mipLevels = description.MipLevels ? description.MipLevels : 1;
      const UINT arraySize = description.DepthOrArraySize;
      const UINT subresource = destination.SubresourceIndex;
      if (!arraySize || subresource >= mipLevels * arraySize)
        return;

      const UINT mip = subresource % mipLevels;
      const UINT width = static_cast<UINT>(std::max<UINT64>(1, description.Width >> mip));
      const UINT height = std::max<UINT>(1, description.Height >> mip);
      if (!kEnableTextureSnapshotCapture
       || width > kMaxTextureCaptureWidth
       || height > kMaxTextureCaptureHeight)
        return;
      const auto& footprint = source.PlacedFootprint.Footprint;
      if (footprint.Width != width || footprint.Height != height || footprint.Depth != 1)
        return;

      ID3D12Device* device = nullptr;
      if (FAILED(destination.pResource->GetDevice(
          __uuidof(ID3D12Device), reinterpret_cast<void**>(&device)))
       || !device)
        return;

      D3D12_PLACED_SUBRESOURCE_FOOTPRINT expectedLayout {};
      UINT rowCount = 0;
      UINT64 rowSize = 0;
      UINT64 totalBytes = 0;
      device->GetCopyableFootprints(
        &description, subresource, 1, 0, &expectedLayout, &rowCount, &rowSize, &totalBytes);
      device->Release();

      if (!rowCount || !rowSize || footprint.RowPitch < rowSize
       || rowSize > kMaxTextureSubresourceBytes
       || rowCount > kMaxTextureSubresourceBytes / rowSize)
        return;

      const UINT64 lastRowOffset = static_cast<UINT64>(rowCount - 1) * footprint.RowPitch;
      if (lastRowOffset > UINT64_MAX - rowSize)
        return;
      const UINT64 readSize = lastRowOffset + rowSize;
      if (readSize > kMaxTextureSubresourceBytes
       || readSize > kMaxTextureCaptureReadBytes)
        return;

      const uint64_t resourceKey = reinterpret_cast<uint64_t>(destination.pResource);
      {
        std::lock_guard lock(g_frameMutex);
        const auto existing = g_textureSnapshots.find(resourceKey);
        if (existing != g_textureSnapshots.end()
         && existing->second.subresources.find(subresource)
              != existing->second.subresources.end())
          return;
        if (existing == g_textureSnapshots.end()
         && g_textureSnapshots.size() >= kMaxTextureCaptureResources)
          return;
      }

      const D3D12_GPU_VIRTUAL_ADDRESS sourceAddress = source.pResource->GetGPUVirtualAddress();
      if (!sourceAddress || sourceAddress > UINT64_MAX - source.PlacedFootprint.Offset)
        return;

      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(
          sourceAddress + source.PlacedFootprint.Offset,
          static_cast<size_t>(readSize), bytes)
       || bytes.size() != readSize) {
        return;
      }

      TextureSubresourceSnapshot snapshot;
      snapshot.width = width;
      snapshot.height = height;
      snapshot.depth = 1;
      snapshot.rowCount = rowCount;
      snapshot.rowPitch = footprint.RowPitch;
      snapshot.rowSize = rowSize;
      snapshot.format = footprint.Format;
      snapshot.bytes = std::move(bytes);

      std::lock_guard lock(g_frameMutex);
      auto& texture = g_textureSnapshots[resourceKey];
      texture.description = description;
      auto existing = texture.subresources.find(subresource);
      const uint64_t existingBytes = existing != texture.subresources.end()
        ? existing->second.bytes.size() : 0;
      const uint64_t textureBytesWithoutExisting = texture.bytes - existingBytes;
      const uint64_t totalBytesWithoutExisting = g_textureSnapshotBytes - existingBytes;
      if (textureBytesWithoutExisting > kMaxTextureSnapshotBytes - snapshot.bytes.size()
       || totalBytesWithoutExisting > kMaxTextureSnapshotBytes - snapshot.bytes.size()) {
        if (kEnableTextureCopyDiagnostics) {
          log("D3D12 texture snapshot skipped: resource=%p subresource=%u bytes=%zu cacheBytes=%llu",
            destination.pResource, subresource, snapshot.bytes.size(),
            static_cast<unsigned long long>(g_textureSnapshotBytes));
        }
        if (texture.subresources.empty())
          g_textureSnapshots.erase(resourceKey);
        return;
      }

      if (existing != texture.subresources.end())
        texture.subresources.erase(existing);
      texture.bytes = textureBytesWithoutExisting + snapshot.bytes.size();
      g_textureSnapshotBytes = totalBytesWithoutExisting + snapshot.bytes.size();
      texture.subresources.emplace(subresource, std::move(snapshot));
      if (kEnableTextureCopyDiagnostics) {
        log("D3D12 texture snapshot captured: resource=%p subresource=%u size=%ux%u rows=%u rowPitch=%u rowSize=%llu format=%u bytes=%llu cacheBytes=%llu",
          destination.pResource, subresource, width, height, rowCount, footprint.RowPitch,
          static_cast<unsigned long long>(rowSize), static_cast<unsigned>(footprint.Format),
          static_cast<unsigned long long>(readSize),
          static_cast<unsigned long long>(g_textureSnapshotBytes));
      }
    }

    D3D12NativeDescriptorKind getNativeDescriptorKind(const DescriptorMetadata& descriptor) {
      if (descriptor.sampler || (descriptor.viewType
          && std::strcmp(descriptor.viewType, "Sampler") == 0))
        return D3D12NativeDescriptorKind::Sampler;
      if (!descriptor.viewType)
        return D3D12NativeDescriptorKind::Unknown;
      if (std::strcmp(descriptor.viewType, "CBV") == 0)
        return D3D12NativeDescriptorKind::Cbv;
      if (std::strcmp(descriptor.viewType, "SRV") == 0)
        return D3D12NativeDescriptorKind::Srv;
      if (std::strcmp(descriptor.viewType, "UAV") == 0)
        return D3D12NativeDescriptorKind::Uav;
      if (std::strcmp(descriptor.viewType, "RTV") == 0)
        return D3D12NativeDescriptorKind::Rtv;
      if (std::strcmp(descriptor.viewType, "DSV") == 0)
        return D3D12NativeDescriptorKind::Dsv;
      return D3D12NativeDescriptorKind::Unknown;
    }

    void collectBoundDescriptors(
        const D3D12_GPU_DESCRIPTOR_HANDLE* tables,
        std::vector<D3D12NativeBoundDescriptor>& output) {
      if (!tables)
        return;

      constexpr UINT kDescriptorsPerTable = 32;
      constexpr size_t kMaxCapturedDescriptors = 128;
      for (UINT rootIndex = 0; rootIndex < 8 && output.size() < kMaxCapturedDescriptors; ++rootIndex) {
        const uint64_t tableGpu = tables[rootIndex].ptr;
        if (!tableGpu)
          continue;

        const DescriptorHeapMetadata* heap = nullptr;
        uint64_t tableIndex = 0;
        for (const auto& entry : g_descriptorHeaps) {
          const auto& candidate = entry.second;
          if (!candidate.gpuStart.ptr || !candidate.increment || tableGpu < candidate.gpuStart.ptr)
            continue;
          const uint64_t delta = tableGpu - candidate.gpuStart.ptr;
          if (delta % candidate.increment)
            continue;
          const uint64_t candidateIndex = delta / candidate.increment;
          if (candidateIndex >= candidate.count)
            continue;
          heap = &candidate;
          tableIndex = candidateIndex;
          break;
        }
        if (!heap)
          continue;

        for (UINT offset = 0; offset < kDescriptorsPerTable
         && tableIndex + offset < heap->count
         && output.size() < kMaxCapturedDescriptors; ++offset) {
          const uint64_t cpu = heap->cpuStart.ptr + (tableIndex + offset) * heap->increment;
          const auto descriptor = g_descriptors.find(cpu);
          if (descriptor == g_descriptors.end())
            continue;

          const auto& source = descriptor->second;
          D3D12NativeBoundDescriptor bound;
          bound.rootParameterIndex = rootIndex;
          bound.tableOffset = offset;
          bound.kind = getNativeDescriptorKind(source);
          bound.resource = reinterpret_cast<uint64_t>(source.resource);
          bound.resourceDescription = source.resourceDescription;
          bound.format = source.format;
          bound.bufferLocation = source.bufferLocation;
          bound.bufferSize = source.bufferSize;
          bound.viewDescription = source.viewDescription;
          bound.viewDescriptionSize = source.viewDescriptionSize;
          if (source.resource) {
            const auto texture = g_textureSnapshots.find(
              reinterpret_cast<uint64_t>(source.resource));
            if (texture != g_textureSnapshots.end()) {
              bound.capturedSubresources = static_cast<UINT>(texture->second.subresources.size());
              bound.capturedBytes = texture->second.bytes;
            }
          }
          output.push_back(std::move(bound));
        }
      }
    }
  }

  void initializeD3D12RuntimeAdapter() {
    std::call_once(g_initialized, [] {
      // Initialize the counter before the first draw. Launcher-only frames
      // still need a valid performance window, and native timing must not
      // depend on a draw or texture-capture callback having run first.
      std::call_once(g_perfFrequencyOnce, [] {
        QueryPerformanceFrequency(&g_perfFrequency);
      });
      log("D3D12 native Remix frontend initialized; D3D9 bridge disabled");
      log("D3D12 native frontend contract: device -> queue -> command list -> draw -> submit");
      initializeD3D12NativeBackend();
    });
  }

  void recordD3D12GeometryCaptureDuration(uint64_t durationTicks) {
    g_perfGeometryTicks.fetch_add(durationTicks, std::memory_order_relaxed);
    g_perfGeometryCalls.fetch_add(1, std::memory_order_relaxed);
  }

  bool getD3D12TextureSnapshot(
      uint64_t resource,
      UINT subresource,
      D3D12NativeTextureSnapshot& snapshot) {
    if (!resource)
      return false;

    std::lock_guard lock(g_frameMutex);
    const auto texture = g_textureSnapshots.find(resource);
    if (texture == g_textureSnapshots.end())
      return false;
    const auto level = texture->second.subresources.find(subresource);
    if (level == texture->second.subresources.end())
      return false;
    snapshot.width = level->second.width;
    snapshot.height = level->second.height;
    snapshot.rowPitch = level->second.rowPitch;
    snapshot.rowSize = level->second.rowSize;
    snapshot.format = level->second.format;
    snapshot.bytes = level->second.bytes;
    return !snapshot.bytes.empty();
  }

  void attachD3D12Device(ID3D12Device* device) {
    initializeD3D12RuntimeAdapter();
    attachD3D12NativeDevice(device);
    const uint64_t count = ++g_deviceCount;
    if (!shouldLog(count))
      return;
    log("D3D12 native frontend device attached: device=%p nodeCount=%u count=%llu",
      device, device ? device->GetNodeCount() : 0u, static_cast<unsigned long long>(count));
  }

  void attachD3D12CommandQueue(ID3D12CommandQueue* commandQueue, const D3D12_COMMAND_QUEUE_DESC& description) {
    initializeD3D12RuntimeAdapter();
    attachD3D12NativeCommandQueue(commandQueue);
    const uint64_t count = ++g_queueCount;
    if (!shouldLog(count))
      return;
    log("D3D12 native frontend queue attached: queue=%p type=%u flags=0x%x node=%u count=%llu",
      commandQueue, static_cast<unsigned>(description.Type), description.Flags, description.NodeMask,
      static_cast<unsigned long long>(count));
  }

  void attachD3D12CommandList(ID3D12GraphicsCommandList* commandList) {
    initializeD3D12RuntimeAdapter();
    const uint64_t count = ++g_commandListCount;
    if (!shouldLog(count))
      return;
    log("D3D12 native frontend command list attached: list=%p count=%llu",
      commandList, static_cast<unsigned long long>(count));
  }

  void recordD3D12Pipeline(
      const D3D12_GRAPHICS_PIPELINE_STATE_DESC* description,
      ID3D12PipelineState* pipelineState,
      HRESULT result) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    ++g_frame.pipelines;
    if (SUCCEEDED(result) && pipelineState && description) {
      PipelineMetadata metadata;
      metadata.vertexShaderHash = bytecodeHash(description->VS);
      metadata.pixelShaderHash = bytecodeHash(description->PS);
      metadata.nativeInfo.vertexShaderHash = metadata.vertexShaderHash;
      metadata.nativeInfo.pixelShaderHash = metadata.pixelShaderHash;
      metadata.nativeInfo.blendEnabled = description->NumRenderTargets != 0
        && description->BlendState.RenderTarget[0].BlendEnable;
      metadata.nativeInfo.depthWriteEnabled =
        description->DepthStencilState.DepthWriteMask != D3D12_DEPTH_WRITE_MASK_ZERO;
      metadata.inputElementCount = description->InputLayout.NumElements;
      metadata.nativeInfo.inputElements.reserve(description->InputLayout.NumElements);
      UINT slotOffsets[32] = {};
      for (UINT i = 0; i < description->InputLayout.NumElements; ++i) {
        const auto& element = description->InputLayout.pInputElementDescs[i];
        D3D12NativeInputElement nativeElement;
        nativeElement.semantic = element.SemanticName ? element.SemanticName : "";
        nativeElement.semanticIndex = element.SemanticIndex;
        nativeElement.format = element.Format;
        nativeElement.slot = element.InputSlot;
        nativeElement.offset = element.AlignedByteOffset;
        if (nativeElement.offset == D3D12_APPEND_ALIGNED_ELEMENT) {
          nativeElement.offset = element.InputSlot < ARRAYSIZE(slotOffsets)
            ? slotOffsets[element.InputSlot] : 0;
        }
        const UINT elementSize = [&] {
          switch (element.Format) {
          case DXGI_FORMAT_R32_FLOAT: return 4u;
          case DXGI_FORMAT_R32G32_FLOAT: return 8u;
          case DXGI_FORMAT_R32G32B32_FLOAT: return 12u;
          case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16u;
          case DXGI_FORMAT_R16G16_FLOAT: return 4u;
          case DXGI_FORMAT_R8G8B8A8_UNORM:
          case DXGI_FORMAT_B8G8R8A8_UNORM: return 4u;
          default: return 0u;
          }
        }();
        if (element.InputSlot < ARRAYSIZE(slotOffsets))
          slotOffsets[element.InputSlot] = nativeElement.offset + elementSize;
        metadata.nativeInfo.inputElements.push_back(std::move(nativeElement));
      }
      metadata.topology = description->PrimitiveTopologyType;
      metadata.renderTargetFormat = description->NumRenderTargets
        ? description->RTVFormats[0] : DXGI_FORMAT_UNKNOWN;
      metadata.depthStencilFormat = description->DSVFormat;
      const uint32_t inputDiagnostic = g_pipelineInputDiagnostics.fetch_add(
        1, std::memory_order_relaxed);
      if (inputDiagnostic < 64) {
        for (UINT i = 0; i < description->InputLayout.NumElements; ++i) {
          const auto& element = description->InputLayout.pInputElementDescs[i];
          log("D3D12 pipeline input: sample=%u state=%p index=%u semantic=%s%u format=%u slot=%u offset=%u stride=%u",
            inputDiagnostic, pipelineState, i,
            element.SemanticName ? element.SemanticName : "", element.SemanticIndex,
            static_cast<unsigned>(element.Format), element.InputSlot,
            element.AlignedByteOffset, metadata.nativeInfo.inputElements[i].offset);
        }
      }
      g_pipelines[pipelineState] = std::move(metadata);
    }
    if (g_frame.pipelines <= 8 || (g_frame.pipelines % 1024) == 0) {
      log("D3D12 native frontend pipeline: frame=%llu state=%p result=0x%08lx vs=%zu ps=%zu inputs=%u topology=%u rtCount=%u rt0=%u dsv=%u",
        static_cast<unsigned long long>(g_frame.frame),
        pipelineState,
        static_cast<unsigned long>(result),
        description ? description->VS.BytecodeLength : 0u,
        description ? description->PS.BytecodeLength : 0u,
        description ? description->InputLayout.NumElements : 0u,
        description ? static_cast<unsigned>(description->PrimitiveTopologyType) : 0u,
        description ? description->NumRenderTargets : 0u,
        description ? static_cast<unsigned>(description->RTVFormats[0]) : 0u,
        description ? static_cast<unsigned>(description->DSVFormat) : 0u);
    }
  }

  void recordD3D12Resource(
      const D3D12_HEAP_PROPERTIES* heapProperties,
      const D3D12_RESOURCE_DESC* description,
      ID3D12Resource* resource,
      HRESULT result) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    ++g_frame.resources;
    if (SUCCEEDED(result) && resource && description) {
      g_resourceMetadata[resource] = ResourceMetadata {
        *description,
        heapProperties ? heapProperties->Type : D3D12_HEAP_TYPE_DEFAULT,
        resource->GetGPUVirtualAddress(),
      };
      const uint64_t resourceKey = reinterpret_cast<uint64_t>(resource);
      const auto oldTexture = g_textureSnapshots.find(resourceKey);
      if (oldTexture != g_textureSnapshots.end()) {
        g_textureSnapshotBytes -= oldTexture->second.bytes;
        g_textureSnapshots.erase(oldTexture);
      }
    }
    if (g_frame.resources <= 8 || (g_frame.resources % 1024) == 0) {
      log("D3D12 native frontend resource: frame=%llu resource=%p result=0x%08lx dimension=%u width=%llu height=%u mips=%u format=%u heapType=%u",
        static_cast<unsigned long long>(g_frame.frame),
        resource,
        static_cast<unsigned long>(result),
        description ? static_cast<unsigned>(description->Dimension) : 0u,
        description ? static_cast<unsigned long long>(description->Width) : 0ull,
        description ? description->Height : 0u,
        description ? description->MipLevels : 0u,
        description ? static_cast<unsigned>(description->Format) : 0u,
        heapProperties ? static_cast<unsigned>(heapProperties->Type) : 0u);
    }
  }

  void recordD3D12DescriptorHeap(
      ID3D12DescriptorHeap* heap,
      const D3D12_DESCRIPTOR_HEAP_DESC& description,
      D3D12_CPU_DESCRIPTOR_HANDLE cpuStart,
      D3D12_GPU_DESCRIPTOR_HANDLE gpuStart,
      UINT increment) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    g_descriptorHeaps[heap] = DescriptorHeapMetadata {
      cpuStart, gpuStart, increment, description.NumDescriptors, description.Type };
    log("D3D12 native frontend descriptor heap: heap=%p type=%u count=%u flags=0x%x cpu=0x%llx gpu=0x%llx increment=%u",
      heap,
      static_cast<unsigned>(description.Type),
      description.NumDescriptors,
      description.Flags,
      static_cast<unsigned long long>(cpuStart.ptr),
      static_cast<unsigned long long>(gpuStart.ptr),
      increment);
  }

  void recordD3D12View(
      const char* viewType,
      D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
      ID3D12Resource* resource,
      DXGI_FORMAT format,
      const void* viewDescription,
      UINT viewDescriptionSize) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    const auto resourceDesc = resource ? resource->GetDesc() : D3D12_RESOURCE_DESC {};
    if (format == DXGI_FORMAT_UNKNOWN && resource)
      format = resourceDesc.Format;
    DescriptorMetadata metadata;
    metadata.resource = resource;
    metadata.resourceDescription = resourceDesc;
    if (resource && resourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
      metadata.resourceGpuAddress = resource->GetGPUVirtualAddress();
    metadata.format = format;
    metadata.viewType = viewType;
    if (viewDescription && viewDescriptionSize) {
      metadata.viewDescriptionSize = std::min<UINT>(
        viewDescriptionSize, static_cast<UINT>(metadata.viewDescription.size()));
      std::memcpy(metadata.viewDescription.data(), viewDescription, metadata.viewDescriptionSize);
    }
    g_descriptors[cpuHandle.ptr] = std::move(metadata);
    log("D3D12 native frontend view: type=%s cpu=0x%llx resource=%p format=%u dimension=%u width=%llu height=%u flags=0x%x",
      viewType ? viewType : "unknown",
      static_cast<unsigned long long>(cpuHandle.ptr),
      resource,
      static_cast<unsigned>(format),
      static_cast<unsigned>(resourceDesc.Dimension),
      static_cast<unsigned long long>(resourceDesc.Width),
      resourceDesc.Height,
      static_cast<unsigned>(resourceDesc.Flags));
    if (viewDescription && viewDescriptionSize >= sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)
     && viewType && std::strcmp(viewType, "SRV") == 0) {
      const auto* srv = static_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(viewDescription);
      if (srv->ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
        log("D3D12 native frontend SRV detail: cpu=0x%llx dimension=buffer format=%u first=%llu elements=%u stride=%u",
          static_cast<unsigned long long>(cpuHandle.ptr),
          static_cast<unsigned>(srv->Format),
          static_cast<unsigned long long>(srv->Buffer.FirstElement),
          srv->Buffer.NumElements,
          srv->Buffer.StructureByteStride);
      } else {
        UINT mostDetailedMip = 0;
        UINT mipLevels = 0;
        UINT arraySize = 1;
        switch (srv->ViewDimension) {
        case D3D12_SRV_DIMENSION_TEXTURE1D:
          mostDetailedMip = srv->Texture1D.MostDetailedMip;
          mipLevels = srv->Texture1D.MipLevels;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
          mostDetailedMip = srv->Texture1DArray.MostDetailedMip;
          mipLevels = srv->Texture1DArray.MipLevels;
          arraySize = srv->Texture1DArray.ArraySize;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE2D:
          mostDetailedMip = srv->Texture2D.MostDetailedMip;
          mipLevels = srv->Texture2D.MipLevels;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
          mostDetailedMip = srv->Texture2DArray.MostDetailedMip;
          mipLevels = srv->Texture2DArray.MipLevels;
          arraySize = srv->Texture2DArray.ArraySize;
          break;
        case D3D12_SRV_DIMENSION_TEXTURECUBE:
          mostDetailedMip = srv->TextureCube.MostDetailedMip;
          mipLevels = srv->TextureCube.MipLevels;
          arraySize = 6;
          break;
        case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
          mostDetailedMip = srv->TextureCubeArray.MostDetailedMip;
          mipLevels = srv->TextureCubeArray.MipLevels;
          arraySize = srv->TextureCubeArray.NumCubes * 6;
          break;
        case D3D12_SRV_DIMENSION_TEXTURE3D:
          mostDetailedMip = srv->Texture3D.MostDetailedMip;
          mipLevels = srv->Texture3D.MipLevels;
          break;
        default:
          break;
        }
        log("D3D12 native frontend SRV detail: cpu=0x%llx dimension=%u format=%u mip=%u levels=%u array=%u",
          static_cast<unsigned long long>(cpuHandle.ptr),
          static_cast<unsigned>(srv->ViewDimension),
          static_cast<unsigned>(srv->Format),
          mostDetailedMip,
          mipLevels,
          arraySize);
      }
    }
  }

  void recordD3D12Sampler(
      D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
      const D3D12_SAMPLER_DESC* description) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    DescriptorMetadata metadata;
    metadata.viewType = "Sampler";
    metadata.sampler = true;
    if (description) {
      metadata.viewDescriptionSize = std::min<UINT>(
        sizeof(*description), static_cast<UINT>(metadata.viewDescription.size()));
      std::memcpy(metadata.viewDescription.data(), description, metadata.viewDescriptionSize);
    }
    g_descriptors[cpuHandle.ptr] = std::move(metadata);
    log("D3D12 native frontend sampler: cpu=0x%llx filter=%u address=(%u,%u,%u) lod=(%.3f,%.3f)",
      static_cast<unsigned long long>(cpuHandle.ptr),
      description ? static_cast<unsigned>(description->Filter) : 0u,
      description ? static_cast<unsigned>(description->AddressU) : 0u,
      description ? static_cast<unsigned>(description->AddressV) : 0u,
      description ? static_cast<unsigned>(description->AddressW) : 0u,
      description ? static_cast<double>(description->MinLOD) : 0.0,
      description ? static_cast<double>(description->MaxLOD) : 0.0);
  }

  void recordD3D12ConstantBufferView(
      D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle,
      const D3D12_CONSTANT_BUFFER_VIEW_DESC* description) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    const D3D12_GPU_VIRTUAL_ADDRESS location = description ? description->BufferLocation : 0;
    const UINT size = description ? description->SizeInBytes : 0;
    DescriptorMetadata metadata;
    metadata.format = DXGI_FORMAT_UNKNOWN;
    metadata.viewType = "CBV";
    metadata.bufferLocation = location;
    metadata.bufferSize = size;
    g_descriptors[cpuHandle.ptr] = std::move(metadata);
    log("D3D12 native frontend view: type=CBV cpu=0x%llx gpu=0x%llx size=%u",
      static_cast<unsigned long long>(cpuHandle.ptr),
      static_cast<unsigned long long>(location), size);
  }

  void recordD3D12CopyDescriptorsSimple(
      UINT numDescriptors,
      D3D12_CPU_DESCRIPTOR_HANDLE destDescriptorRangeStart,
      D3D12_CPU_DESCRIPTOR_HANDLE srcDescriptorRangeStart,
      UINT descriptorIncrement) {
    if (!descriptorIncrement)
      return;
    std::lock_guard lock(g_frameMutex);
    for (UINT i = 0; i < numDescriptors; ++i) {
      const uint64_t src = srcDescriptorRangeStart.ptr + static_cast<uint64_t>(i) * descriptorIncrement;
      const uint64_t dst = destDescriptorRangeStart.ptr + static_cast<uint64_t>(i) * descriptorIncrement;
      const auto it = g_descriptors.find(src);
      if (it != g_descriptors.end())
        g_descriptors[dst] = it->second;
      else
        g_descriptors.erase(dst);
    }
    log("D3D12 descriptor copy simple: count=%u src=0x%llx dst=0x%llx increment=%u",
      numDescriptors, static_cast<unsigned long long>(srcDescriptorRangeStart.ptr),
      static_cast<unsigned long long>(destDescriptorRangeStart.ptr), descriptorIncrement);
  }

  void recordD3D12CopyDescriptors(
      UINT numDestDescriptorRanges,
      const D3D12_CPU_DESCRIPTOR_HANDLE* destDescriptorRangeStarts,
      const UINT* destDescriptorRangeSizes,
      UINT numSrcDescriptorRanges,
      const D3D12_CPU_DESCRIPTOR_HANDLE* srcDescriptorRangeStarts,
      const UINT* srcDescriptorRangeSizes,
      UINT descriptorIncrement) {
    if (!descriptorIncrement || !destDescriptorRangeStarts || !destDescriptorRangeSizes
     || !srcDescriptorRangeStarts || !srcDescriptorRangeSizes)
      return;

    UINT destRange = 0;
    UINT srcRange = 0;
    UINT destOffset = 0;
    UINT srcOffset = 0;
    std::lock_guard lock(g_frameMutex);
    while (destRange < numDestDescriptorRanges && srcRange < numSrcDescriptorRanges) {
      const UINT destRemaining = destDescriptorRangeSizes[destRange] - destOffset;
      const UINT srcRemaining = srcDescriptorRangeSizes[srcRange] - srcOffset;
      const UINT count = std::min(destRemaining, srcRemaining);
      for (UINT i = 0; i < count; ++i) {
        const uint64_t src = srcDescriptorRangeStarts[srcRange].ptr
          + static_cast<uint64_t>(srcOffset + i) * descriptorIncrement;
        const uint64_t dst = destDescriptorRangeStarts[destRange].ptr
          + static_cast<uint64_t>(destOffset + i) * descriptorIncrement;
        const auto it = g_descriptors.find(src);
        if (it != g_descriptors.end())
          g_descriptors[dst] = it->second;
        else
          g_descriptors.erase(dst);
      }
      destOffset += count;
      srcOffset += count;
      if (destOffset == destDescriptorRangeSizes[destRange]) {
        ++destRange;
        destOffset = 0;
      }
      if (srcOffset == srcDescriptorRangeSizes[srcRange]) {
        ++srcRange;
        srcOffset = 0;
      }
    }
    log("D3D12 descriptor copy: destRanges=%u srcRanges=%u increment=%u",
      numDestDescriptorRanges, numSrcDescriptorRanges, descriptorIncrement);
  }

  void recordD3D12ResourceBarrier(
      UINT count,
      const D3D12_RESOURCE_BARRIER* barriers) {
    initializeD3D12RuntimeAdapter();
    if (!barriers)
      return;
    for (UINT i = 0; i < count; ++i) {
      const auto& barrier = barriers[i];
      if (barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
        recordResourceBarrier(
          barrier.Transition.pResource,
          barrier.Transition.StateBefore,
          barrier.Transition.StateAfter,
          barrier.Transition.Subresource);
      } else if (barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
        std::lock_guard lock(g_frameMutex);
        const ID3D12Resource* resources[] = {
          barrier.Aliasing.pResourceBefore,
          barrier.Aliasing.pResourceAfter,
        };
        for (const ID3D12Resource* resource : resources) {
          if (!resource)
            continue;
          const uint64_t resourceKey = reinterpret_cast<uint64_t>(resource);
          const auto texture = g_textureSnapshots.find(resourceKey);
          if (texture != g_textureSnapshots.end()) {
            g_textureSnapshotBytes -= texture->second.bytes;
            g_textureSnapshots.erase(texture);
          }
        }
        if (!barrier.Aliasing.pResourceBefore || !barrier.Aliasing.pResourceAfter) {
          g_textureSnapshots.clear();
          g_textureSnapshotBytes = 0;
        }
        log("D3D12 texture snapshot invalidated by aliasing barrier: before=%p after=%p",
          barrier.Aliasing.pResourceBefore, barrier.Aliasing.pResourceAfter);
      }
    }
  }

  void recordD3D12CopyTextureRegion(
      const D3D12_TEXTURE_COPY_LOCATION* destination,
      const D3D12_TEXTURE_COPY_LOCATION* source,
      UINT destinationX,
      UINT destinationY,
      UINT destinationZ,
      const D3D12_BOX* sourceBox) {
    initializeD3D12RuntimeAdapter();
    if (!destination || !source)
      return;
    if (!kEnableTextureSnapshotCapture)
      return;
    const uint64_t textureStartTicks = perfTicks();
    if (kEnableTextureCopyDiagnostics) {
      log("D3D12 native frontend texture copy: dst=%p subresource=%u src=%p subresource=%u dstOffset=(%u,%u,%u) sourceBox=%s",
        destination->pResource,
        destination->Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX ? destination->SubresourceIndex : 0u,
        source->pResource,
        source->Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX ? source->SubresourceIndex : 0u,
        destinationX, destinationY, destinationZ, sourceBox ? "set" : "full");
    }
    if (source->Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
      const auto& footprint = source->PlacedFootprint;
      if (kEnableTextureCopyDiagnostics) {
        log("D3D12 native frontend texture footprint: offset=%llu rowPitch=%u rows=%u depth=%u",
          static_cast<unsigned long long>(footprint.Offset),
          footprint.Footprint.RowPitch,
          footprint.Footprint.Height,
          footprint.Footprint.Depth);
      }
      captureTextureUpload(
        *destination, *source, destinationX, destinationY, destinationZ, sourceBox);
    }
    g_perfTextureTicks.fetch_add(
      perfTicks() - textureStartTicks, std::memory_order_relaxed);
    g_perfTextureCalls.fetch_add(1, std::memory_order_relaxed);
  }

  // The flight camera block pairs a backward third basis row with a right
  // handed depth row, which is the form the scene was authored for. Remix needs
  // the projection's vertical scale positive while the scene stores it negative,
  // so that axis is flipped here. Both stay switchable at runtime in case the
  // hangar scene differs: F9 cycles the depth row, F8 flips the vertical axis.
  std::atomic<uint32_t> g_cameraDepthReversed { 1 };
  std::atomic<uint32_t> g_cameraFlipVertical { 1 };

  void pollCameraConventionKeys() {
    static bool depthKeyDown = false;
    static bool verticalKeyDown = false;

    const bool depthKey = (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
    if (depthKey && !depthKeyDown) {
      const bool reversed = g_cameraDepthReversed.load(std::memory_order_relaxed) != 0;
      g_cameraDepthReversed.store(reversed ? 0u : 1u, std::memory_order_relaxed);
      log("ACZ camera convention: depth row=%s",
        reversed ? "third row forward" : "third row backward");
    }
    depthKeyDown = depthKey;

    const bool verticalKey = (GetAsyncKeyState(VK_F8) & 0x8000) != 0;
    if (verticalKey && !verticalKeyDown) {
      const bool flipped = g_cameraFlipVertical.load(std::memory_order_relaxed) != 0;
      g_cameraFlipVertical.store(flipped ? 0u : 1u, std::memory_order_relaxed);
      log("ACZ camera convention: vertical scale=%s",
        flipped ? "negative" : "positive");
    }
    verticalKeyDown = verticalKey;
  }

  struct AczCameraMatrices {
    std::array<float, 16> view {};
    std::array<float, 16> projection {};
    float eye[3] {};
    float third[3] {};
    float nearPlane = 0.0f;
    float farPlane = 0.0f;
    float p00 = 0.0f;
    float p11 = 0.0f;
    float depthSign = 0.0f;
    float verticalSign = 0.0f;
  };

  // Rebuilds the camera the scene is rendered with from the game's own camera
  // block: three view basis rows at +0x50, the eye at +0x30 and the row-vector
  // projection at +0xD0. Returns false before the block has been captured.
  bool buildAczCamera(AczCameraMatrices& out) {
    AczCameraProbe probe;
    if (!readAczCameraProbe(probe) || !probe.transformArgs[2])
      return false;

    const auto* pCamera = reinterpret_cast<const float*>(probe.transformArgs[2]);
    MEMORY_BASIC_INFORMATION cameraInfo {};
    if (!VirtualQuery(pCamera, &cameraInfo, sizeof(cameraInfo))
     || cameraInfo.State != MEM_COMMIT)
      return false;

    const auto* pViewRows = reinterpret_cast<const float*>(
      reinterpret_cast<const uint8_t*>(pCamera) + 0x50);
    const float eyeX = pCamera[12];
    const float eyeY = pCamera[13];
    const float eyeZ = pCamera[14];
    // The block's third basis row is the view direction: a point in front of the
    // eye lands on a view space Z of the same sign as that row, which is what
    // keeps clip W positive in front once the projection carries the scene's own
    // depth row.
    const float forwardX = pViewRows[8];
    const float forwardY = pViewRows[9];
    const float forwardZ = pViewRows[10];
    const float forwardLength = std::sqrt(
      forwardX * forwardX + forwardY * forwardY + forwardZ * forwardZ);
    if (!std::isfinite(eyeX) || !std::isfinite(eyeY) || !std::isfinite(eyeZ)
     || !std::isfinite(forwardLength) || forwardLength < 0.5f)
      return false;

    // One-off dump of the whole block so the basis and depth conventions can
    // be settled from a single run.
    {
      static uint32_t blockDumps = 0;
      const auto* pFull = reinterpret_cast<const uint8_t*>(pCamera);
      MEMORY_BASIC_INFORMATION fullInfo {};
      if (blockDumps < 2
       && VirtualQuery(pFull, &fullInfo, sizeof(fullInfo))
       && fullInfo.State == MEM_COMMIT
       && reinterpret_cast<uintptr_t>(pFull) + 0x120
          <= reinterpret_cast<uintptr_t>(fullInfo.BaseAddress) + fullInfo.RegionSize) {
        ++blockDumps;
        float full[72] = {};
        std::memcpy(full, pFull, sizeof(full));
        for (uint32_t row = 0; row < 18; ++row) {
          log("ACZ flight block +0x%03X: %13.4f %13.4f %13.4f %13.4f", row * 16,
            static_cast<double>(full[row * 4 + 0]),
            static_cast<double>(full[row * 4 + 1]),
            static_cast<double>(full[row * 4 + 2]),
            static_cast<double>(full[row * 4 + 3]));
        }
      }
    }

    // The viewport constants follow the two matrices as a row-vector
    // projection, so they carry the scene's own near and far planes and the
    // depth row orientation.
    float gameProjection[16] = {};
    bool hasGameProjection = false;
    {
      const auto* pProjectionBytes = reinterpret_cast<const uint8_t*>(pCamera) + 0xD0;
      MEMORY_BASIC_INFORMATION projectionInfo {};
      if (VirtualQuery(pProjectionBytes, &projectionInfo, sizeof(projectionInfo))
       && projectionInfo.State == MEM_COMMIT
       && reinterpret_cast<uintptr_t>(pProjectionBytes) + sizeof(gameProjection)
          <= reinterpret_cast<uintptr_t>(projectionInfo.BaseAddress) + projectionInfo.RegionSize) {
        std::memcpy(gameProjection, pProjectionBytes, sizeof(gameProjection));
        bool projectionFinite = true;
        for (float value : gameProjection)
          projectionFinite = projectionFinite && std::isfinite(value);
        hasGameProjection = projectionFinite;
      }
    }

    // The block's three rows already form the view basis, so they are used
    // verbatim. The block carries the eye the scene renders from, so it is used
    // as-is: moving it away from that point scales every camera translation,
    // which shows up as exaggerated pitch motion.
    const float rightLength = std::sqrt(
      pViewRows[0] * pViewRows[0] + pViewRows[1] * pViewRows[1] + pViewRows[2] * pViewRows[2]);
    const float upLength = std::sqrt(
      pViewRows[4] * pViewRows[4] + pViewRows[5] * pViewRows[5] + pViewRows[6] * pViewRows[6]);
    if (!std::isfinite(rightLength) || !std::isfinite(upLength)
     || rightLength < 0.5f || upLength < 0.5f)
      return false;

    // The basis is normalized and the translation rebuilt from the eye, which is
    // exactly what the backend does before handing the matrix to Remix. The fused
    // decode cancels this same matrix, so any difference between the two would
    // show up as the placement being scaled away from the camera.
    const float rx = pViewRows[0] / rightLength;
    const float ry = pViewRows[1] / rightLength;
    const float rz = pViewRows[2] / rightLength;
    const float ux = pViewRows[4] / upLength;
    const float uy = pViewRows[5] / upLength;
    const float uz = pViewRows[6] / upLength;
    const float fx = forwardX / forwardLength;
    const float fy = forwardY / forwardLength;
    const float fz = forwardZ / forwardLength;

    out.eye[0] = eyeX;
    out.eye[1] = eyeY;
    out.eye[2] = eyeZ;
    out.third[0] = fx;
    out.third[1] = fy;
    out.third[2] = fz;
    out.view = {
      rx, ry, rz, 0.0f,
      ux, uy, uz, 0.0f,
      fx, fy, fz, 0.0f,
      -(rx * eyeX + ry * eyeY + rz * eyeZ),
      -(ux * eyeX + uy * eyeY + uz * eyeZ),
      -(fx * eyeX + fy * eyeY + fz * eyeZ),
      1.0f,
    };

    // The depth terms come from the game's own projection, so the near and far
    // planes match the scene the geometry was authored for.
    float nearPlane = 11.5254f;
    float farPlane = 62993.0f;
    float p00 = 1.786676f;
    float p11 = 3.176313f;
    if (hasGameProjection) {
      const float m00 = std::abs(gameProjection[0]);
      const float m11 = std::abs(gameProjection[5]);
      const float m22 = std::abs(gameProjection[10]);
      const float m32 = std::abs(gameProjection[14]);
      // A non-reversed depth mapping satisfies |m32| = near * |m22| and
      // |m22| = far / (far - near).
      if (m00 > 0.1f && m11 > 0.1f && m22 > 1.0f && m32 > 0.0f) {
        const float derivedNear = m32 / m22;
        const float derivedFar = derivedNear * m22 / (m22 - 1.0f);
        if (derivedNear > 0.0f && derivedFar > derivedNear * 2.0f
         && derivedFar < 5.0e6f) {
          p00 = m00;
          p11 = m11;
          nearPlane = derivedNear;
          farPlane = derivedFar;
        }
      }
    }

    // A point in front of the eye lands on a view space Z whose sign is
    // opposite to the third row, so the depth row is negated to keep clip W
    // positive in front.
    const bool depthReversed = g_cameraDepthReversed.load(std::memory_order_relaxed) != 0;
    const float depthSign = depthReversed ? -1.0f : 1.0f;
    const float verticalSign =
      g_cameraFlipVertical.load(std::memory_order_relaxed) != 0 ? 1.0f : -1.0f;
    const float p22 = farPlane / (farPlane - nearPlane);
    const float pB = -nearPlane * p22;

    out.projection = {
      p00, 0.0f, 0.0f, 0.0f,
      0.0f, verticalSign * p11, 0.0f, 0.0f,
      0.0f, 0.0f, depthSign * p22, depthSign,
      0.0f, 0.0f, pB, 0.0f,
    };
    out.nearPlane = nearPlane;
    out.farPlane = farPlane;
    out.p00 = p00;
    out.p11 = p11;
    out.depthSign = depthSign;
    out.verticalSign = verticalSign;
    return true;
  }

  // Reasons a fused draw is not placed, each counted separately so one frequent
  // failure does not hide the others.
  enum FusedRejectReason : uint32_t {
    kFusedNoBuffer,
    kFusedUnreadableMatrix,
    kFusedIdentityMatrix,
    kFusedNoCamera,
    kFusedCameraNotInvertible,
    kFusedNonFinite,
    kFusedNotAffine,
    kFusedNoTranslation,
    kFusedImplausibleScale,
    kFusedOutOfRange,
    kFusedRejectReasonCount,
  };

  const char* fusedReasonText(FusedRejectReason reason) {
    switch (reason) {
    case kFusedNoBuffer: return "no draw buffer";
    case kFusedUnreadableMatrix: return "unreadable M0";
    case kFusedIdentityMatrix: return "identity M0";
    case kFusedNoCamera: return "no camera";
    case kFusedCameraNotInvertible: return "camera not invertible";
    case kFusedNonFinite: return "non-finite result";
    case kFusedNotAffine: return "not affine";
    case kFusedNoTranslation: return "no translation";
    case kFusedImplausibleScale: return "implausible scale";
    case kFusedOutOfRange: return "placement out of range";
    default: return "unknown";
    }
  }

  // Turns one candidate matrix into a world placement by cancelling the camera
  // out of it. Silent, so a search over many candidates can call it freely.
  bool composeFusedPlacement(
      const std::array<float, 16>& objectToWorld,
      const std::array<float, 16>& modelToClip,
      const std::array<float, 16>& inverseViewProjection,
      const float eye[3],
      std::array<float, 16>& placed,
      std::array<float, 16>& result,
      FusedRejectReason& reason) {
    multiplyRowVectorMatrices(objectToWorld.data(), modelToClip.data(), placed.data());
    multiplyRowVectorMatrices(placed.data(), inverseViewProjection.data(), result.data());

    for (float value : result) {
      if (!std::isfinite(value) || std::abs(value) > 1.0e7f) {
        reason = kFusedNonFinite;
        return false;
      }
    }

    // An affine placement has no perspective terms and a unit last component.
    if (std::abs(result[3]) > 0.02f || std::abs(result[7]) > 0.02f
     || std::abs(result[11]) > 0.02f || std::abs(result[15] - 1.0f) > 0.02f) {
      reason = kFusedNotAffine;
      return false;
    }

    const float tx = result[12];
    const float ty = result[13];
    const float tz = result[14];
    const float translationLength = std::sqrt(tx * tx + ty * ty + tz * tz);
    // Cancelling the camera out of a matrix that did not contain it leaves the
    // identity behind, which is not a placement.
    if (translationLength < 1.0f) {
      reason = kFusedNoTranslation;
      return false;
    }

    const float n0 = std::sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
    const float n1 = std::sqrt(result[4] * result[4] + result[5] * result[5] + result[6] * result[6]);
    const float n2 = std::sqrt(result[8] * result[8] + result[9] * result[9] + result[10] * result[10]);
    if (n0 < 1.0e-3f || n0 > 1.0e4f || n1 < 1.0e-3f || n1 > 1.0e4f
     || n2 < 1.0e-3f || n2 > 1.0e4f) {
      reason = kFusedImplausibleScale;
      return false;
    }

    // The placement has to land in the scene the camera looks at, otherwise the
    // leading matrix belonged to a different pass.
    const float dx = tx - eye[0];
    const float dy = ty - eye[1];
    const float dz = tz - eye[2];
    if (std::sqrt(dx * dx + dy * dy + dz * dz) > 4.0e5f) {
      reason = kFusedOutOfRange;
      return false;
    }

    return true;
  }

  // Model draws report an identity object matrix because their vertex shader
  // multiplies the position by the leading matrix of the structured buffer at
  // +0x000, which already contains the placement and the camera together. The
  // game therefore computes clip = v * M4 * M0 while Remix renders
  // v * objectToWorld * worldToView * viewToProjection, so the placement is
  // recovered by cancelling that camera out:
  //   objectToWorld = M4 * M0 * (worldToView * viewToProjection)^-1
  // A camera mismatch shows up as a non-affine result and is rejected.
  bool decodeFusedModelTransform(
      D3D12_GPU_VIRTUAL_ADDRESS buffer,
      UINT64 bufferWidth,
      const std::array<float, 16>& objectToWorld,
      D3D12_GPU_VIRTUAL_ADDRESS vertexBuffer,
      UINT32 vertexStride,
      std::array<float, 16>& result) {
    const auto reject = [](FusedRejectReason reason) {
      static std::atomic<uint32_t> counters[kFusedRejectReasonCount];
      const uint32_t count = counters[reason].fetch_add(1, std::memory_order_relaxed);
      if (count < 4) {
        log("ACZ fused model transform rejected: reason=%s count=%u",
          fusedReasonText(reason), count + 1);
      }
      return false;
    };

    if (!buffer || bufferWidth < 0x40)
      return reject(kFusedNoBuffer);

    std::array<float, 16> modelToClip {};
    if (!readMatrixFromShaderBuffer(buffer, bufferWidth, 0x000, modelToClip))
      return reject(kFusedUnreadableMatrix);
    if (isIdentityObjectTransform(modelToClip))
      return reject(kFusedIdentityMatrix);

    AczCameraMatrices camera {};
    if (!buildAczCamera(camera))
      return reject(kFusedNoCamera);

    std::array<float, 16> viewProjection {};
    multiplyRowVectorMatrices(camera.view.data(), camera.projection.data(), viewProjection.data());
    std::array<float, 16> inverseViewProjection {};
    if (!invertMatrix4(viewProjection.data(), inverseViewProjection.data()))
      return reject(kFusedCameraNotInvertible);

    std::array<float, 16> placed {};
    FusedRejectReason failure = kFusedRejectReasonCount;
    if (!composeFusedPlacement(objectToWorld, modelToClip, inverseViewProjection,
          camera.eye, placed, result, failure))
      return reject(failure);

    {
      static std::atomic<uint32_t> accepts { 0 };
      const uint32_t index = accepts.fetch_add(1, std::memory_order_relaxed);
      if (index < 12) {
        // Verify the composition by reproducing the game's own clip matrix and
        // comparing it with the one Remix will build.
        std::array<float, 16> remixComposed {};
        multiplyRowVectorMatrices(result.data(), viewProjection.data(), remixComposed.data());
        float worstError = 0.0f;
        for (uint32_t i = 0; i < 16; ++i)
          worstError = std::max(worstError, std::abs(remixComposed[i] - placed[i]));

        const float determinant =
            result[0] * (result[5] * result[10] - result[6] * result[9])
          - result[1] * (result[4] * result[10] - result[6] * result[8])
          + result[2] * (result[4] * result[9] - result[5] * result[8]);
        const float row0 = std::sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
        const float row1 = std::sqrt(result[4] * result[4] + result[5] * result[5] + result[6] * result[6]);
        const float row2 = std::sqrt(result[8] * result[8] + result[9] * result[9] + result[10] * result[10]);
        const float translationLength = std::sqrt(
          result[12] * result[12] + result[13] * result[13] + result[14] * result[14]);
        const float originW = modelToClip[15];
        const float originX = std::abs(originW) > 1.0e-6f ? modelToClip[12] / originW : 0.0f;
        const float originY = std::abs(originW) > 1.0e-6f ? modelToClip[13] / originW : 0.0f;

        log("ACZ fused model transform: sample=%u t=(%.3f,%.3f,%.3f) translation=%.1f rows=(%.4f,%.4f,%.4f) det=%.4f originW=%.4f originNdc=(%.3f,%.3f) maxError=%.3g",
          index,
          static_cast<double>(result[12]),
          static_cast<double>(result[13]),
          static_cast<double>(result[14]),
          static_cast<double>(translationLength),
          static_cast<double>(row0), static_cast<double>(row1), static_cast<double>(row2),
          static_cast<double>(determinant),
          static_cast<double>(originW),
          static_cast<double>(originX), static_cast<double>(originY),
          static_cast<double>(worstError));

        if (index < 2) {
          // The draw buffer holds one element per object, so the shader may pick
          // an element through a constant instead of always reading element 0.
          for (uint32_t slot = 0; slot <= 2; ++slot) {
            const auto constantsAddress = currentGraphicsRootCbv(slot);
            if (!constantsAddress)
              continue;
            std::vector<uint8_t> constants;
            if (readGpuBuffer(constantsAddress, sizeof(uint32_t) * 4, constants)
             && constants.size() >= sizeof(uint32_t) * 2) {
              const auto* words = reinterpret_cast<const uint32_t*>(constants.data());
              log("ACZ fused model constant: root=%u address=0x%llx words=(%u,%u,%u,%u)",
                slot, static_cast<unsigned long long>(constantsAddress),
                words[0], words[1], words[2], words[3]);
            }
          }

          for (uint32_t row = 0; row < 4; ++row) {
            log("ACZ fused model M0 r%u: %12.5f %12.5f %12.5f %12.5f", row,
              static_cast<double>(modelToClip[row * 4 + 0]),
              static_cast<double>(modelToClip[row * 4 + 1]),
              static_cast<double>(modelToClip[row * 4 + 2]),
              static_cast<double>(modelToClip[row * 4 + 3]));
          }
        }

        // A single vertex proves the composed matrix reproduces the position the
        // game's shader would send to the rasterizer.
        if (index < 2 && vertexBuffer && vertexStride >= sizeof(float) * 3) {
          std::vector<uint8_t> vertexBytes;
          if (readGpuBuffer(vertexBuffer, vertexStride, vertexBytes)
           && vertexBytes.size() >= sizeof(float) * 3) {
            const auto* position = reinterpret_cast<const float*>(vertexBytes.data());
            const float vertex[4] = { position[0], position[1], position[2], 1.0f };
            float gameClip[4] = {};
            float remixClip[4] = {};
            for (uint32_t column = 0; column < 4; ++column) {
              for (uint32_t row = 0; row < 4; ++row) {
                gameClip[column] += vertex[row] * placed[row * 4 + column];
                remixClip[column] += vertex[row] * remixComposed[row * 4 + column];
              }
            }
            log("ACZ fused model check: v=(%.3f,%.3f,%.3f) gameClip=(%.4f,%.4f,%.4f,%.4f) remixClip=(%.4f,%.4f,%.4f,%.4f)",
              static_cast<double>(vertex[0]), static_cast<double>(vertex[1]),
              static_cast<double>(vertex[2]),
              static_cast<double>(gameClip[0]), static_cast<double>(gameClip[1]),
              static_cast<double>(gameClip[2]), static_cast<double>(gameClip[3]),
              static_cast<double>(remixClip[0]), static_cast<double>(remixClip[1]),
              static_cast<double>(remixClip[2]), static_cast<double>(remixClip[3]));
          }
        }
      }
    }

    return true;
  }

  // Collects the buffer views bound to a draw. The model vertex shader reads its
  // transform data from a structured buffer that the draw trace cannot map onto a
  // root parameter, so the bound tables are scanned instead and the views are
  // validated by content rather than by binding.
  void collectDrawBufferCandidates(
      const D3D12_GPU_DESCRIPTOR_HANDLE* tables,
      uint64_t drawId,
      std::vector<D3D12_GPU_VIRTUAL_ADDRESS>& addresses,
      std::vector<UINT64>& widths) {
    if (!tables)
      return;

    // The transform buffers sit past the texture range of the first table, which
    // already holds more than sixteen views.
    constexpr uint32_t kDescriptorOffsets = 96;
    constexpr uint32_t kMaxCandidates = 16;
    constexpr size_t kMaxTables = 8;

    static std::atomic<uint32_t> reports { 0 };
    const bool report = reports.fetch_add(1, std::memory_order_relaxed) < 2;

    std::lock_guard lock(g_frameMutex);
    for (size_t table = 0; table < kMaxTables && addresses.size() < kMaxCandidates; ++table) {
      const uint64_t tableGpu = tables[table].ptr;
      if (!tableGpu)
        continue;

      const DescriptorHeapMetadata* heap = nullptr;
      for (const auto& entry : g_descriptorHeaps) {
        const auto& metadata = entry.second;
        if (!metadata.gpuStart.ptr || !metadata.increment || tableGpu < metadata.gpuStart.ptr)
          continue;
        const uint64_t delta = tableGpu - metadata.gpuStart.ptr;
        if (delta % metadata.increment)
          continue;
        if (delta / metadata.increment >= metadata.count)
          continue;
        heap = &metadata;
        break;
      }
      if (!heap)
        continue;

      for (uint32_t offset = 0; offset < kDescriptorOffsets && addresses.size() < kMaxCandidates; ++offset) {
        const uint64_t descriptorGpu = tableGpu + static_cast<uint64_t>(offset) * heap->increment;
        const uint64_t descriptorDelta = descriptorGpu - heap->gpuStart.ptr;
        if (descriptorDelta % heap->increment)
          continue;
        const uint64_t index = descriptorDelta / heap->increment;
        if (index >= heap->count)
          break;
        const uint64_t cpu = heap->cpuStart.ptr + index * heap->increment;
        const auto descriptor = g_descriptors.find(cpu);
        if (descriptor == g_descriptors.end())
          continue;

        const DescriptorMetadata& metadata = descriptor->second;
        D3D12_GPU_VIRTUAL_ADDRESS address = 0;
        UINT64 width = 0;
        const char* kind = nullptr;
        if (metadata.viewType && std::strcmp(metadata.viewType, "CBV") == 0) {
          address = metadata.bufferLocation;
          width = metadata.bufferSize ? metadata.bufferSize : 0x1000;
          kind = "cbv";
        } else if (metadata.viewType && std::strcmp(metadata.viewType, "SRV") == 0
                && metadata.resource
                && metadata.resourceDescription.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
          address = metadata.resourceGpuAddress;
          width = metadata.resourceDescription.Width;
          kind = "srv";
        }
        if (!address || width < 0x40)
          continue;

        bool duplicate = false;
        for (const auto existing : addresses) {
          if (existing == address) {
            duplicate = true;
            break;
          }
        }
        if (duplicate)
          continue;

        addresses.push_back(address);
        widths.push_back(width);
        if (!report)
          continue;
        log("ACZ draw buffer candidate: draw=%llu table=%zu offset=%u kind=%s gpu=0x%llx width=%llu",
          drawId, table, offset, kind,
          static_cast<unsigned long long>(address),
          static_cast<unsigned long long>(width));
        for (uint32_t row = 0; row < 2; ++row) {
          std::vector<uint8_t> bytes;
          if (!readGpuBuffer(address + row * 0x100, 16, bytes) || bytes.size() < 16) {
            log("ACZ draw buffer candidate +0x%03X: unreadable", row * 0x100);
            continue;
          }
          const auto* values = reinterpret_cast<const float*>(bytes.data());
          log("ACZ draw buffer candidate +0x%03X: %12.5g %12.5g %12.5g %12.5g", row * 0x100,
            static_cast<double>(values[0]), static_cast<double>(values[1]),
            static_cast<double>(values[2]), static_cast<double>(values[3]));
        }
      }
    }
  }

  // Decodes one already-read matrix into a placement. Silent, so several
  // candidates can be tried in one draw.
  bool composeFusedMatrix(
      const std::array<float, 16>& modelToClip,
      const std::array<float, 16>& objectToWorld,
      std::array<float, 16>& result) {
    if (isIdentityObjectTransform(modelToClip))
      return false;

    AczCameraMatrices camera {};
    if (!buildAczCamera(camera))
      return false;

    std::array<float, 16> viewProjection {};
    multiplyRowVectorMatrices(camera.view.data(), camera.projection.data(), viewProjection.data());
    std::array<float, 16> inverseViewProjection {};
    if (!invertMatrix4(viewProjection.data(), inverseViewProjection.data()))
      return false;

    std::array<float, 16> placed {};
    FusedRejectReason failure = kFusedRejectReasonCount;
    return composeFusedPlacement(objectToWorld, modelToClip, inverseViewProjection,
      camera.eye, placed, result, failure);
  }

  // Reads one candidate matrix from a known offset of a constant buffer and
  // decodes it into a placement. Silent, so several offsets and arenas can be
  // tried in one draw.
  bool decodeFusedModelMatrixAt(
      D3D12_GPU_VIRTUAL_ADDRESS buffer,
      UINT64 bufferWidth,
      uint32_t offset,
      const std::array<float, 16>& objectToWorld,
      std::array<float, 16>& result) {
    if (!buffer || bufferWidth < offset + sizeof(float) * 16)
      return false;

    std::array<float, 16> modelToClip {};
    if (!readMatrixFromShaderBuffer(buffer, bufferWidth, offset, modelToClip))
      return false;

    return composeFusedMatrix(modelToClip, objectToWorld, result);
  }

  // Enumerates the tracked buffer views and reports their leading content, so the
  // structured buffer that carries the fused matrices can be located by what it
  // holds rather than by which root parameter binds it. The draw's own tables are
  // already known not to carry it, so this walks every view the layer has seen.
  void reportTrackedBufferViews(uint64_t drawId, uint64_t frameId, bool forced) {
    static std::atomic<uint32_t> reports { 0 };
    if (!forced && reports.fetch_add(1, std::memory_order_relaxed) >= 2)
      return;

    struct Candidate {
      D3D12_GPU_VIRTUAL_ADDRESS address;
      UINT64 width;
    };
    std::vector<Candidate> candidates;
    {
      std::lock_guard lock(g_frameMutex);
      for (const auto& entry : g_descriptors) {
        const DescriptorMetadata& metadata = entry.second;
        if (!metadata.viewType || std::strcmp(metadata.viewType, "SRV") != 0)
          continue;
        if (metadata.resourceDescription.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
          continue;
        if (!metadata.resourceGpuAddress || metadata.resourceDescription.Width < 0x100)
          continue;

        UINT64 firstElementOffset = 0;
        if (metadata.viewDescriptionSize >= sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) {
          const auto* srv = reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(
            metadata.viewDescription.data());
          if (srv->ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
            firstElementOffset = static_cast<UINT64>(srv->Buffer.FirstElement)
              * srv->Buffer.StructureByteStride;
          }
        }
        if (metadata.resourceDescription.Width <= firstElementOffset)
          continue;

        candidates.push_back({
          metadata.resourceGpuAddress + firstElementOffset,
          metadata.resourceDescription.Width - firstElementOffset });
        if (candidates.size() >= 24)
          break;
      }
    }

    log("ACZ tracked buffer views: draw=%llu frame=%llu count=%zu",
      drawId, frameId, candidates.size());
    for (const Candidate& candidate : candidates) {
      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(candidate.address, 16, bytes) || bytes.size() < 16) {
        log("ACZ tracked buffer: gpu=0x%llx width=%llu unreadable",
          static_cast<unsigned long long>(candidate.address),
          static_cast<unsigned long long>(candidate.width));
        continue;
      }
      const auto* values = reinterpret_cast<const float*>(bytes.data());
      log("ACZ tracked buffer: gpu=0x%llx width=%llu first=(%.6g,%.6g,%.6g,%.6g)",
        static_cast<unsigned long long>(candidate.address),
        static_cast<unsigned long long>(candidate.width),
        static_cast<double>(values[0]), static_cast<double>(values[1]),
        static_cast<double>(values[2]), static_cast<double>(values[3]));
    }
  }

  void recordD3D12Draw(
      ID3D12GraphicsCommandList* commandList,
      const D3D12_VERTEX_BUFFER_VIEW& vertexBuffer,
      const D3D12_INDEX_BUFFER_VIEW& indexBuffer,
      const D3D12_GPU_DESCRIPTOR_HANDLE* graphicsDescriptorTables,
      D3D12_PRIMITIVE_TOPOLOGY topology,
      ID3D12PipelineState* pipelineState,
      UINT indexCount,
      UINT instanceCount,
      UINT startIndexLocation,
      INT baseVertexLocation,
      UINT startInstanceLocation,
      D3D12_GPU_VIRTUAL_ADDRESS objectTransformCbv,
      UINT vertexCount,
      const D3D12_GPU_VIRTUAL_ADDRESS* graphicsRootSrvs) {
    initializeD3D12RuntimeAdapter();
    const uint64_t drawStartTicks = perfTicks();
    const uint64_t count = ++g_drawCount;
    uint64_t drawFrame = 0;
    if (count <= 24 || (count % 1024) == 0) {
      log("D3D12 binding state: count=%llu cbv2=0x%llx tables=(0x%llx,0x%llx,0x%llx,0x%llx)",
        static_cast<unsigned long long>(count),
        static_cast<unsigned long long>(objectTransformCbv),
        graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[0].ptr) : 0ull,
        graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[1].ptr) : 0ull,
        graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[2].ptr) : 0ull,
        graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[3].ptr) : 0ull);
    }
    {
      std::lock_guard lock(g_frameMutex);
      ++g_frame.draws;
      drawFrame = g_frame.frame;
      if (indexBuffer.BufferLocation && indexCount)
        ++g_frame.indexedDraws;
      else
        ++g_frame.instancedDraws;

      const auto pipeline = g_pipelines.find(pipelineState);
      const uint64_t bindingHash = drawBindingHash(
        pipelineState, vertexBuffer, indexBuffer, graphicsDescriptorTables);
      const bool firstBinding = kEnableDrawDiagnostics && g_loggedDrawBindings.size() < 4096
        && g_loggedDrawBindings.emplace(bindingHash).second;
      if (kEnableDrawDiagnostics && (shouldLog(count) || firstBinding)) {
        uint64_t resolvedResources[4] = {};
        uint32_t resolvedFormats[4] = {};
        uint64_t resolvedBufferLocations[4] = {};
        uint32_t resolvedBufferSizes[4] = {};
        const ResourceMetadata* resolvedMetadata[4] = {};
        for (size_t table = 0; table < 4; ++table) {
          const uint64_t gpu = graphicsDescriptorTables ? graphicsDescriptorTables[table].ptr : 0;
          if (!gpu)
            continue;
          for (const auto& heapEntry : g_descriptorHeaps) {
            const auto& metadata = heapEntry.second;
            if (!metadata.gpuStart.ptr || !metadata.increment || gpu < metadata.gpuStart.ptr)
              continue;
            const uint64_t delta = gpu - metadata.gpuStart.ptr;
            const uint64_t index = delta / metadata.increment;
            if (delta % metadata.increment || index >= metadata.count)
              continue;
            const uint64_t cpu = metadata.cpuStart.ptr + index * metadata.increment;
            const auto descriptor = g_descriptors.find(cpu);
            if (descriptor != g_descriptors.end()) {
              resolvedResources[table] = reinterpret_cast<uint64_t>(descriptor->second.resource);
              resolvedFormats[table] = static_cast<uint32_t>(descriptor->second.format);
              resolvedBufferLocations[table] = descriptor->second.bufferLocation;
              resolvedBufferSizes[table] = descriptor->second.bufferSize;
              resolvedMetadata[table] = findResourceMetadata(descriptor->second.resource);
            }
            break;
          }
        }

        // Model vertex shaders use a structured SRV from a descriptor
        // range, so the table base is not necessarily the descriptor consumed
        // by t10. Inspect the first 32 descriptors in each bound table and
        // capture buffer-backed SRVs that can contain transform matrices.
        if (firstBinding && graphicsDescriptorTables) {
          for (size_t table = 0; table < 4; ++table) {
            const uint64_t baseGpu = graphicsDescriptorTables[table].ptr;
            if (!baseGpu)
              continue;
            for (size_t offset = 0; offset < 32; ++offset) {
              const uint64_t gpu = baseGpu + offset * 32;
              for (const auto& heapEntry : g_descriptorHeaps) {
                const auto& heap = heapEntry.second;
                if (!heap.gpuStart.ptr || !heap.increment || gpu < heap.gpuStart.ptr)
                  continue;
                const uint64_t delta = gpu - heap.gpuStart.ptr;
                const uint64_t index = delta / heap.increment;
                if (delta % heap.increment || index >= heap.count)
                  continue;
                const uint64_t cpu = heap.cpuStart.ptr + index * heap.increment;
                const auto descriptor = g_descriptors.find(cpu);
                if (descriptor != g_descriptors.end() && descriptor->second.resource) {
                  const auto metadata = findResourceMetadata(descriptor->second.resource);
                  if (metadata)
                    logSrvBufferData(gpu, *metadata);
                }
                break;
              }
            }
          }
        }
        log("D3D12 native frontend draw metadata: frame=%llu vsHash=0x%016llx psHash=0x%016llx inputs=%u rt0=%u dsv=%u table0=0x%llx table1=0x%llx table2=0x%llx table3=0x%llx",
          static_cast<unsigned long long>(g_frame.frame),
          pipeline == g_pipelines.end() ? 0ull : static_cast<unsigned long long>(pipeline->second.vertexShaderHash),
          pipeline == g_pipelines.end() ? 0ull : static_cast<unsigned long long>(pipeline->second.pixelShaderHash),
          pipeline == g_pipelines.end() ? 0u : pipeline->second.inputElementCount,
          pipeline == g_pipelines.end() ? 0u : static_cast<unsigned>(pipeline->second.renderTargetFormat),
          pipeline == g_pipelines.end() ? 0u : static_cast<unsigned>(pipeline->second.depthStencilFormat),
          graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[0].ptr) : 0ull,
          graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[1].ptr) : 0ull,
          graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[2].ptr) : 0ull,
          graphicsDescriptorTables ? static_cast<unsigned long long>(graphicsDescriptorTables[3].ptr) : 0ull);
        log("D3D12 native frontend draw resources: table0=%p/%u cbv0=0x%llx/%u table1=%p/%u cbv1=0x%llx/%u table2=%p/%u cbv2=0x%llx/%u table3=%p/%u cbv3=0x%llx/%u",
          reinterpret_cast<void*>(resolvedResources[0]), resolvedFormats[0],
          static_cast<unsigned long long>(resolvedBufferLocations[0]), resolvedBufferSizes[0],
          reinterpret_cast<void*>(resolvedResources[1]), resolvedFormats[1],
          static_cast<unsigned long long>(resolvedBufferLocations[1]), resolvedBufferSizes[1],
          reinterpret_cast<void*>(resolvedResources[2]), resolvedFormats[2],
          static_cast<unsigned long long>(resolvedBufferLocations[2]), resolvedBufferSizes[2],
          reinterpret_cast<void*>(resolvedResources[3]), resolvedFormats[3],
          static_cast<unsigned long long>(resolvedBufferLocations[3]), resolvedBufferSizes[3]);

        if (firstBinding) {
          std::call_once(g_bindingFileOnce, openBindingFile);
          if (g_bindingFile) {
            const auto pipeline = g_pipelines.find(pipelineState);
            const uint64_t vsHash = pipeline == g_pipelines.end() ? 0ull : pipeline->second.vertexShaderHash;
            const uint64_t psHash = pipeline == g_pipelines.end() ? 0ull : pipeline->second.pixelShaderHash;
            g_bindingFile << g_frame.frame << ','
              << reinterpret_cast<uint64_t>(pipelineState) << ','
              << vsHash << ',' << psHash << ','
              << static_cast<unsigned>(topology) << ',' << indexCount << ',' << instanceCount << ','
              << vertexBuffer.BufferLocation << ',' << vertexBuffer.SizeInBytes << ',' << vertexBuffer.StrideInBytes << ','
              << indexBuffer.BufferLocation << ',' << indexBuffer.SizeInBytes << ',' << static_cast<unsigned>(indexBuffer.Format);
            for (size_t table = 0; table < 4; ++table) {
              const uint64_t gpu = graphicsDescriptorTables ? graphicsDescriptorTables[table].ptr : 0;
              const ResourceMetadata* metadata = resolvedMetadata[table];
              g_bindingFile << ',' << gpu
                << ',' << resolvedResources[table]
                << ',' << resolvedFormats[table]
                << ',' << (metadata ? metadata->description.Width : 0ull)
                << ',' << (metadata ? metadata->description.Height : 0u);
            }
            g_bindingFile << '\n';
            g_bindingFile.flush();
          }
        }
      }
    }
    D3D12NativePipelineInfo nativePipelineInfo;
    const D3D12NativePipelineInfo* nativePipelineInfoPtr = nullptr;
    bool captureNativeBinding = false;
    {
      std::lock_guard lock(g_frameMutex);
      const auto pipeline = g_pipelines.find(pipelineState);
      if (pipeline != g_pipelines.end()) {
        nativePipelineInfo = pipeline->second.nativeInfo;
        nativePipelineInfoPtr = &nativePipelineInfo;
      }
      nativePipelineInfo.drawId = count;
      nativePipelineInfo.pipelineState = reinterpret_cast<uint64_t>(pipelineState);
      nativePipelineInfo.vertexBufferLocation = vertexBuffer.BufferLocation;
      nativePipelineInfo.vertexBufferSize = vertexBuffer.SizeInBytes;
      nativePipelineInfo.vertexBufferStride = vertexBuffer.StrideInBytes;
      nativePipelineInfo.indexBufferLocation = indexBuffer.BufferLocation;
      nativePipelineInfo.indexBufferSize = indexBuffer.SizeInBytes;
      nativePipelineInfo.indexBufferFormat = indexBuffer.Format;
      nativePipelineInfo.indexCount = indexCount;
      nativePipelineInfo.vertexCount = vertexCount;
      nativePipelineInfo.implicitIndices = (indexCount == 0 && vertexCount >= 3);
      nativePipelineInfo.instanceCount = instanceCount;
      nativePipelineInfo.startIndexLocation = startIndexLocation;
      nativePipelineInfo.baseVertexLocation = baseVertexLocation;
      nativePipelineInfo.startInstanceLocation = startInstanceLocation;
      nativePipelineInfo.objectTransformCbv = objectTransformCbv;
      if (graphicsDescriptorTables) {
        for (size_t i = 0; i < nativePipelineInfo.descriptorTables.size(); ++i)
          nativePipelineInfo.descriptorTables[i] = graphicsDescriptorTables[i].ptr;
        const uint64_t bindingHash = drawBindingHash(
          pipelineState, vertexBuffer, indexBuffer, graphicsDescriptorTables);
        if (kEnableNativeBindingSnapshots
         && g_loggedNativeBindings.size() < kMaxNativeBindingSnapshots
         && g_loggedNativeBindings.emplace(bindingHash).second) {
          captureNativeBinding = true;
          const uint64_t bindingStartTicks = perfTicks();
          collectBoundDescriptors(graphicsDescriptorTables, nativePipelineInfo.boundDescriptors);
          g_perfBindingTicks.fetch_add(
            perfTicks() - bindingStartTicks, std::memory_order_relaxed);
        }
      }
      nativePipelineInfoPtr = &nativePipelineInfo;
    }
    if (graphicsRootSrvs) {
      static std::atomic<uint32_t> rootSrvDiag { 0 };
      if (rootSrvDiag.fetch_add(1, std::memory_order_relaxed) < 2) {
        log("D3D12 root srv diag: (0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx)",
          static_cast<unsigned long long>(graphicsRootSrvs[0]),
          static_cast<unsigned long long>(graphicsRootSrvs[1]),
          static_cast<unsigned long long>(graphicsRootSrvs[2]),
          static_cast<unsigned long long>(graphicsRootSrvs[3]),
          static_cast<unsigned long long>(graphicsRootSrvs[4]),
          static_cast<unsigned long long>(graphicsRootSrvs[5]),
          static_cast<unsigned long long>(graphicsRootSrvs[6]),
          static_cast<unsigned long long>(graphicsRootSrvs[7]));
      }
    }
    bool objectTransformFromSrv = false;
    // The per-frame camera lives at offset 0 of the draw buffer, which can be
    // bound either as a root SRV or through a descriptor table. Forward it
    // before the per-object resolution so Remix has a camera even for draws
    // without a readable object transform.
    {
      std::array<float, 16> cameraMatrix {};
      bool haveCamera = false;
      if (graphicsRootSrvs) {
        for (uint32_t slot = 0; slot < 16 && !haveCamera; ++slot) {
          const D3D12_GPU_VIRTUAL_ADDRESS address = graphicsRootSrvs[slot];
          if (!address)
            continue;
          std::array<float, 16> candidate {};
          if (!readMatrixFromShaderBuffer(address, 0x100000, 0, candidate))
            continue;
          if (!isCameraViewProjection(candidate))
            continue;
          cameraMatrix = candidate;
          haveCamera = true;
        }
      }
      if (!haveCamera && graphicsDescriptorTables
       && readCameraFromDescriptorTables(graphicsDescriptorTables, cameraMatrix))
        haveCamera = true;
      if (haveCamera)
        submitD3D12Camera(cameraMatrix.data(), nullptr);
    }
    // Probe only the model vertex shader's draws: the earlier diagnostic
    // sampled the first draws of a frame, which never bind the draw buffer.
    {
      const auto probePipeline = g_pipelines.find(pipelineState);
      if (probePipeline != g_pipelines.end()
       && probePipeline->second.vertexShaderHash == 0xb6a7d1d096ce8fe7ull)
        probeDrawBufferBindings(graphicsDescriptorTables, graphicsRootSrvs);
    }
    // The per-frame view matrix lives in the shared draw CBV at offset 0x100;
    // per-object matrices arrive through the same slot for object draws. The
    // first draw of a frame binds the frame-level CBV, so take that one and
    // combine it with the projection measured from the game.
    {
      static std::array<float, 16> frameView {};
      static bool hasFrameView = false;
      static uint64_t lastFrameStamp = ~0ull;
      uint64_t frameStamp = 0;
      {
        std::lock_guard lock(g_frameMutex);
        frameStamp = g_presentCount.load(std::memory_order_relaxed);
      }

      if (frameStamp != lastFrameStamp) {
        lastFrameStamp = frameStamp;
        hasFrameView = false;
      }

      std::array<float, 16> candidateView {};
      if (!hasFrameView && objectTransformCbv
       && readObjectTransform(objectTransformCbv, candidateView)) {
        // A camera transform carries a world-space translation. Matrices that
        // stay at the origin come from the CPU-transformed PS2 path, which has
        // no camera on the GPU at all, so they are ignored.
        const float tx = candidateView[12];
        const float ty = candidateView[13];
        const float tz = candidateView[14];
        const float translationLength = std::sqrt(tx * tx + ty * ty + tz * tz);
        if (translationLength >= 1.0f) {
          frameView = candidateView;
          hasFrameView = true;

        // Row-vector projection measured from the game (1280x720).
        constexpr float p00 = 1.786676f;
        constexpr float p11 = 3.176313f;
        constexpr float p22 = 1.000183f;
        constexpr float pB = -11.5316f;
        std::array<float, 16> projection {
          p00, 0.0f, 0.0f, 0.0f,
          0.0f, -p11, 0.0f, 0.0f,
          0.0f, 0.0f, p22, 1.0f,
          0.0f, 0.0f, pB, 0.0f,
        };

        std::array<float, 16> viewProjection {};
        for (uint32_t row = 0; row < 4; ++row) {
          for (uint32_t column = 0; column < 4; ++column) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < 4; ++k)
              sum += frameView[row * 4 + k] * projection[k * 4 + column];
            viewProjection[row * 4 + column] = sum;
          }
        }

        log("D3D12 camera from frame CBV: t=(%.1f,%.1f,%.1f)",
          static_cast<double>(frameView[12]),
          static_cast<double>(frameView[13]),
          static_cast<double>(frameView[14]));
        // The frame constants do not form a view-projection, so nothing is submitted.
        }
      }
    }
    const uint64_t transformStartTicks = perfTicks();
    // Address of the draw buffer the object matrix came from, so the leading
    // matrix at +0x000 can be read from the very same binding.
    D3D12_GPU_VIRTUAL_ADDRESS drawBufferAddress = 0;
    UINT64 drawBufferWidth = 0;
    if (graphicsRootSrvs) {
      for (uint32_t slot = 0; slot < 16 && !nativePipelineInfo.hasObjectToWorld; ++slot) {
        const D3D12_GPU_VIRTUAL_ADDRESS address = graphicsRootSrvs[slot];
        if (!address)
          continue;
        std::array<float, 16> candidate {};
        if (!readMatrixFromShaderBuffer(address, 0x100000, 0x100, candidate))
          continue;
        nativePipelineInfo.objectToWorld = candidate;
        nativePipelineInfo.hasObjectToWorld = true;
        nativePipelineInfo.objectTransformSource = 1;
        objectTransformFromSrv = true;
        drawBufferAddress = address;
        drawBufferWidth = 0x100000;
        nativePipelineInfoPtr = &nativePipelineInfo;
      }
    }
    if (!nativePipelineInfo.hasObjectToWorld
     && graphicsDescriptorTables
     && readObjectTransformFromDescriptorTables(
          graphicsDescriptorTables, nativePipelineInfo.objectToWorld,
          &drawBufferAddress, &drawBufferWidth)) {
      nativePipelineInfo.hasObjectToWorld = true;
      objectTransformFromSrv = true;
      nativePipelineInfo.objectTransformSource = 1;
      nativePipelineInfoPtr = &nativePipelineInfo;
    } else if (objectTransformCbv
     && readObjectTransform(objectTransformCbv, nativePipelineInfo.objectToWorld)) {
      nativePipelineInfo.hasObjectToWorld = true;
      nativePipelineInfo.objectTransformSource = 2;
      drawBufferAddress = objectTransformCbv;
      drawBufferWidth = 0x100000;
      if (isIdentityObjectTransform(nativePipelineInfo.objectToWorld)) {
        // Terrain and some world batches use a different CBV matrix slot.
        // Probe only when the established 0x100 slot is identity, and only
        // accept a validated non-identity affine matrix.
        constexpr uint32_t kAlternateOffsets[] = {
          0x000, 0x040, 0x080, 0x0c0, 0x140, 0x180, 0x1c0,
          0x200, 0x240, 0x280, 0x2c0, 0x300, 0x340, 0x380, 0x3c0,
        };
        std::array<float, 16> alternate {};
        for (const uint32_t offset : kAlternateOffsets) {
          if (offset == 0x100 || !readObjectTransform(
              objectTransformCbv, alternate, offset)
           || isIdentityObjectTransform(alternate))
            continue;
          nativePipelineInfo.objectToWorld = alternate;
          nativePipelineInfo.objectTransformSource = 4;
          break;
        }
      }
      // Some pipelines bind their per-draw constants in roots 3..5 instead of
      // root 2, so those slots are probed before giving up on the transform.
      if (!nativePipelineInfo.hasObjectToWorld
       || isIdentityObjectTransform(nativePipelineInfo.objectToWorld)) {
        for (uint32_t slot = 3; slot <= 5; ++slot) {
          const auto address = currentGraphicsRootCbv(slot);
          if (!address)
            continue;
          std::array<float, 16> candidate {};
          bool found = false;
          for (uint32_t offset = 0; offset <= 0x3c0; offset += 0x40) {
            if (!readObjectTransform(address, candidate, offset)
             || isIdentityObjectTransform(candidate))
              continue;
            found = true;
            break;
          }
          if (!found)
            continue;
          nativePipelineInfo.objectToWorld = candidate;
          nativePipelineInfo.hasObjectToWorld = true;
          nativePipelineInfo.objectTransformSource = static_cast<uint8_t>(7 + slot);
          nativePipelineInfoPtr = &nativePipelineInfo;
          break;
        }
      }
      nativePipelineInfoPtr = &nativePipelineInfo;
    }
    // An identity object matrix means the vertices are still in model space and
    // the placement lives in the leading draw buffer matrix instead. Draws whose
    // object matrix could not be resolved at all are tried as well, because a
    // model draw reads every transform from that buffer.
    if (!nativePipelineInfo.hasObjectToWorld
     || isIdentityObjectTransform(nativePipelineInfo.objectToWorld)) {
      std::array<float, 16> basePlacement {};
      if (nativePipelineInfo.hasObjectToWorld) {
        basePlacement = nativePipelineInfo.objectToWorld;
      } else {
        basePlacement[0] = 1.0f;
        basePlacement[5] = 1.0f;
        basePlacement[10] = 1.0f;
        basePlacement[15] = 1.0f;
      }

      // The buffer the object matrix came from is tried first. The root SRVs and
      // the root CBV follow for draws whose matrix could not be resolved, since
      // those bind the draw buffer directly.
      D3D12_GPU_VIRTUAL_ADDRESS candidates[18] = {};
      UINT64 widths[18] = {};
      uint32_t candidateCount = 0;
      const auto addCandidate = [&](D3D12_GPU_VIRTUAL_ADDRESS address, UINT64 width) {
        if (!address || candidateCount >= 18)
          return;
        for (uint32_t i = 0; i < candidateCount; ++i) {
          if (candidates[i] == address)
            return;
        }
        candidates[candidateCount] = address;
        widths[candidateCount] = width;
        ++candidateCount;
      };
      addCandidate(drawBufferAddress, drawBufferWidth);
      if (graphicsRootSrvs) {
        for (uint32_t slot = 0; slot < 16; ++slot)
          addCandidate(graphicsRootSrvs[slot], 0x100000);
      }
      addCandidate(objectTransformCbv, 0x100000);

      // The bound tables are scanned once per frame, because the views they hold
      // outlive the individual draw while their base pointers do not.
      {
        static std::vector<D3D12_GPU_VIRTUAL_ADDRESS> cachedAddresses;
        static std::vector<UINT64> cachedWidths;
        static uint64_t cachedFrame = ~0ull;
        if (drawFrame != cachedFrame) {
          cachedFrame = drawFrame;
          cachedAddresses.clear();
          cachedWidths.clear();
          collectDrawBufferCandidates(graphicsDescriptorTables, count,
            cachedAddresses, cachedWidths);
        }
        for (size_t i = 0; i < cachedAddresses.size(); ++i)
          addCandidate(cachedAddresses[i], cachedWidths[i]);
      }

      reportTrackedBufferViews(count, drawFrame, (drawFrame % 600) == 0);

      // Raw dump of the buffer the object matrix came from, so the layout around
      // the leading matrix can be read off directly when decoding fails. One
      // sample per frame spreads the dumps over the whole session instead of
      // spending them on the first frame.
      {
        static std::atomic<uint32_t> blockDumps { 0 };
        static uint64_t lastDumpFrame = ~0ull;
        if (drawBufferAddress && drawFrame != lastDumpFrame
         && blockDumps.fetch_add(1, std::memory_order_relaxed) < 10) {
          lastDumpFrame = drawFrame;
          log("ACZ identity draw: draw=%llu frame=%llu vs=0x%016llx ps=0x%016llx source=%u buffer=0x%llx width=%llu stride=%u indices=%u",
            static_cast<unsigned long long>(count),
            static_cast<unsigned long long>(drawFrame),
            static_cast<unsigned long long>(nativePipelineInfo.vertexShaderHash),
            static_cast<unsigned long long>(nativePipelineInfo.pixelShaderHash),
            static_cast<unsigned>(nativePipelineInfo.objectTransformSource),
            static_cast<unsigned long long>(drawBufferAddress),
            static_cast<unsigned long long>(drawBufferWidth),
            vertexBuffer.StrideInBytes, indexCount);
          for (uint32_t row = 0; row < 24; ++row) {
            std::vector<uint8_t> bytes;
            if (!readGpuBuffer(drawBufferAddress + row * 16, 16, bytes)
             || bytes.size() < 16) {
              log("ACZ identity block +0x%03X: unreadable", row * 16);
              continue;
            }
            const auto* words = reinterpret_cast<const uint32_t*>(bytes.data());
            const auto* values = reinterpret_cast<const float*>(bytes.data());
            log("ACZ identity block +0x%03X: %08x %08x %08x %08x | %12.5g %12.5g %12.5g %12.5g",
              row * 16, words[0], words[1], words[2], words[3],
              static_cast<double>(values[0]), static_cast<double>(values[1]),
              static_cast<double>(values[2]), static_cast<double>(values[3]));
          }
        }
      }

      // Model space vertices of this draw. A candidate matrix is only accepted
      // when it also puts them in front of the camera and near the screen, which
      // is what separates this draw's matrix from a neighbour's.
      // Model space vertices of this draw, read through the pipeline's own
      // POSITION element because the attribute is not always first. A candidate
      // matrix is only accepted when it puts them in front of the camera and near
      // the screen, which separates this draw's matrix from a neighbour's.
      float modelVertices[6] = {};
      uint32_t modelVertexCount = 0;
      uint32_t positionOffset = 0;
      bool hasPosition = false;
      for (const auto& element : nativePipelineInfo.inputElements) {
        if (element.slot != 0)
          continue;
        std::string semantic = element.semantic;
        std::transform(semantic.begin(), semantic.end(), semantic.begin(),
          [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (semantic == "POSITION") {
          positionOffset = element.offset;
          hasPosition = true;
          break;
        }
        // Model draws name their attributes ATTRIBUTE0..3 and keep the position in
        // the first one.
        if (!hasPosition || element.offset < positionOffset) {
          positionOffset = element.offset;
          hasPosition = true;
        }
      }
      if (hasPosition && vertexBuffer.StrideInBytes > positionOffset + sizeof(float) * 3) {
        std::vector<uint8_t> vertexBytes;
        if (readGpuBuffer(vertexBuffer.BufferLocation, vertexBuffer.StrideInBytes * 2, vertexBytes)) {
          const uint32_t available = static_cast<uint32_t>(
            vertexBytes.size() / vertexBuffer.StrideInBytes);
          modelVertexCount = std::min<uint32_t>(available, 2u);
          for (uint32_t i = 0; i < modelVertexCount; ++i) {
            const auto* position = reinterpret_cast<const float*>(
              vertexBytes.data() + static_cast<size_t>(i) * vertexBuffer.StrideInBytes
              + positionOffset);
            modelVertices[i * 3 + 0] = position[0];
            modelVertices[i * 3 + 1] = position[1];
            modelVertices[i * 3 + 2] = position[2];
          }
        }
      }

      const auto verticesInFront = [&](const std::array<float, 16>& matrix) {
        if (modelVertexCount == 0)
          return true;
        float centroidX = 0.0f;
        float centroidY = 0.0f;
        for (uint32_t i = 0; i < modelVertexCount; ++i) {
          const float* v = &modelVertices[i * 3];
          float clip[4] = {};
          for (uint32_t column = 0; column < 4; ++column) {
            clip[column] = v[0] * matrix[0 * 4 + column]
              + v[1] * matrix[1 * 4 + column]
              + v[2] * matrix[2 * 4 + column]
              + matrix[3 * 4 + column];
          }
          if (clip[3] <= 0.0f)
            return false;
          centroidX += clip[0] / clip[3];
          centroidY += clip[1] / clip[3];
        }
        centroidX /= static_cast<float>(modelVertexCount);
        centroidY /= static_cast<float>(modelVertexCount);
        return std::abs(centroidX) <= 2.0f && std::abs(centroidY) <= 2.0f;
      };

      bool fusedDecoded = false;
      for (uint32_t i = 0; i < candidateCount; ++i) {
        std::array<float, 16> decoded {};
        if (!decodeFusedModelTransform(candidates[i], widths[i], basePlacement,
              vertexBuffer.BufferLocation, vertexBuffer.StrideInBytes, decoded))
          continue;
        if (!verticesInFront(decoded))
          continue;
        nativePipelineInfo.objectToWorld = decoded;
        nativePipelineInfo.hasObjectToWorld = true;
        nativePipelineInfo.objectTransformSource = 5;
        nativePipelineInfoPtr = &nativePipelineInfo;
        fusedDecoded = true;
        break;
      }

      // Nothing in the known bindings carried the matrix. The per-draw constants
      // block keeps a candidate at +0x40, so that offset is read directly instead
      // of scanning: the scan of the surrounding memory cost more per frame than
      // the whole captured scene.
      if (!fusedDecoded && indexCount >= 20) {
        D3D12_GPU_VIRTUAL_ADDRESS arenas[4] = {};
        uint32_t arenaCount = 0;
        const auto addArena = [&](D3D12_GPU_VIRTUAL_ADDRESS address) {
          if (!address || arenaCount >= 4)
            return;
          for (uint32_t i = 0; i < arenaCount; ++i) {
            if (arenas[i] == address)
              return;
          }
          arenas[arenaCount++] = address;
        };
        addArena(objectTransformCbv);
        for (uint32_t slot = 0; slot <= 1; ++slot)
          addArena(currentGraphicsRootCbv(slot));

        static std::atomic<uint32_t> drawLogs { 0 };
        if (drawLogs.fetch_add(1, std::memory_order_relaxed) < 16) {
          log("ACZ fused draw: draw=%llu vs=0x%016llx indices=%u stride=%u vb=0x%llx/%u position=%d verts=%u first=(%.4g,%.4g,%.4g) second=(%.4g,%.4g,%.4g) arenas=%u",
            static_cast<unsigned long long>(count),
            static_cast<unsigned long long>(nativePipelineInfo.vertexShaderHash),
            indexCount, vertexBuffer.StrideInBytes,
            static_cast<unsigned long long>(vertexBuffer.BufferLocation),
            vertexBuffer.SizeInBytes,
            hasPosition ? static_cast<int32_t>(positionOffset) : -1,
            modelVertexCount,
            static_cast<double>(modelVertices[0]), static_cast<double>(modelVertices[1]),
            static_cast<double>(modelVertices[2]),
            static_cast<double>(modelVertices[3]), static_cast<double>(modelVertices[4]),
            static_cast<double>(modelVertices[5]), arenaCount);
        }

        // The per-draw block keeps several matrices side by side, so both the
        // placement and the fused matrix are read from it as a pair: taking the
        // fused matrix alone assumes an identity placement, which scales the
        // result away from the camera and multiplies any camera motion with it.
        constexpr uint32_t kBlockOffsets[] = { 0x000, 0x040, 0x080, 0x0C0, 0x100 };
        constexpr uint32_t kBlockBytes = 0x140;

        // A placement is usable when its rotation stays uniform and affine; the
        // eye distance then picks among the usable pairs.
        const auto scorePlacement = [&](const std::array<float, 16>& matrix) {
          const float r0 = std::sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2]);
          const float r1 = std::sqrt(matrix[4] * matrix[4] + matrix[5] * matrix[5] + matrix[6] * matrix[6]);
          const float r2 = std::sqrt(matrix[8] * matrix[8] + matrix[9] * matrix[9] + matrix[10] * matrix[10]);
          if (r0 < 1.0e-4f || r1 < 1.0e-4f || r2 < 1.0e-4f)
            return -1.0f;
          const float largest = std::max(r0, std::max(r1, r2));
          const float smallest = std::min(r0, std::min(r1, r2));
          if (largest > smallest * 1.02f)
            return -1.0f;
          const float dot01 = (matrix[0] * matrix[4] + matrix[1] * matrix[5] + matrix[2] * matrix[6])
            / (r0 * r1);
          const float dot02 = (matrix[0] * matrix[8] + matrix[1] * matrix[9] + matrix[2] * matrix[10])
            / (r0 * r2);
          const float dot12 = (matrix[4] * matrix[8] + matrix[5] * matrix[9] + matrix[6] * matrix[10])
            / (r1 * r2);
          if (std::abs(dot01) > 0.02f || std::abs(dot02) > 0.02f || std::abs(dot12) > 0.02f)
            return -1.0f;
          AczCameraMatrices camera {};
          if (!buildAczCamera(camera))
            return -1.0f;
          const float dx = matrix[12] - camera.eye[0];
          const float dy = matrix[13] - camera.eye[1];
          const float dz = matrix[14] - camera.eye[2];
          return std::sqrt(dx * dx + dy * dy + dz * dz);
        };

        for (uint32_t i = 0; i < arenaCount && !fusedDecoded; ++i) {
          std::vector<uint8_t> block;
          if (!readGpuBuffer(arenas[i], kBlockBytes, block) || block.size() < kBlockBytes)
            continue;

          std::array<std::array<float, 16>, 5> matrices {};
          for (uint32_t k = 0; k < 5; ++k)
            std::memcpy(matrices[k].data(), block.data() + kBlockOffsets[k], sizeof(float) * 16);

          // Raw dump of the per-draw block for the first few draws. The matrices
          // are identified by their content, which is faster than guessing which
          // offset holds the placement.
          {
            static std::atomic<uint32_t> dumps { 0 };
            if (dumps.fetch_add(1, std::memory_order_relaxed) < 2) {
              log("ACZ fused block: draw=%llu arena=0x%llx vs=0x%016llx indices=%u stride=%u",
                static_cast<unsigned long long>(count),
                static_cast<unsigned long long>(arenas[i]),
                static_cast<unsigned long long>(nativePipelineInfo.vertexShaderHash),
                indexCount, vertexBuffer.StrideInBytes);
              for (uint32_t row = 0; row < kBlockBytes / 16; ++row) {
                const auto* values = reinterpret_cast<const float*>(block.data() + row * 16);
                log("ACZ fused block +0x%03X: %12.5g %12.5g %12.5g %12.5g", row * 16,
                  static_cast<double>(values[0]), static_cast<double>(values[1]),
                  static_cast<double>(values[2]), static_cast<double>(values[3]));
              }
            }
          }

          float bestScore = -1.0f;
          uint32_t bestPlacement = 0;
          uint32_t bestFused = 0;
          std::array<float, 16> bestResult {};
          static std::atomic<uint32_t> pairLogs { 0 };
          const bool logPairs = pairLogs.fetch_add(1, std::memory_order_relaxed) < 2;
          for (uint32_t fused = 0; fused < 5; ++fused) {
            // Index 5 means an identity placement, which is what a draw whose
            // vertices are already in world space uses.
            for (uint32_t placement = 0; placement <= 5; ++placement) {
              if (placement == fused)
                continue;
              std::array<float, 16> base {};
              if (placement == 5) {
                base[0] = 1.0f;
                base[5] = 1.0f;
                base[10] = 1.0f;
                base[15] = 1.0f;
              } else {
                base = matrices[placement];
              }
              std::array<float, 16> decoded {};
              const bool composed = composeFusedMatrix(matrices[fused], base, decoded);
              const float score = composed ? scorePlacement(decoded) : -1.0f;
              if (logPairs) {
                const float r0 = std::sqrt(decoded[0] * decoded[0] + decoded[1] * decoded[1]
                  + decoded[2] * decoded[2]);
                const float r1 = std::sqrt(decoded[4] * decoded[4] + decoded[5] * decoded[5]
                  + decoded[6] * decoded[6]);
                const float r2 = std::sqrt(decoded[8] * decoded[8] + decoded[9] * decoded[9]
                  + decoded[10] * decoded[10]);
                log("ACZ fused pair: fused=0x%X placement=0x%X composed=%u rows=(%.4f,%.4f,%.4f) t=(%.2f,%.2f,%.2f) score=%.2f",
                  kBlockOffsets[fused],
                  placement == 5 ? 0xFFFFu : kBlockOffsets[placement],
                  composed ? 1u : 0u,
                  static_cast<double>(r0), static_cast<double>(r1), static_cast<double>(r2),
                  static_cast<double>(decoded[12]), static_cast<double>(decoded[13]),
                  static_cast<double>(decoded[14]), static_cast<double>(score));
              }
              if (score < 0.0f)
                continue;
              if (bestScore >= 0.0f && score >= bestScore)
                continue;
              bestScore = score;
              bestPlacement = placement;
              bestFused = fused;
              bestResult = decoded;
            }
          }

          if (bestScore < 0.0f)
            continue;

          static std::atomic<uint32_t> hitLogs { 0 };
          if (hitLogs.fetch_add(1, std::memory_order_relaxed) < 20) {
            log("ACZ fused matrix found: draw=%llu arena=0x%llx fused=0x%X placement=0x%X t=(%.3f,%.3f,%.3f) eyeDistance=%.2f indices=%u",
              static_cast<unsigned long long>(count),
              static_cast<unsigned long long>(arenas[i]),
              kBlockOffsets[bestFused],
              bestPlacement == 5 ? 0xFFFFu : kBlockOffsets[bestPlacement],
              static_cast<double>(bestResult[12]), static_cast<double>(bestResult[13]),
              static_cast<double>(bestResult[14]),
              static_cast<double>(bestScore), indexCount);
          }
          nativePipelineInfo.objectToWorld = bestResult;
          nativePipelineInfo.hasObjectToWorld = true;
          nativePipelineInfo.objectTransformSource = 5;
          nativePipelineInfoPtr = &nativePipelineInfo;
          fusedDecoded = true;
        }
      }
    }

    g_perfTransformTicks.fetch_add(
      perfTicks() - transformStartTicks, std::memory_order_relaxed);
    if (captureNativeBinding) {
      const auto& m = nativePipelineInfo.objectToWorld;
      log("D3D12 native draw snapshot: draw=%llu frame=%llu pso=0x%llx vs=0x%016llx ps=0x%016llx vb=0x%llx/%u/%u ib=0x%llx/%u/%u args=(%u,%u,%u,%d,%u) transformSource=%u valid=%u t=(%.6g,%.6g,%.6g) descriptors=%zu",
        static_cast<unsigned long long>(count),
        static_cast<unsigned long long>(drawFrame),
        static_cast<unsigned long long>(nativePipelineInfo.pipelineState),
        static_cast<unsigned long long>(nativePipelineInfo.vertexShaderHash),
        static_cast<unsigned long long>(nativePipelineInfo.pixelShaderHash),
        static_cast<unsigned long long>(nativePipelineInfo.vertexBufferLocation),
        nativePipelineInfo.vertexBufferSize, nativePipelineInfo.vertexBufferStride,
        static_cast<unsigned long long>(nativePipelineInfo.indexBufferLocation),
        nativePipelineInfo.indexBufferSize,
        static_cast<unsigned>(nativePipelineInfo.indexBufferFormat),
        indexCount, instanceCount, startIndexLocation, baseVertexLocation,
        startInstanceLocation, nativePipelineInfo.objectTransformSource,
        nativePipelineInfo.hasObjectToWorld ? 1u : 0u,
        static_cast<double>(m[12]), static_cast<double>(m[13]), static_cast<double>(m[14]),
        nativePipelineInfo.boundDescriptors.size());
      size_t loggedDescriptors = 0;
      for (const auto& descriptor : nativePipelineInfo.boundDescriptors) {
        if (descriptor.kind != D3D12NativeDescriptorKind::Srv
         || descriptor.resourceDescription.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
         || loggedDescriptors >= 4)
          continue;
        log("D3D12 native draw SRV snapshot: draw=%llu root=%u offset=%u resource=0x%llx format=%u size=%llux%u mips=%u capturedSubresources=%u capturedBytes=%llu",
          static_cast<unsigned long long>(count),
          descriptor.rootParameterIndex, descriptor.tableOffset,
          static_cast<unsigned long long>(descriptor.resource),
          static_cast<unsigned>(descriptor.format),
          static_cast<unsigned long long>(descriptor.resourceDescription.Width),
          descriptor.resourceDescription.Height,
          descriptor.resourceDescription.MipLevels,
          descriptor.capturedSubresources,
          static_cast<unsigned long long>(descriptor.capturedBytes));
        ++loggedDescriptors;
      }
    }
    submitD3D12NativeGeometry(
      vertexBuffer, indexBuffer, topology, indexCount, instanceCount,
      startIndexLocation, baseVertexLocation, startInstanceLocation,
      nativePipelineInfoPtr);
    if (!shouldLog(count))
      return;
    log("D3D12 native frontend draw: list=%p pipeline=%p topology=%u indices=%u instances=%u startIndex=%u baseVertex=%d startInstance=%u vb=0x%llx/%u/%u ib=0x%llx/%u/%u count=%llu",
      commandList, pipelineState, static_cast<unsigned>(topology), indexCount, instanceCount,
      startIndexLocation, baseVertexLocation, startInstanceLocation,
      static_cast<unsigned long long>(vertexBuffer.BufferLocation), vertexBuffer.SizeInBytes,
      vertexBuffer.StrideInBytes, static_cast<unsigned long long>(indexBuffer.BufferLocation),
      indexBuffer.SizeInBytes, static_cast<unsigned>(indexBuffer.Format),
      static_cast<unsigned long long>(count));
    if (objectTransformFromSrv && shouldLog(count))
      log("D3D12 native frontend object transform source: descriptor-table t10 SRV");
    g_perfDrawTicks.fetch_add(
      perfTicks() - drawStartTicks, std::memory_order_relaxed);
    g_perfDrawCalls.fetch_add(1, std::memory_order_relaxed);
  }

  void recordD3D12Submit(ID3D12CommandQueue* commandQueue, UINT commandListCount, ID3D12CommandList* const* commandLists) {
    initializeD3D12RuntimeAdapter();
    const uint64_t count = ++g_submitCount;
    {
      std::lock_guard lock(g_frameMutex);
      ++g_frame.submits;
    }
    if (!shouldLog(count))
      return;
    log("D3D12 native frontend submit: queue=%p lists=%u first=%p count=%llu",
      commandQueue, commandListCount, commandListCount && commandLists ? commandLists[0] : nullptr,
      static_cast<unsigned long long>(count));
  }

  bool readBoundBufferFromDescriptorTable(
      uint64_t tableGpu,
      uint32_t descriptorIndex,
      size_t size,
      std::vector<uint8_t>& output,
      const char** kindOut,
      uint64_t* bufferGpuOut) {
    if (!tableGpu || !size)
      return false;

    D3D12_GPU_VIRTUAL_ADDRESS location = 0;
    {
      std::lock_guard lock(g_frameMutex);
      const DescriptorHeapMetadata* heap = nullptr;
      for (const auto& entry : g_descriptorHeaps) {
        const auto& metadata = entry.second;
        if (!metadata.gpuStart.ptr || !metadata.increment || tableGpu < metadata.gpuStart.ptr)
          continue;
        const uint64_t delta = tableGpu - metadata.gpuStart.ptr;
        if (delta % metadata.increment)
          continue;
        if (delta / metadata.increment >= metadata.count)
          continue;
        heap = &metadata;
        break;
      }
      if (!heap)
        return false;

      const uint64_t descriptorGpu = tableGpu
        + static_cast<uint64_t>(descriptorIndex) * heap->increment;
      const uint64_t delta = descriptorGpu - heap->gpuStart.ptr;
      if (delta % heap->increment)
        return false;
      const uint64_t index = delta / heap->increment;
      if (index >= heap->count)
        return false;

      const uint64_t cpu = heap->cpuStart.ptr + index * heap->increment;
      const auto descriptor = g_descriptors.find(cpu);
      if (descriptor == g_descriptors.end())
        return false;

      const DescriptorMetadata& metadata = descriptor->second;
      if (kindOut)
        *kindOut = metadata.viewType ? metadata.viewType : "unknown";
      if (metadata.viewType && std::strcmp(metadata.viewType, "CBV") == 0)
        location = metadata.bufferLocation;
      else
        location = metadata.resourceGpuAddress;
    }

    if (bufferGpuOut)
      *bufferGpuOut = location;
    if (!location)
      return false;
    return readGpuBuffer(location, size, output);
  }

  void recordD3D12FenceSignal(ID3D12CommandQueue* commandQueue, ID3D12Fence* fence, UINT64 value, HRESULT result) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    if (SUCCEEDED(result))
      ++g_frame.fenceSignals;
    if (g_frame.fenceSignals <= 3 || (g_frame.fenceSignals % 1024) == 0)
      log("D3D12 native frontend fence signal: frame=%llu queue=%p fence=%p value=%llu result=0x%08lx",
        static_cast<unsigned long long>(g_frame.frame), commandQueue, fence,
        static_cast<unsigned long long>(value), static_cast<unsigned long>(result));
  }

  void recordD3D12FenceWait(ID3D12CommandQueue* commandQueue, ID3D12Fence* fence, UINT64 value, HRESULT result) {
    initializeD3D12RuntimeAdapter();
    std::lock_guard lock(g_frameMutex);
    if (SUCCEEDED(result))
      ++g_frame.fenceWaits;
    if (g_frame.fenceWaits <= 3 || (g_frame.fenceWaits % 1024) == 0)
      log("D3D12 native frontend fence wait: frame=%llu queue=%p fence=%p value=%llu result=0x%08lx",
        static_cast<unsigned long long>(g_frame.frame), commandQueue, fence,
        static_cast<unsigned long long>(value), static_cast<unsigned long>(result));
  }

  void recordD3D12PresentBegin(
      void* swapChain,
      UINT backBufferIndex,
      ID3D12Resource* backBuffer,
      const D3D12_RESOURCE_DESC* backBufferDescription) {
    initializeD3D12RuntimeAdapter();
    const uint64_t frame = g_presentCount.load(std::memory_order_relaxed) + 1;
    {
      std::lock_guard lock(g_frameMutex);
      g_frame.frame = frame;
      g_frame.backBufferIndex = backBufferIndex;
      g_frame.presentBegin = true;
    }
    beginD3D12NativeFrame(swapChain, backBufferIndex, backBuffer, backBufferDescription);
    log("D3D12 native frontend frame begin: frame=%llu swapChain=%p backBufferIndex=%u resource=%p",
      static_cast<unsigned long long>(frame), swapChain, backBufferIndex, backBuffer);
  }

  // Content-based camera search. The draw buffer that carries the camera can be
  // bound through a root-signature range the command-list trace cannot map, so
  // instead of guessing bindings this walks the tracked buffer SRVs and accepts
  // the first one whose leading 64 bytes form a row-vector view-projection.
  void scanTrackedBuffersForCamera() {
    static std::atomic<uint32_t> scans { 0 };
    if (scans.fetch_add(1, std::memory_order_relaxed) > 8)
      return;

    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> candidates;
    std::vector<UINT64> widths;
    {
      std::lock_guard lock(g_frameMutex);
      for (const auto& entry : g_descriptors) {
        const DescriptorMetadata& metadata = entry.second;
        if (!metadata.viewType || std::strcmp(metadata.viewType, "SRV") != 0)
          continue;
        if (metadata.resourceDescription.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
          continue;
        if (!metadata.resourceGpuAddress)
          continue;

        UINT64 elementOffset = 0;
        if (metadata.viewDescriptionSize >= sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) {
          const auto* srv = reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(
            metadata.viewDescription.data());
          if (srv->ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
            elementOffset = static_cast<UINT64>(srv->Buffer.FirstElement)
              * srv->Buffer.StructureByteStride;
          }
        }
        const UINT64 width = metadata.resourceDescription.Width > elementOffset
          ? metadata.resourceDescription.Width - elementOffset : 0;
        candidates.push_back(metadata.resourceGpuAddress + elementOffset);
        widths.push_back(width);
        if (candidates.size() >= 2048)
          break;
      }
    }

    log("D3D12 camera scan: tracked buffers=%zu", candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      std::array<float, 16> matrix {};
      if (!readMatrixFromShaderBuffer(candidates[i], widths[i], 0, matrix))
        continue;
      if (!isCameraViewProjection(matrix))
        continue;
      log("D3D12 camera scan hit: gpu=0x%llx width=%llu",
        static_cast<unsigned long long>(candidates[i]),
        static_cast<unsigned long long>(widths[i]));
      submitD3D12Camera(matrix.data(), nullptr);
      return;
    }
  }

  void recordD3D12Present(void* swapChain, UINT syncInterval, UINT flags, HRESULT result) {
    pollCameraConventionKeys();
    initializeD3D12RuntimeAdapter();
    scanTrackedBuffersForCamera();

    // The camera is rebuilt from the game's own block every frame. Remix only
    // reports a camera as valid on the frame it was updated, so a frame without a
    // readable block keeps the last submitted matrices.
    {
      static uint32_t fitLogs = 0;
      float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f, radius = 0.0f;
      if (readD3D12InstanceBounds(centerX, centerY, centerZ, radius) && radius > 1.0f) {
        AczCameraMatrices camera {};
        if (buildAczCamera(camera)) {
          submitD3D12CameraMatrices(camera.view.data(), camera.projection.data());

          if (fitLogs < 4) {
            ++fitLogs;
            log("ACZ camera fit: center=(%.1f,%.1f,%.1f) radius=%.1f near=%.3f far=%.1f eye=(%.1f,%.1f,%.1f) third=(%.3f,%.3f,%.3f) depth=%s vertical=%s fovY=%.2f",
              static_cast<double>(centerX), static_cast<double>(centerY),
              static_cast<double>(centerZ), static_cast<double>(radius),
              static_cast<double>(camera.nearPlane), static_cast<double>(camera.farPlane),
              static_cast<double>(camera.eye[0]), static_cast<double>(camera.eye[1]),
              static_cast<double>(camera.eye[2]),
              static_cast<double>(camera.third[0]), static_cast<double>(camera.third[1]),
              static_cast<double>(camera.third[2]),
              camera.depthSign < 0.0f ? "backward" : "forward",
              camera.verticalSign > 0.0f ? "positive" : "negative",
              static_cast<double>(2.0 * std::atan(1.0 / camera.p11)
                * 180.0 / 3.14159265358979));

            // Reading the block's rows as right, up, third only holds when the
            // three vectors are orthonormal and r x u reproduces the third row.
            const float rx = camera.view[0];
            const float ry = camera.view[1];
            const float rz = camera.view[2];
            const float ux = camera.view[4];
            const float uy = camera.view[5];
            const float uz = camera.view[6];
            const float crossX = ry * uz - rz * uy;
            const float crossY = rz * ux - rx * uz;
            const float crossZ = rx * uy - ry * ux;
            log("ACZ camera basis: |r|=%.4f |u|=%.4f |f|=%.4f r.u=%.4f r.f=%.4f u.f=%.4f rxu.f=%.4f",
              static_cast<double>(std::sqrt(rx * rx + ry * ry + rz * rz)),
              static_cast<double>(std::sqrt(ux * ux + uy * uy + uz * uz)),
              static_cast<double>(std::sqrt(
                camera.third[0] * camera.third[0] + camera.third[1] * camera.third[1]
                + camera.third[2] * camera.third[2])),
              static_cast<double>(rx * ux + ry * uy + rz * uz),
              static_cast<double>(rx * camera.third[0] + ry * camera.third[1]
                + rz * camera.third[2]),
              static_cast<double>(ux * camera.third[0] + uy * camera.third[1]
                + uz * camera.third[2]),
              static_cast<double>(crossX * camera.third[0] + crossY * camera.third[1]
                + crossZ * camera.third[2]));
          }
        } else {
          static uint32_t missingBlockLogs = 0;
          if (missingBlockLogs < 2) {
            ++missingBlockLogs;
            log("ACZ camera: block unavailable, keeping the last submitted camera");
          }
        }
      }
    }
    // The scene is prepared against the camera that was last submitted, and Remix
    // only reports a camera as valid on the frame it was updated, so the camera
    // for this frame has to be in place before the scene is prepared.
    recordD3D12NativePresent();
    {
      static uint32_t dumps = 0;
      if (dumps < 4 && (g_presentCount % 120) == 0) {
        AczCameraProbe probe;
        if (readAczCameraProbe(probe)) {
          ++dumps;
          log("ACZ transform args: ctx=0x%llx out=0x%llx in=0x%llx count=%llu",
            static_cast<unsigned long long>(probe.transformArgs[0]),
            static_cast<unsigned long long>(probe.transformArgs[1]),
            static_cast<unsigned long long>(probe.transformArgs[2]),
            static_cast<unsigned long long>(probe.transformArgs[3]));
          log("ACZ camera probe: samples=%u object=0x%llx inner=0x%llx camera=0x%llx",
            probe.samples,
            static_cast<unsigned long long>(probe.object),
            static_cast<unsigned long long>(probe.inner),
            static_cast<unsigned long long>(probe.camera));
          for (uint32_t row = 0; row < 8; ++row) {
            log("ACZ camera block r%u: %12.3f %12.3f %12.3f %12.6f", row,
              static_cast<double>(probe.block[row * 4 + 0]),
              static_cast<double>(probe.block[row * 4 + 1]),
              static_cast<double>(probe.block[row * 4 + 2]),
              static_cast<double>(probe.block[row * 4 + 3]));
          }
        }
      }
    }
    const uint64_t count = ++g_presentCount;
    FrameLedger completed;
    {
      std::lock_guard lock(g_frameMutex);
      completed = g_frame;
      completed.frame = count;
      g_frame = {};
      g_frame.frame = count;
    }
    log("D3D12 native frontend frame summary: frame=%llu draws=%llu indexed=%llu nonIndexed=%llu submits=%llu pipelines=%llu resources=%llu fences(signal=%llu wait=%llu) backBufferIndex=%u begin=%u",
      static_cast<unsigned long long>(completed.frame),
      static_cast<unsigned long long>(completed.draws),
      static_cast<unsigned long long>(completed.indexedDraws),
      static_cast<unsigned long long>(completed.instancedDraws),
      static_cast<unsigned long long>(completed.submits),
      static_cast<unsigned long long>(completed.pipelines),
      static_cast<unsigned long long>(completed.resources),
      static_cast<unsigned long long>(completed.fenceSignals),
      static_cast<unsigned long long>(completed.fenceWaits),
      completed.backBufferIndex,
      completed.presentBegin ? 1u : 0u);
    if (count % 60 == 0 && g_perfFrequency.QuadPart) {
      const auto toMs = [](uint64_t ticks, LONGLONG frequency) {
        return static_cast<double>(ticks) * 1000.0
          / static_cast<double>(frequency);
      };
      log("D3D12 performance window: frames=60 drawCalls=%llu drawMs=%.3f transformMs=%.3f bindingMs=%.3f textureCalls=%llu textureMs=%.3f geometryCalls=%llu geometryMs=%.3f",
        static_cast<unsigned long long>(g_perfDrawCalls.exchange(0, std::memory_order_relaxed)),
        toMs(g_perfDrawTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart),
        toMs(g_perfTransformTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart),
        toMs(g_perfBindingTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart),
        static_cast<unsigned long long>(g_perfTextureCalls.exchange(0, std::memory_order_relaxed)),
        toMs(g_perfTextureTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart),
        static_cast<unsigned long long>(g_perfGeometryCalls.exchange(0, std::memory_order_relaxed)),
        toMs(g_perfGeometryTicks.exchange(0, std::memory_order_relaxed), g_perfFrequency.QuadPart));
    }
    if (!shouldLog(count))
      return;
    log("D3D12 native frontend frame boundary: swapChain=%p present=%llu sync=%u flags=0x%x result=0x%08lx",
      swapChain,
      static_cast<unsigned long long>(count),
      syncInterval,
      flags,
      static_cast<unsigned long>(result));
  }

  extern "C" void __cdecl remixD3D12RuntimePresent(
      void* swapChain, UINT syncInterval, UINT flags, HRESULT result) {
    recordD3D12Present(swapChain, syncInterval, flags, result);
  }

  extern "C" void __cdecl remixD3D12RuntimePresentBegin(
      void* swapChain,
      UINT backBufferIndex,
      ID3D12Resource* backBuffer,
      const D3D12_RESOURCE_DESC* backBufferDescription) {
    recordD3D12PresentBegin(swapChain, backBufferIndex, backBuffer, backBufferDescription);
  }

  // Control surface for a game-side adapter. A frontend that knows its engine's
  // camera and geometry conventions calls these each frame; the compatibility
  // layer itself carries no game-specific knowledge.
  extern "C" void __cdecl remixD3D12SetCamera(
      const float* viewProjection,
      const float* worldToView) {
    submitD3D12Camera(viewProjection, worldToView);
  }

  extern "C" void __cdecl remixD3D12SetOutputEnabled(UINT enabled) {
    setD3D12OutputEnabled(enabled != 0);
  }

  extern "C" void __cdecl remixD3D12SetWorldSpacePolicy(UINT mode, float minExtent) {
    setD3D12WorldSpacePolicy(mode, minExtent);
  }

}
