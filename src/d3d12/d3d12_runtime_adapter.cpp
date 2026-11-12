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

    // GPU address of the structured buffer the model vertex shaders read their
    // transform from, captured from the view that creates it. Zero until that
    // view exists.
    std::atomic<uint64_t> g_aczSrtBufferAddress { 0 };
    std::atomic<uint32_t> g_aczSrtBufferElements { 0 };

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

    // `maxMagnitude` guards against reading a non-matrix. A fused model-to-clip
    // matrix legitimately reaches 1e9 because it carries the projection scale,
    // so that caller passes a wider limit and relies on the affine test instead.
    bool readMatrixFromShaderBuffer(
        D3D12_GPU_VIRTUAL_ADDRESS gpuAddress,
        UINT64 resourceWidth,
        uint32_t offset,
        std::array<float, 16>& result,
        float maxMagnitude = 1.0e7f) {
      if (!gpuAddress || resourceWidth < offset + sizeof(float) * 16)
        return false;

      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(gpuAddress + offset, sizeof(float) * 16, bytes)
       || bytes.size() < sizeof(float) * 16)
        return false;

      const auto* matrix = reinterpret_cast<const float*>(bytes.data());
      for (unsigned i = 0; i < 16; ++i) {
        if (!std::isfinite(matrix[i]) || std::abs(matrix[i]) > maxMagnitude)
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
    // The heap ranges are what lets a draw's descriptor be read straight out of
    // host memory, so they are logged under a prefix that is never filtered.
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
        // Structured buffers wide enough to hold per-object data are the ones the
        // model vertex shaders read their transform from, so they are reported
        // under a tag that is never filtered out of the log.
        // A view may leave the stride at zero and take it from the resource, so
        // both have to be consulted before deciding this is a wide structured
        // buffer.
        // A view may leave the stride at zero for a structured buffer, in which
        // case the stride is derived from the resource width and element count.
        const UINT effectiveStride = srv->Buffer.StructureByteStride
          ? srv->Buffer.StructureByteStride
          : (srv->Buffer.NumElements
            ? static_cast<UINT>(resourceDesc.Width / srv->Buffer.NumElements)
            : 0u);
        {
          if (effectiveStride == 416 && metadata.resourceGpuAddress) {
            g_aczSrtBufferAddress.store(
              metadata.resourceGpuAddress
                + static_cast<uint64_t>(srv->Buffer.FirstElement) * effectiveStride,
              std::memory_order_relaxed);
            g_aczSrtBufferElements.store(srv->Buffer.NumElements, std::memory_order_relaxed);
          }
          static std::atomic<uint32_t> bufferSrvLogs { 0 };
          const uint32_t index = bufferSrvLogs.fetch_add(1, std::memory_order_relaxed);

        }
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
      if (barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
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

  struct AczCameraMatrices {    std::array<float, 16> view {};
    std::array<float, 16> projection {};
    // The projection the scene is rasterised with, which stores clip coordinates
    // offset by half the viewport. Transform records carry positions in this
    // space, so converting one back to world needs this matrix, not the clean one.
    std::array<float, 16> displayProjection {};
    bool hasDisplayProjection = false;
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

    // The aircraft's own world position lives in the scene node that shares this
    // object: the node sits at block + 0x7B8 and its position at node + 0x258,
    // which is block + 0xA10. Reported on change so it can be held against the
    // position this layer reconstructs from the record.
    {
      const auto* pPosition = reinterpret_cast<const float*>(
        reinterpret_cast<const uint8_t*>(pCamera) + 0xA10);
      MEMORY_BASIC_INFORMATION positionInfo {};
      if (VirtualQuery(pPosition, &positionInfo, sizeof(positionInfo))
       && positionInfo.State == MEM_COMMIT
       && reinterpret_cast<uintptr_t>(pPosition) + 4 * sizeof(float)
          <= reinterpret_cast<uintptr_t>(positionInfo.BaseAddress) + positionInfo.RegionSize) {
        static float lastPosition[4] = {};
        bool changed = false;
        for (uint32_t index = 0; index < 4; ++index)
          changed = changed || pPosition[index] != lastPosition[index];
        if (changed) {
          for (uint32_t index = 0; index < 4; ++index)
            lastPosition[index] = pPosition[index];
          log("ACZ node position: p=(%.3f,%.3f,%.3f) w=%.3f",
            static_cast<double>(pPosition[0]), static_cast<double>(pPosition[1]),
            static_cast<double>(pPosition[2]), static_cast<double>(pPosition[3]));
        }
      }
    }

    // One dump of the whole camera block per distinct block address, so a field
    // that already holds the aircraft position or a reference scale can be found
    // without guessing at offsets.
    {
      static uintptr_t dumpedCamera = 0;
      const auto* pFull = reinterpret_cast<const uint8_t*>(pCamera);
      MEMORY_BASIC_INFORMATION fullInfo {};
      if (reinterpret_cast<uintptr_t>(pFull) != dumpedCamera
       && VirtualQuery(pFull, &fullInfo, sizeof(fullInfo))
       && fullInfo.State == MEM_COMMIT
       && reinterpret_cast<uintptr_t>(pFull) + 0x120
          <= reinterpret_cast<uintptr_t>(fullInfo.BaseAddress) + fullInfo.RegionSize) {
        dumpedCamera = reinterpret_cast<uintptr_t>(pFull);
        float full[72] = {};
        std::memcpy(full, pFull, sizeof(full));
        log("ACZ camera block: ptr=0x%llx eye=(%.2f,%.2f,%.2f)",
          static_cast<unsigned long long>(dumpedCamera),
          static_cast<double>(eyeX), static_cast<double>(eyeY), static_cast<double>(eyeZ));
        for (uint32_t row = 0; row < 18; ++row) {
          log("ACZ camera block +0x%03X: %11.4f %11.4f %11.4f %11.4f", row * 16,
            static_cast<double>(full[row * 4 + 0]), static_cast<double>(full[row * 4 + 1]),
            static_cast<double>(full[row * 4 + 2]), static_cast<double>(full[row * 4 + 3]));
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
    float verticalSign =
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

    // The rasterising projection sits right before the clean one. Its two
    // diagonal terms and its third row carry the half viewport offsets that the
    // object records use, so it is what turns a record position into a world one.
    {
      const auto* pDisplayBytes = reinterpret_cast<const uint8_t*>(pCamera) + 0x90;
      MEMORY_BASIC_INFORMATION displayInfo {};
      if (VirtualQuery(pDisplayBytes, &displayInfo, sizeof(displayInfo))
       && displayInfo.State == MEM_COMMIT
       && reinterpret_cast<uintptr_t>(pDisplayBytes) + sizeof(out.displayProjection)
          <= reinterpret_cast<uintptr_t>(displayInfo.BaseAddress) + displayInfo.RegionSize) {
        std::memcpy(out.displayProjection.data(), pDisplayBytes, sizeof(out.displayProjection));
        bool displayFinite = true;
        for (float value : out.displayProjection)
          displayFinite = displayFinite && std::isfinite(value);
        // A projection has a scale on its first two rows and a negated depth term
        // in the third, which tells it apart from whatever else follows the block.
        out.hasDisplayProjection = displayFinite
          && std::abs(out.displayProjection[0]) > 0.1f
          && std::abs(out.displayProjection[5]) > 0.1f
          && std::abs(out.displayProjection[11] + 1.0f) < 0.1f;
      }
    }
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

      return false;
    };

    if (!buffer || bufferWidth < 0x40)
      return reject(kFusedNoBuffer);

    std::array<float, 16> modelToClip {};
    if (!readMatrixFromShaderBuffer(buffer, bufferWidth, 0x000, modelToClip, 1.0e12f))
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
            }
          }

          for (uint32_t row = 0; row < 4; ++row) {
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
          }
        }
      }
    }

    return true;
  }

  // Vertex shaders that place object space geometry with a per-draw matrix. Their
  // transforms come from the structured buffer described in acz-shaders.md, so
  // only these draws are worth probing for one.
  //
  // `DrawCPT_v` is deliberately absent: it declares no SV_Position output, passes
  // its ATTRIBUTE0 through to a generic POSITION and writes a clamped 0..1 value
  // into the fourth component, so it draws a screen space overlay rather than
  // object space geometry and has no placement to recover.
  bool isModelVertexShader(uint64_t hash) {
    switch (hash) {
    case 0xb6a7d1d096ce8fe7ull:  // DrawACM_v, aircraft
    case 0xeb39ece513987798ull:  // DrawModelGsh_v
    case 0x8a562777e73aa8d5ull:  // DrawModelIcon_v
    case 0x3c10411c7e513ba4ull:  // DrawSelMap_v
    case 0x9fcd88e1b090fcaaull:  // DrawModelShadow_v
      return true;
    default:
      return false;
    }
  }

  // Object geometry that reports its placement relative to the camera rather
  // than in world space. The terrain submits absolute world coordinates, so the
  // two conventions have to be told apart per shader.
  bool placesInCameraRelativeFrame(uint64_t hash) {
    switch (hash) {
    case 0xb6a7d1d096ce8fe7ull:  // DrawACM_v, aircraft
    case 0xeb39ece513987798ull:  // DrawModelGsh_v
      return true;
    default:
      return false;
    }
  }

  // How a draw's geometry has to be placed, decided by its vertex shader. Each
  // family in acz-shaders.md transforms differently, so one generic placement
  // cannot cover them all; the mode is looked up per shader instead and every
  // draw is routed through its own rule.
  enum class AczPlacementMode : uint8_t {
    kScreenSpace,      // Vertices are already in clip space; nothing to place.
    kWorldSpace,       // Vertices are already in world space; identity placement.
    kObjectMatrix,     // Model space vertices with one readable object matrix.
    kBlendedMatrices,  // Model space vertices blended from the SrtData buffer.
    kProbing,          // Not classified yet, keep validating candidates.
  };

  AczPlacementMode aczPlacementModeFor(uint64_t hash) {
    switch (hash) {
    case 0xdbac2cdada190473ull:  // DrawCPT_v, cockpit overlay
    case 0x110c4e2ff4ad2b54ull:  // ps2_v
    case 0x3f0ddb245dcbdd6full:  // ps2Line_v
    case 0x9f0c04b171687f1eull:  // ps2AtestCA_v
      return AczPlacementMode::kScreenSpace;
    case 0xb6a7d1d096ce8fe7ull:  // DrawACM_v, aircraft
    case 0xeb39ece513987798ull:  // DrawModelGsh_v
    case 0x8a562777e73aa8d5ull:  // DrawModelIcon_v
      return AczPlacementMode::kBlendedMatrices;
    case 0x3c10411c7e513ba4ull:  // DrawSelMap_v
    case 0x9fcd88e1b090fcaaull:  // DrawModelShadow_v
      return AczPlacementMode::kObjectMatrix;
    default:
      return AczPlacementMode::kProbing;
    }
  }

  // Reads the descriptors bound to a draw straight out of the descriptor heap and
  // collects every word that looks like a GPU address. The layer holds no views for
  // these descriptors and the model shaders read their matrices through them, so
  // the words are taken as they are and validated by content afterwards.
  void collectDescriptorAddresses(
      const D3D12_GPU_DESCRIPTOR_HANDLE* tables,
      D3D12_GPU_VIRTUAL_ADDRESS* out,
      uint32_t& count,
      uint32_t maxAddresses) {
    if (!tables)
      return;

    constexpr size_t kMaxTables = 8;
    constexpr uint32_t kDescriptorOffsets = 8;
    constexpr uint64_t kMinGpuAddress = 0x100000000ull;
    constexpr uint64_t kMaxGpuAddress = 0x7fffffffffffull;

    std::lock_guard lock(g_frameMutex);
    for (size_t table = 0; table < kMaxTables && count < maxAddresses; ++table) {
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

      const uint64_t baseIndex = (tableGpu - heap->gpuStart.ptr) / heap->increment;
      for (uint32_t offset = 0; offset < kDescriptorOffsets && count < maxAddresses; ++offset) {
        const uint64_t index = baseIndex + offset;
        if (index >= heap->count)
          break;
        const uint64_t cpu = heap->cpuStart.ptr + index * heap->increment;

        for (uint32_t word = 0; word < 4 && count < maxAddresses; ++word) {
          uint64_t value = 0;
          std::memcpy(&value, reinterpret_cast<const void*>(cpu + word * 8), sizeof(value));
          if (value < kMinGpuAddress || value > kMaxGpuAddress || (value & 0xf) != 0)
            continue;
          bool duplicate = false;
          for (uint32_t i = 0; i < count; ++i) {
            if (out[i] == value) {
              duplicate = true;
              break;
            }
          }
          if (duplicate)
            continue;
          out[count++] = static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(value);
        }
      }
    }
  }

  // Collects the buffer views bound to a draw. The model vertex shader reads its
  // transform data from a structured buffer that the draw trace cannot map onto a
  // root parameter, so the bound tables are scanned instead and the views are
  // validated by content rather than by binding.
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

  // Tracks which records the t10 probe has already reported. The game cycles
  // through a handful of records, so remembering only the previous address would
  // re-report every draw; this keeps one line per distinct record instead.
  struct RecordState {
    bool described = false;
    bool readable = false;
  };

  constexpr size_t kReportedRecordLimit = 8192;
  std::unordered_map<D3D12_GPU_VIRTUAL_ADDRESS, RecordState> g_recordStates;

  // Descriptor tables already scanned for the slot that carries the record, and
  // descriptor handles already reported. Both keep the probes to one line per
  // table rather than one line per draw.
  std::unordered_set<uint64_t> g_scannedTables;
  std::unordered_set<uint64_t> g_reportedDescriptors;

  // World position of the player aircraft rebuilt from its transform record.
  // While it is valid the placement below uses it and leaves the camera rig alone.
  std::atomic<bool> g_recordWorldValid { false };
  std::atomic<bool> g_recordWorldReported { false };
  float g_recordWorld[3] = { 0.0f, 0.0f, 0.0f };

  // The record is found by its own shape instead of by an address, because the
  // buffer view that names it cannot be resolved and the addresses read out of
  // the descriptor table belong to a texture. Two of its rows do not occur by
  // accident: the rotation basis ends in a zero translation, and the projection
  // constants are the half viewport of the rasterised target.
  constexpr size_t kRecordShapeSize = 0x1A0;

  // The viewport remap row is a constant of the render target, so testing it first
  // rejects almost every offset a full-buffer scan visits for the price of four
  // comparisons. Every test is written so that a NaN fails it: a tolerance
  // comparison against NaN is false, which would otherwise let a half-written
  // record through, and an explicit finiteness pass over all 104 floats costs far
  // too much to run at every offset the scan visits.
  bool recordShapeMatches(const float* f) {
    if (!(std::abs(f[0x190 / 4] - 512.0f) <= 0.01f))
      return false;
    if (!(std::abs(f[0x194 / 4] + 448.0f) <= 0.01f))
      return false;
    if (!(std::abs(f[0x198 / 4] - 2048.0f) <= 0.01f))
      return false;
    if (!(std::abs(f[0x19C / 4] - 2048.0f) <= 0.01f))
      return false;
    // The fourth row of the same matrix is the sub-mesh's offset inside the model, so
    // it is zero only for the parts that sit on the model origin. It is bounded rather
    // than required to be zero, which keeps the zero-filled and projection-like blocks
    // out while letting every other part of the model through: a capture of the takeoff
    // frame shows offsets of 11.96, 16.55 and 20.29 against a model roughly 20 units
    // long, and a NaN fails the comparison and is rejected with the rest.
    if (!(std::abs(f[0x13C / 4] - 1.0f) <= 1.0e-4f))
      return false;
    {
      const float offsetX = f[0x130 / 4];
      const float offsetY = f[0x134 / 4];
      const float offsetZ = f[0x138 / 4];
      const float offsetLength = std::sqrt(offsetX * offsetX + offsetY * offsetY
        + offsetZ * offsetZ);
      if (!(offsetLength <= 256.0f))
        return false;
    }
    // The basis above the translation must be orthonormal, which rejects the many
    // zero-filled and projection-like blocks that also end in (0,0,0,1).
    for (uint32_t row = 0; row < 3; ++row) {
      const float* r = f + (0x100 / 4) + row * 4;
      const float length = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      if (!(std::abs(length - 1.0f) <= 2.0e-3f) || !(std::abs(r[3]) <= 1.0e-4f))
        return false;
    }
    // The player aircraft is drawn centred, so the origin's clip x is exactly half
    // the viewport times the view depth. Other aircraft are off centre and miss
    // this, which is what makes the relation a reliable discriminator.
    const float depth = f[0x03C / 4];
    const float clipX = f[0x030 / 4];
    if (!(std::abs(depth) > 1.0f) || !(std::abs(depth) <= 1.0e5f))
      return false;
    if (!(std::abs(clipX - 2048.0f * depth) <= 0.02f * std::abs(clipX)))
      return false;
    return true;
  }

  // The arena holding the records is a ring buffer that is written while the frame
  // is recorded, so an address found in it once holds a later frame's contents
  // afterwards. Scanning it for a record therefore cannot be trusted: the frame a
  // candidate came from is unknown, and the camera it describes is that frame's.
  // Records are now read only from the vertex buffer the draw itself names.

  // A RenderDoc capture of this shader shows the instance block and the draw's
  // vertex stream adjacent in one buffer, with the block exactly 416 bytes before
  // the stream. Only the draws where that holds are reported, which keeps the line
  // to the aircraft itself instead of every object the shader covers.
  std::unordered_set<uint64_t> g_vertexBufferRecordLogged;

  void verifyRecordBeforeVertexBuffer(const D3D12_VERTEX_BUFFER_VIEW& vertexBuffer) {
    if (!vertexBuffer.BufferLocation || vertexBuffer.BufferLocation < kRecordShapeSize)
      return;
    const D3D12_GPU_VIRTUAL_ADDRESS address =
      vertexBuffer.BufferLocation - kRecordShapeSize;
    if (!g_vertexBufferRecordLogged.emplace(address).second)
      return;
    std::vector<uint8_t> bytes;
    if (!readGpuBuffer(address, kRecordShapeSize, bytes) || bytes.size() < kRecordShapeSize)
      return;
    const float* record = reinterpret_cast<const float*>(bytes.data());
    if (!recordShapeMatches(record))
      return;
    log("ACZ vb record: va=0x%llx placementT=(%.3f,%.3f,%.3f) w=%.4f depth=%.3f",
      static_cast<unsigned long long>(address),
      static_cast<double>(record[0x130 / 4]), static_cast<double>(record[0x134 / 4]),
      static_cast<double>(record[0x138 / 4]), static_cast<double>(record[0x13C / 4]),
      static_cast<double>(record[0x03C / 4]));
  }

  // Vertex buffer of the draw being recorded, which names its instance block.
  D3D12_GPU_VIRTUAL_ADDRESS g_currentVertexBuffer = 0;
  // Index count of the draw being placed, so the summary can tell two submissions of the
  // same mesh apart from two different meshes without another round trip.
  uint32_t g_currentIndexCount = 0;

  // Turns a draw's record into the world position the draw puts its object at. The
  // record's leading matrix maps the model origin to the clip position the game
  // rasterises the aircraft at, so pushing that position back through the camera
  // the game rendered with yields the world position.
  //
  // The record is read only from the vertex buffer the draw names, because a record
  // found anywhere else belongs to a different frame or a different pass and the
  // camera it describes is not the one this draw uses. A draw whose own record is
  // unreadable keeps the position of the last record that was readable rather than
  // taking one from elsewhere, which is what used to slide the model away from the
  // camera as the arena ring advanced.
  float g_resolvedAircraft[3] = { 0.0f, 0.0f, 0.0f };
  bool g_resolvedAircraftValid = false;
  bool g_resolvedAircraftReused = false;
  float g_resolvedAircraftLogged[31] = {};
  // The engine target seen on the previous draw. The recorded draw renders the position
  // the engine reported one frame earlier, so this is what the aircraft is placed at.
  float g_cachedTarget[3] = { 0.0f, 0.0f, 0.0f };
  bool g_cachedTargetValid = false;
  // The last target value read, kept apart from the cached one so a frame's first draw
  // moves the cache and every later draw of that same frame keeps using it.
  float g_targetSeen[3] = { 0.0f, 0.0f, 0.0f };
  bool g_targetSeenValid = false;
  float g_resolvedAircraftBasis[9] = {};
  // The record's attitude is advanced one frame ahead of the camera the frame is submitted
  // with, the same way its position is ahead of the engine target, so every sub-mesh is
  // held back by one frame. Each part keeps its own basis because the record's matrix also
  // carries that part's local rotation.
  // The record carries no stable per part identity: its offset rotates with the aircraft,
  // its vertex buffer address is recycled, and neither the index count nor the offset
  // length separates the parts that share a model origin. What is stable is the order the
  // sub-meshes are drawn in, so the attitude cache is keyed by that order instead.
  constexpr uint32_t kAczPartSlots = 256;
  struct AczPartSlot {
    float basis[9] = {};
    float prev[9] = {};
    float offset[3] = {};
    float prevOffset[3] = {};
    bool prevValid = false;
    bool basisValid = false;
  };
  AczPartSlot g_partSlots[kAczPartSlots] {};
  uint32_t g_partSlotIndex = 0;
  float g_frameMarker[9] = {};
  bool g_frameMarkerValid = false;
  // The offset the resolved record gave its sub-mesh, kept beside the basis so a draw
  // that has to reuse the last record reuses the matching offset as well.
  float g_resolvedAircraftOffset[3] = { 0.0f, 0.0f, 0.0f };
  // Meshes already placed in this frame. ACZ draws every part of the aircraft twice: the
  // base pass first, then an additive highlight pass that reads the same record and the
  // same vertices. The second submission is the same geometry with the same transform, so
  // it is dropped instead of putting two coincident copies of every part in the scene.
  std::vector<std::pair<D3D12_GPU_VIRTUAL_ADDRESS, uint32_t>> g_placedMeshes;

  bool placedMeshSubmitted(D3D12_GPU_VIRTUAL_ADDRESS vertexBuffer, uint32_t indexCount) {
    for (const auto& entry : g_placedMeshes) {
      if (entry.first == vertexBuffer && entry.second == indexCount)
        return true;
    }
    return false;
  }

  // Copies from the game image or from an object the game published, but only after
  // every page the range touches is proved committed and readable. The engine writes
  // these pointers during scene setup, so a render-thread read can otherwise land on
  // a page that is still reserved or guarded and fault the whole frame.
  bool copyReadableMemory(uintptr_t address, void* destination, size_t size) {
    if (address == 0 || destination == nullptr || size == 0)
      return false;
    uintptr_t cursor = address;
    const uintptr_t end = address + size;
    while (cursor < end) {
      MEMORY_BASIC_INFORMATION info {};
      if (VirtualQuery(reinterpret_cast<LPCVOID>(cursor), &info, sizeof(info)) == 0)
        return false;
      if (info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0)
        return false;
      const uintptr_t regionEnd = reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize;
      if (regionEnd <= cursor)
        return false;
      cursor = regionEnd;
    }
    std::memcpy(destination, reinterpret_cast<const void*>(address), size);
    return true;
  }

  // The engine keeps one view per player and that view names the object its camera is
  // framed around, whose position is a world position the chase camera is rebuilt from
  // every frame. Reading it therefore returns where the game itself believes the
  // aircraft is, with no reconstruction and no fit to a captured framing.
  //
  // The chain comes from the game's own code: the engine stores its own address in a
  // global while the scene is built, the per-player view array hangs off the engine at
  // +0x697B8, the view names its target at +0x110 and the target's world position is the
  // float4 at +0x258.
  constexpr uintptr_t kEngineGlobalOffset = 0x3409F8;
  constexpr uintptr_t kViewIndexOffset = 0x3409F2;
  constexpr uintptr_t kViewArrayOffset = 0x697B8;
  constexpr uintptr_t kViewTargetOffset = 0x110;
  constexpr uintptr_t kTargetWorldOffset = 0x258;

  bool readPlayerTargetWorld(float out[3]) {
    const HMODULE image = GetModuleHandleW(nullptr);
    if (!image)
      return false;
    const uintptr_t base = reinterpret_cast<uintptr_t>(image);
    uintptr_t engine = 0;
    if (!copyReadableMemory(base + kEngineGlobalOffset, &engine, sizeof(engine)) || !engine)
      return false;
    uintptr_t views = 0;
    if (!copyReadableMemory(engine + kViewArrayOffset, &views, sizeof(views)) || !views)
      return false;
    uint8_t viewIndex = 0;
    if (!copyReadableMemory(base + kViewIndexOffset, &viewIndex, sizeof(viewIndex)))
      return false;
    uintptr_t view = 0;
    if (!copyReadableMemory(views + static_cast<uintptr_t>(viewIndex) * sizeof(uintptr_t),
          &view, sizeof(view)) || !view)
      return false;
    uintptr_t target = 0;
    if (!copyReadableMemory(view + kViewTargetOffset, &target, sizeof(target)) || !target)
      return false;
    float position[4] = {};
    if (!copyReadableMemory(target + kTargetWorldOffset, position, sizeof(position)))
      return false;
    if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2]))
      return false;
    if (position[0] == 0.0f && position[1] == 0.0f && position[2] == 0.0f)
      return false;
    for (uint32_t index = 0; index < 3; ++index)
      out[index] = position[index];
    return true;
  }

  bool resolveRecordWorld(
      const AczCameraMatrices& camera,
      float out[3],
      float outBasis[9],
      float outOffset[3]) {
    std::vector<uint8_t> bytes;
    // The instance block sits exactly 416 bytes before the draw's vertex stream in
    // the same buffer, so the vertex buffer address names the record directly.
    const bool readable = g_currentVertexBuffer >= kRecordShapeSize
      && readGpuBuffer(g_currentVertexBuffer - kRecordShapeSize, kRecordShapeSize, bytes)
      && bytes.size() >= kRecordShapeSize;
    const bool direct = readable
      && recordShapeMatches(reinterpret_cast<const float*>(bytes.data()));
    if (!direct) {
      // A record that is there and does not describe the player aircraft belongs to
      // another object this shader also draws, so that draw keeps its own matrix. Only a
      // record that could not be read at all continues the last placement.
      if (readable)
        return false;
      if (!g_resolvedAircraftValid)
        return false;
      if (!g_resolvedAircraftReused) {
        g_resolvedAircraftReused = true;
        log("ACZ aircraft carry: draw has no record of its own, keeping (%.2f,%.2f,%.2f)",
          static_cast<double>(g_resolvedAircraft[0]),
          static_cast<double>(g_resolvedAircraft[1]),
          static_cast<double>(g_resolvedAircraft[2]));
      }
      out[0] = g_resolvedAircraft[0];
      out[1] = g_resolvedAircraft[1];
      out[2] = g_resolvedAircraft[2];
      for (uint32_t index = 0; index < 9; ++index)
        outBasis[index] = g_resolvedAircraftBasis[index];
      for (uint32_t index = 0; index < 3; ++index)
        outOffset[index] = g_resolvedAircraftOffset[index];
      return true;
    }

    const float* record = reinterpret_cast<const float*>(bytes.data());

    // The record states the aircraft's placement relative to the camera, so the record's own
    // reading is anchored on the camera block's eye. That reading is kept only as the
    // comparison and as the fallback before the engine has published a position.
    float anchor[4] = { camera.eye[0], camera.eye[1], camera.eye[2], 1.0f };

    // The draw's own object matrix is exactly the rotation the shader applies to the
    // model before the record's projection, so copying it is not a reconstruction:
    // it is the value the game rasterises with. Everything after this point only
    // decides where its origin sits in the world.
    const float* objectBasis = record + (0x100 / 4);
    for (uint32_t row = 0; row < 3; ++row) {
      for (uint32_t column = 0; column < 3; ++column)
        outBasis[row * 3 + column] = objectBasis[row * 4 + column];
    }
    // The fourth row of that matrix is where the sub-mesh sits inside the model, in the
    // same frame the placement translation below is expressed in, so the two add. It is
    // zero for the part the model is built around, which is why placing the record alone
    // put the whole aircraft on one point and only the largest part stayed visible.
    for (uint32_t index = 0; index < 3; ++index)
      outOffset[index] = objectBasis[12 + index];

    // The record's projection rows map the draw's frame into clip space, with half
    // the viewport folded in. Removing that offset leaves the view axis as the
    // fourth column and the two screen axes as the first two columns, so the point
    // whose clip position collapses to zero can be solved for directly: it is the
    // camera's position inside the frame. A draw whose placement translation is
    // zero puts its object on the frame origin, so the object's world position is
    // the absolute eye minus that camera position.
    //
    // Those columns are stored pre-rotated by the object's own basis, and the
    // shader cancels that rotation again inside the product it forms, so the matrix
    // the draw really uses is placement * record. The recovered directions have to
    // be taken through the placement as well, otherwise the offset is measured in
    // the object's frame and swings with the object's heading instead of staying
    // fixed to the camera.
    constexpr float kHalfViewport = 2048.0f;
    const float axisLocal[3] = { record[3], record[7], record[11] };
    const float screenXLocal[3] = {
      record[0] - kHalfViewport * axisLocal[0],
      record[4] - kHalfViewport * axisLocal[1],
      record[8] - kHalfViewport * axisLocal[2],
    };
    const float screenYLocal[3] = {
      record[1] - kHalfViewport * axisLocal[0],
      record[5] - kHalfViewport * axisLocal[1],
      record[9] - kHalfViewport * axisLocal[2],
    };
    // The record states the object's offsets in the draw's camera space: how far
    // along the view axis it sits and how far off centre on each screen axis. Those
    // three numbers are the part only the record knows. Their directions come from
    // the camera the scene is rasterised with instead, because the record's own
    // columns describe the camera of its own pass and drift a few degrees away from
    // the one used here as soon as the object turns.
    const float lengthX = std::sqrt(screenXLocal[0] * screenXLocal[0]
      + screenXLocal[1] * screenXLocal[1] + screenXLocal[2] * screenXLocal[2]);
    const float lengthY = std::sqrt(screenYLocal[0] * screenYLocal[0]
      + screenYLocal[1] * screenYLocal[1] + screenYLocal[2] * screenYLocal[2]);
    // These are the projection's half extents, a few hundred either way.
    if (!(lengthX > 1.0f) || !(lengthY > 1.0f))
      return false;
    // The block's view rows are the camera basis in world space: row 2 is the
    // negated view direction, row 1 the up axis and row 0 the right axis.
    const float* basis = camera.view.data();
    float axis[3] = { -basis[8], -basis[9], -basis[10] };
    float screenX[3] = { basis[0], basis[1], basis[2] };
    float screenY[3] = { -basis[4], -basis[5], -basis[6] };
    const float axisLength = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1]
      + axis[2] * axis[2]);
    const float rowXLength = std::sqrt(screenX[0] * screenX[0]
      + screenX[1] * screenX[1] + screenX[2] * screenX[2]);
    const float rowYLength = std::sqrt(screenY[0] * screenY[0]
      + screenY[1] * screenY[1] + screenY[2] * screenY[2]);
    if (!(axisLength > 0.5f) || !(rowXLength > 0.5f) || !(rowYLength > 0.5f))
      return false;
    for (uint32_t index = 0; index < 3; ++index) {
      axis[index] /= axisLength;
      screenX[index] /= rowXLength;
      screenY[index] /= rowYLength;
    }
    const float deltaX = record[12] - kHalfViewport * record[15];
    const float deltaY = record[13] - kHalfViewport * record[15];
    for (uint32_t index = 0; index < 3; ++index) {
      const float cameraInFrame = -record[15] * axis[index]
                                - (deltaX / lengthX) * screenX[index]
                                - (deltaY / lengthY) * screenY[index];
      out[index] = anchor[index] - cameraInFrame;
    }
    if (!std::isfinite(out[0]) || !std::isfinite(out[1]) || !std::isfinite(out[2]))
      return false;

    // Kept beside the record's own reading: the two disagree only while the aircraft is
    // rotating or the camera is pitching, which is exactly what the summary line reports.
    const float outCamera[3] = { out[0], out[1], out[2] };

    // The record also carries the camera directions, written in the object's own frame
    // as the fourth column of its leading matrix and as the two screen columns offset by
    // the half viewport. Taking those through the object basis at +0x100 gives the same
    // directions in world space without borrowing them from the render camera, so the
    // two readings differ only in where the directions came from. Only one of them can
    // reproduce the position the game publishes for an aircraft standing still, which is
    // the point this reports beside it.
    {
      const float* rotation = outBasis;
      float local[3][3] = {
        { axisLocal[0], axisLocal[1], axisLocal[2] },
        { screenXLocal[0], screenXLocal[1], screenXLocal[2] },
        { screenYLocal[0], screenYLocal[1], screenYLocal[2] },
      };
      float own[3][3] = {};
      for (uint32_t vector = 0; vector < 3; ++vector) {
        for (uint32_t column = 0; column < 3; ++column) {
          own[vector][column] = local[vector][0] * rotation[0 * 3 + column]
                              + local[vector][1] * rotation[1 * 3 + column]
                              + local[vector][2] * rotation[2 * 3 + column];
        }
        const float length = std::sqrt(own[vector][0] * own[vector][0]
          + own[vector][1] * own[vector][1] + own[vector][2] * own[vector][2]);
        if (!(length > 0.0f))
          continue;
        for (uint32_t column = 0; column < 3; ++column)
          own[vector][column] /= length;
      }
      // The render-camera directions are the reference orientation, so each recovered
      // direction is flipped onto the side it agrees with. That leaves the angle between
      // them as the only difference, which is what the two readings are being compared on.
      const float* reference[3] = { axis, screenX, screenY };
      bool usable = true;
      for (uint32_t vector = 0; vector < 3; ++vector) {
        const float dotProduct = own[vector][0] * reference[vector][0]
                               + own[vector][1] * reference[vector][1]
                               + own[vector][2] * reference[vector][2];
        if (std::abs(dotProduct) < 0.5f)
          usable = false;
        else if (dotProduct < 0.0f) {
          for (uint32_t column = 0; column < 3; ++column)
            own[vector][column] = -own[vector][column];
        }
      }
      if (usable) {
        float ownWorld[3] = {};
        for (uint32_t index = 0; index < 3; ++index) {
          ownWorld[index] = anchor[index] + record[15] * own[0][index]
                          + (deltaX / lengthX) * own[1][index]
                          + (deltaY / lengthY) * own[2][index];
        }
        // The record's own directions are the ones that belong to this draw, so they
        // define the world position. The render camera's directions stay in the log as
        // the comparison and remain the fallback when the record's cannot be used.
        for (uint32_t index = 0; index < 3; ++index)
          out[index] = ownWorld[index];
      }
    }

    // The engine publishes the position it simulates, one frame ahead of what the
    // recorded draw renders. Read every frame so the summary can show it.
    float targetWorld[3] = {};
    const bool haveTarget = readPlayerTargetWorld(targetWorld);

    // The renderer's position is the engine target read during the previous frame, and the
    // value is advanced once per frame at present time: the aircraft is submitted several
    // times per frame, so advancing it here would put the later submissions a whole frame
    // ahead of the first one, which is what showed up as a second aircraft.
    const float recordWorld[3] = { out[0], out[1], out[2] };
    bool usedCached = false;
    if (haveTarget) {
      for (uint32_t index = 0; index < 3; ++index)
        g_targetSeen[index] = targetWorld[index];
      g_targetSeenValid = true;
      if (g_cachedTargetValid) {
        for (uint32_t index = 0; index < 3; ++index)
          out[index] = g_cachedTarget[index];
        usedCached = true;
      }
    }

    // Self check on the basis: the directions come from the same view rows the
    // scene is drawn with, so these read minus one on the view axis, plus one on
    // the right axis and minus one on the up axis. Anything else means the camera
    // the block was read from is not the camera this draw used.
    const float* view = camera.view.data();
    const float alignAxis = axis[0] * view[8] + axis[1] * view[9] + axis[2] * view[10];
    const float alignX = screenX[0] * view[0] + screenX[1] * view[1] + screenX[2] * view[2];
    const float alignY = screenY[0] * view[4] + screenY[1] * view[5] + screenY[2] * view[6];

    // Second, independent reading of the same record: its fourth row is the clip
    // position of the frame origin, so pushing that through the inverse of the
    // matrix the scene is rasterised with gives the world point without assuming
    // anything about how the record is laid out. Reported next to the first so a
    // disagreement between the two is visible instead of silently averaged.
    float alternate[3] = {};
    if (camera.hasDisplayProjection) {
      std::array<float, 16> viewProjection {};
      std::array<float, 16> inverseViewProjection {};
      multiplyRowVectorMatrices(camera.view.data(), camera.displayProjection.data(),
        viewProjection.data());
      if (invertMatrix4(viewProjection.data(), inverseViewProjection.data())) {
        float world[4] = {};
        for (uint32_t column = 0; column < 4; ++column) {
          world[column] = record[12] * inverseViewProjection[0 + column]
                        + record[13] * inverseViewProjection[4 + column]
                        + record[14] * inverseViewProjection[8 + column]
                        + record[15] * inverseViewProjection[12 + column];
        }
        if (std::abs(world[3]) > 1.0e-6f) {
          for (uint32_t component = 0; component < 3; ++component)
            alternate[component] = world[component] / world[3];
        }
      }
    }

    // One line per distinct state: the camera eye, the position actually used, the
    // record-derived position and the engine target, so the disagreement between them
    // stays visible while the placement is being settled. The record's rotation basis and
    // the camera's own world axes are carried as well, because how far the aircraft has
    // rolled on screen depends on both: it is the angle between them, and a mismatch
    // between the two is what makes the roll look larger than the game renders it.
    const float logged[31] = {
      camera.eye[0], camera.eye[1], camera.eye[2],
      out[0], out[1], out[2],
      recordWorld[0], recordWorld[1], recordWorld[2],
      targetWorld[0], targetWorld[1], targetWorld[2],
      record[15], outOffset[0], outOffset[1], outOffset[2],
      outBasis[0], outBasis[1], outBasis[2],
      outBasis[3], outBasis[4], outBasis[5],
      outBasis[6], outBasis[7], outBasis[8],
      screenX[0], screenX[1], screenX[2],
      screenY[0], screenY[1], screenY[2] };
    bool changed = false;
    for (uint32_t index = 0; index < 31; ++index)
      changed = changed || logged[index] != g_resolvedAircraftLogged[index];
    static D3D12_GPU_VIRTUAL_ADDRESS loggedVertexBuffer = 0;
    static uint32_t loggedIndexCount = 0;
    changed = changed || g_currentVertexBuffer != loggedVertexBuffer
                      || g_currentIndexCount != loggedIndexCount;
    if (changed) {
      loggedVertexBuffer = g_currentVertexBuffer;
      loggedIndexCount = g_currentIndexCount;
      std::memcpy(g_resolvedAircraftLogged, logged, sizeof(logged));
      log("ACZ aircraft: eye=(%.2f,%.2f,%.2f) pos=(%.2f,%.2f,%.2f) off=(%.2f,%.2f,%.2f) rec=(%.2f,%.2f,%.2f) target=(%.2f,%.2f,%.2f) depth=%.2f cached=%d tgt=%d vb=0x%llx idx=%u rot=(%.4f,%.4f,%.4f)(%.4f,%.4f,%.4f)(%.4f,%.4f,%.4f) cr=(%.4f,%.4f,%.4f) cu=(%.4f,%.4f,%.4f)",
        static_cast<double>(camera.eye[0]), static_cast<double>(camera.eye[1]),
        static_cast<double>(camera.eye[2]),
        static_cast<double>(out[0]), static_cast<double>(out[1]),
        static_cast<double>(out[2]),
        static_cast<double>(outOffset[0]), static_cast<double>(outOffset[1]),
        static_cast<double>(outOffset[2]),
        static_cast<double>(recordWorld[0]), static_cast<double>(recordWorld[1]),
        static_cast<double>(recordWorld[2]),
        static_cast<double>(targetWorld[0]), static_cast<double>(targetWorld[1]),
        static_cast<double>(targetWorld[2]),
        static_cast<double>(record[15]), usedCached ? 1 : 0, haveTarget ? 1 : 0,
        static_cast<unsigned long long>(g_currentVertexBuffer), g_currentIndexCount,
        static_cast<double>(outBasis[0]), static_cast<double>(outBasis[1]),
        static_cast<double>(outBasis[2]), static_cast<double>(outBasis[3]),
        static_cast<double>(outBasis[4]), static_cast<double>(outBasis[5]),
        static_cast<double>(outBasis[6]), static_cast<double>(outBasis[7]),
        static_cast<double>(outBasis[8]),
        static_cast<double>(screenX[0]), static_cast<double>(screenX[1]),
        static_cast<double>(screenX[2]),
        static_cast<double>(screenY[0]), static_cast<double>(screenY[1]),
        static_cast<double>(screenY[2]));
    }

    g_resolvedAircraft[0] = out[0];
    g_resolvedAircraft[1] = out[1];
    g_resolvedAircraft[2] = out[2];
    for (uint32_t index = 0; index < 9; ++index)
      g_resolvedAircraftBasis[index] = outBasis[index];
    for (uint32_t index = 0; index < 3; ++index)
      g_resolvedAircraftOffset[index] = outOffset[index];
    g_resolvedAircraftValid = true;
    g_resolvedAircraftReused = false;
    return true;
  }

  // ACZ draws the player aircraft with a zero translation, so the transform that resolves
  // from the bindings this layer can read carries rotation only. Its position comes from
  // the engine, passed through the cache that runs one frame behind the simulation.
  // Reports whether the draw's own record described the player aircraft, which is what
  // decides that this draw is one of its sub-meshes.
  bool placeAircraftInCameraFrame(
      const AczCameraMatrices& camera,
      std::array<float, 16>& objectToWorld,
      uint8_t& objectTransformSource) {
    // The record wins over the rig reconstruction whenever it can be resolved.
    {
      float resolved[3] {};
      float resolvedBasis[9] {};
      float resolvedOffset[3] {};
      if (resolveRecordWorld(camera, resolved, resolvedBasis, resolvedOffset)) {
        // The vertex stream already carries the camera orientation: the object matrix
        // in the record cancels against the record's projection, so the game folds the
        // object's rotation into the vertices themselves. Submitting the mesh with an
        // identity matrix therefore still makes it follow the camera, and tipping the
        // view up tips the aircraft into the ground. The inverse of the camera rotation
        // undoes it, which is the transpose of the view basis in row vector form.
        // The vertex stream is model space: a captured draw's first vertex is
        // (-14.754, -1.0937, 3.812), the same values the record's own shader reads,
        // so the object's orientation cannot come from the vertices. It comes from
        // the record's own object matrix at +0x100, which a capture taken with the
        // camera turned 45 degrees shows as the identity while the record's view
        // axis carries the 45 degrees: the matrix is the object's world rotation,
        // not the camera's.
        for (uint32_t row = 0; row < 3; ++row) {
          for (uint32_t column = 0; column < 3; ++column)
            objectToWorld[row * 4 + column] = resolvedBasis[row * 3 + column];
          objectToWorld[row * 4 + 3] = 0.0f;
        }
        // The record's matrix is one frame ahead of the camera the frame is submitted
        // with, the same way its position is ahead of the engine target, so each sub-mesh
        // is held back by one frame. The cache is keyed by the order the sub-meshes are
        // drawn in, which is the only identity that survives a frame: the record's offset
        // rotates with the aircraft and the vertex buffer address is recycled.
        //
        // The body's rotation marks a drawn frame. When it changes, the slot cursor and
        // every slot's prev advance, so all sub-meshes of one frame read the attitude of
        // the previous frame whatever order they are drawn in.
        {
          bool newFrame = false;
          if (g_currentIndexCount == 1695) {
            for (uint32_t index = 0; index < 9; ++index) {
              if (!g_frameMarkerValid || resolvedBasis[index] != g_frameMarker[index]) {
                newFrame = true;
                break;
              }
            }
            if (newFrame) {
              for (uint32_t slot = 0; slot < kAczPartSlots; ++slot) {
                if (!g_partSlots[slot].basisValid)
                  continue;
                for (uint32_t element = 0; element < 9; ++element)
                  g_partSlots[slot].prev[element] = g_partSlots[slot].basis[element];
                for (uint32_t element = 0; element < 3; ++element)
                  g_partSlots[slot].prevOffset[element] = g_partSlots[slot].offset[element];
                g_partSlots[slot].prevValid = true;
              }
              for (uint32_t index = 0; index < 9; ++index)
                g_frameMarker[index] = resolvedBasis[index];
              g_frameMarkerValid = true;
              g_partSlotIndex = 0;
              static bool loggedLag = false;
              if (!loggedLag) {
                loggedLag = true;
                log("ACZ aircraft lag: sub-mesh attitude and offset held back one frame");
              }
            }
          }
          const uint32_t slotIndex = g_partSlotIndex % kAczPartSlots;
          ++g_partSlotIndex;
          AczPartSlot& slot = g_partSlots[slotIndex];
          float chosen[9] {};
          if (slot.prevValid) {
            for (uint32_t index = 0; index < 9; ++index)
              chosen[index] = slot.prev[index];
          } else {
            for (uint32_t index = 0; index < 9; ++index)
              chosen[index] = resolvedBasis[index];
          }
          for (uint32_t index = 0; index < 9; ++index)
            slot.basis[index] = resolvedBasis[index];
          for (uint32_t index = 0; index < 3; ++index)
            slot.offset[index] = resolvedOffset[index];
          slot.basisValid = true;
          // The offset rotates with the aircraft, so it has to come from the same frame as
          // the attitude. Pairing a current offset with a lagged attitude makes every part
          // off the model origin swing one frame further than the fuselage.
          if (slot.prevValid) {
            for (uint32_t index = 0; index < 3; ++index)
              resolvedOffset[index] = slot.prevOffset[index];
          }
          for (uint32_t row = 0; row < 3; ++row) {
            for (uint32_t column = 0; column < 3; ++column)
              objectToWorld[row * 4 + column] = chosen[row * 3 + column];
          }
        }
        // The record places the model the part belongs to, and the part's own offset is
        // part of the same matrix, so the two add to give where this sub-mesh sits.
        objectToWorld[12] = resolved[0] + resolvedOffset[0];
        objectToWorld[13] = resolved[1] + resolvedOffset[1];
        objectToWorld[14] = resolved[2] + resolvedOffset[2];
        objectToWorld[15] = 1.0f;
        // Marks this draw as deliberately placed. The backend drops draws whose
        // matrix is the identity because the game uses that to mean "the placement
        // lives in the fused matrix", which does not apply here: the translation
        // above is a real world position, and the rotation can legitimately be the
        // identity while the camera looks down a world axis.
        objectTransformSource = 8;
        // The record only says where the model sits; the rotation and any scale on
        // the mesh are whatever the draw already carried. Reported on change, since
        // a mesh that is tilted or scaled wrongly is the one remaining way the
        // aircraft can end up under the ground while its origin is correct.
        static float lastMatrix[16] = {};
        bool changed = false;
        for (uint32_t index = 0; index < 16; ++index)
          changed = changed || objectToWorld[index] != lastMatrix[index];
        if (changed) {
          std::memcpy(lastMatrix, objectToWorld.data(), sizeof(lastMatrix));
          log("ACZ aircraft matrix: (%.4f,%.4f,%.4f,%.4f)(%.4f,%.4f,%.4f,%.4f)(%.4f,%.4f,%.4f,%.4f)(%.4f,%.4f,%.4f,%.4f)",
            static_cast<double>(objectToWorld[0]), static_cast<double>(objectToWorld[1]),
            static_cast<double>(objectToWorld[2]), static_cast<double>(objectToWorld[3]),
            static_cast<double>(objectToWorld[4]), static_cast<double>(objectToWorld[5]),
            static_cast<double>(objectToWorld[6]), static_cast<double>(objectToWorld[7]),
            static_cast<double>(objectToWorld[8]), static_cast<double>(objectToWorld[9]),
            static_cast<double>(objectToWorld[10]), static_cast<double>(objectToWorld[11]),
            static_cast<double>(objectToWorld[12]), static_cast<double>(objectToWorld[13]),
            static_cast<double>(objectToWorld[14]), static_cast<double>(objectToWorld[15]));
        }
        return true;
      }
    }
    return false;
  }
  // DrawACM_v reads its record through t10 as a structured buffer whose element is
  // 416 bytes, and the descriptor naming it is the tenth slot of the SRV table. The
  // view base is the record address, so the record is read straight from it and the
  // object matrix is reported from its first 64 bytes.
  // SrtData is 416 bytes: five 4x4 matrices followed by six float4 terms.
  constexpr size_t kRecordBytes = 416;

  // Reports what the descriptor points at. The buffers holding the records are
  // created through device interfaces this layer does not register, so the view
  // registry is the only record of their resource; this inspects that record
  // without touching the resource itself, because the view registry keeps a raw
  // pointer that may already have been released.
  void reportRecordDescriptor(
      D3D12_CPU_DESCRIPTOR_HANDLE descriptor,
      D3D12_GPU_VIRTUAL_ADDRESS recordAddress) {
    std::lock_guard lock(g_frameMutex);
    const auto it = g_descriptors.find(descriptor.ptr);
    if (it == g_descriptors.end()) {
      log("ACZ t10 view: found=0");
      return;
    }
    const DescriptorMetadata& metadata = it->second;
    const UINT64 offset = recordAddress >= metadata.resourceGpuAddress
      ? recordAddress - metadata.resourceGpuAddress
      : 0;
    const bool covered = metadata.resourceDescription.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
      && metadata.resourceGpuAddress
      && recordAddress >= metadata.resourceGpuAddress
      && offset + kRecordBytes <= metadata.resourceDescription.Width;
    log("ACZ t10 view: found=1 dimension=%u width=%llu gpu=0x%llx off=%llu covered=%d",
      static_cast<unsigned>(metadata.resourceDescription.Dimension),
      static_cast<unsigned long long>(metadata.resourceDescription.Width),
      static_cast<unsigned long long>(metadata.resourceGpuAddress),
      static_cast<unsigned long long>(offset),
      covered ? 1 : 0);
  }

  // Lists the buffer resources the descriptor registry knows about. The record is
  // a StructuredBuffer<SrtData> read through a buffer view, so a buffer resource of
  // the right width identifies it. The view description is optional in D3D12 and
  // the game leaves it null, so the resource description is what has to be used.
  // Reported once per run.
  void reportStructuredBufferViews() {
    static std::atomic<bool> reportedOnce { false };
    if (reportedOnce.exchange(true))
      return;
    std::lock_guard lock(g_frameMutex);
    uint32_t reported = 0;
    for (const auto& entry : g_descriptors) {
      const DescriptorMetadata& view = entry.second;
      if (view.resourceDescription.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
        continue;
      const bool hasView = view.viewDescriptionSize
        >= sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC);
      const auto* srv = hasView
        ? reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(
            view.viewDescription.data())
        : nullptr;
      if (reported >= 96)
        break;
      ++reported;
      log("ACZ t10 struct: cpu=0x%llx hasView=%d stride=%u elements=%u first=%llu width=%llu gpu=0x%llx format=%u",
        static_cast<unsigned long long>(entry.first),
        hasView ? 1 : 0,
        srv ? srv->Buffer.StructureByteStride : 0u,
        srv ? srv->Buffer.NumElements : 0u,
        srv ? static_cast<unsigned long long>(srv->Buffer.FirstElement) : 0ull,
        static_cast<unsigned long long>(view.resourceDescription.Width),
        static_cast<unsigned long long>(view.resourceGpuAddress),
        static_cast<unsigned>(view.format));
    }
    log("ACZ t10 struct: total=%u", reported);
  }

  void reportSrtDescriptor(const D3D12_GPU_DESCRIPTOR_HANDLE* tables, uint32_t tableCount) {
    if (!tables)
      return;
    for (uint32_t table = 0; table < tableCount; ++table) {
      if (!tables[table].ptr)
        continue;
      for (const auto& entry : g_descriptorHeaps) {
        const DescriptorHeapMetadata& meta = entry.second;
        if (meta.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
          continue;
        if (!meta.cpuStart.ptr || !meta.gpuStart.ptr || !meta.increment)
          continue;
        if (tables[table].ptr < meta.gpuStart.ptr)
          continue;
        const UINT64 span = static_cast<UINT64>(meta.increment) * meta.count;
        if (tables[table].ptr >= meta.gpuStart.ptr + span)
          continue;
        const UINT64 delta = tables[table].ptr - meta.gpuStart.ptr;
        if ((delta % meta.increment) != 0)
          continue;
        const uint32_t index = static_cast<uint32_t>(delta / meta.increment) + 10u;
        if (index >= meta.count)
          continue;
        const UINT64 tableCpu = meta.cpuStart.ptr + delta;
        const auto* words = reinterpret_cast<const uint32_t*>(
          meta.cpuStart.ptr + static_cast<SIZE_T>(index) * meta.increment);
        const D3D12_GPU_VIRTUAL_ADDRESS recordAddress =
          static_cast<UINT64>(words[0]) | (static_cast<UINT64>(words[1]) << 32);
        auto stateIt = g_recordStates.find(recordAddress);
        if (stateIt == g_recordStates.end()) {
          if (g_recordStates.size() >= kReportedRecordLimit)
            return;
          stateIt = g_recordStates.emplace(recordAddress, RecordState {}).first;
        }
        RecordState& state = stateIt->second;
        const bool newAddress = !state.described;
        if (newAddress) {
          state.described = true;
          log("ACZ t10: table=%u slot=%u va=0x%llx size=%u w3=0x%x cpu=0x%llx",
            table, index, static_cast<unsigned long long>(recordAddress), words[2], words[3],
            static_cast<unsigned long long>(static_cast<UINT64>(meta.cpuStart.ptr)
              + static_cast<UINT64>(index) * meta.increment));
          // The record lives in a default heap, so the host can only ever see it
          // after a copy into that memory has been mirrored. Noting the address
          // arms the mirror for the next copy that lands on it.
          noteRecordAddress(recordAddress);
        }

        std::vector<uint8_t> record;
        const bool haveRecord = recordAddress
          && (readRecordShadow(recordAddress, 416, record)
           || readGpuBuffer(recordAddress, 416, record))
          && record.size() >= 416;
        // The buffers holding the records are created through device interfaces
        // that never reach the address registry, so the view registry is the only
        // route to their resource.
        // The shader's register index is not the slot index: a root descriptor
        // The record is a StructuredBuffer<SrtData>, so its view carries a
        // 416-byte structure stride. The register index is not the slot index, and
        // the slot it points at holds a texture, so the table is scanned in raw
        // memory for that stride instead of offsetting from the register number.
        if (g_scannedTables.emplace(tableCpu).second) {
          for (uint32_t candidate = 0; candidate < 256; ++candidate) {
            const UINT64 offset = static_cast<UINT64>(candidate) * meta.increment;
            if (offset + meta.increment > static_cast<UINT64>(meta.count) * meta.increment)
              break;
            const auto* w = reinterpret_cast<const uint32_t*>(tableCpu + offset);
            if (w[3] != kRecordBytes)
              continue;
            const D3D12_GPU_VIRTUAL_ADDRESS va =
              static_cast<UINT64>(w[0]) | (static_cast<UINT64>(w[1]) << 32);
            log("ACZ t10 hit: slot=%u va=0x%llx size=%u stride=%u",
              candidate, static_cast<unsigned long long>(va), w[2], w[3]);
          }
          log("ACZ t10 hit: scanned table=0x%llx", static_cast<unsigned long long>(tableCpu));
        }
        if (!haveRecord && recordAddress) {
          const D3D12_CPU_DESCRIPTOR_HANDLE descriptor {
            meta.cpuStart.ptr + static_cast<SIZE_T>(index) * meta.increment };
          if (g_reportedDescriptors.emplace(descriptor.ptr).second)
            reportRecordDescriptor(descriptor, recordAddress);
        }
        if (!haveRecord) {
          if (newAddress) {
            uint32_t heapType = 0;
            UINT64 bufferSize = 0;
            const int probe = probeGpuBuffer(recordAddress, 416, heapType, bufferSize);
            const RegistryCoverage coverage = describeRegistry(recordAddress, 416);
            log("ACZ t10: unreadable probe=%d heap=%u size=%llu total=%zu near=0x%llx+%llu off=%llu covered=%d",
              probe, heapType, static_cast<unsigned long long>(bufferSize),
              coverage.resourceCount,
              static_cast<unsigned long long>(coverage.nearestBase),
              static_cast<unsigned long long>(coverage.nearestSize),
              static_cast<unsigned long long>(coverage.offset),
              coverage.covered ? 1 : 0);
          }
          return;
        }
        if (state.readable)
          return;
        state.readable = true;
        const auto* r = reinterpret_cast<const float*>(record.data());
        log("ACZ t10 m0: t=(%.2f,%.2f,%.2f) r=(%.3f,%.3f,%.3f)(%.3f,%.3f,%.3f)(%.3f,%.3f,%.3f) w=%.3f",
          static_cast<double>(r[12]), static_cast<double>(r[13]), static_cast<double>(r[14]),
          static_cast<double>(r[0]), static_cast<double>(r[1]), static_cast<double>(r[2]),
          static_cast<double>(r[4]), static_cast<double>(r[5]), static_cast<double>(r[6]),
          static_cast<double>(r[8]), static_cast<double>(r[9]), static_cast<double>(r[10]),
          static_cast<double>(r[15]));
        log("ACZ t10 m4: (%.2f,%.2f,%.2f,%.2f)(%.2f,%.2f,%.2f,%.2f)(%.2f,%.2f,%.2f,%.2f)(%.2f,%.2f,%.2f,%.2f)",
          static_cast<double>(r[64]), static_cast<double>(r[65]),
          static_cast<double>(r[66]), static_cast<double>(r[67]),
          static_cast<double>(r[68]), static_cast<double>(r[69]),
          static_cast<double>(r[70]), static_cast<double>(r[71]),
          static_cast<double>(r[72]), static_cast<double>(r[73]),
          static_cast<double>(r[74]), static_cast<double>(r[75]),
          static_cast<double>(r[76]), static_cast<double>(r[77]),
          static_cast<double>(r[78]), static_cast<double>(r[79]));

        // The record's leading matrix maps the model origin to the clip position
        // the game rasterises the aircraft at, so pushing that clip position back
        // through the camera the game rendered with yields its world position.
        // Nothing here is fitted: both matrices come from the game.
        {
          AczCameraMatrices recordCamera {};
          if (buildAczCamera(recordCamera) && recordCamera.hasDisplayProjection) {
            std::array<float, 16> viewProjection {};
            multiplyRowVectorMatrices(recordCamera.view.data(),
              recordCamera.displayProjection.data(), viewProjection.data());
            std::array<float, 16> inverseViewProjection {};
            if (invertMatrix4(viewProjection.data(), inverseViewProjection.data())) {
              float world[4] = {};
              for (uint32_t column = 0; column < 4; ++column) {
                world[column] = r[12] * inverseViewProjection[0 + column]
                              + r[13] * inverseViewProjection[4 + column]
                              + r[14] * inverseViewProjection[8 + column]
                              + r[15] * inverseViewProjection[12 + column];
              }
              if (std::isfinite(world[3]) && std::abs(world[3]) > 1.0e-6f) {
                float resolved[3] = {
                  world[0] / world[3], world[1] / world[3], world[2] / world[3] };
                const bool finite = std::isfinite(resolved[0])
                  && std::isfinite(resolved[1]) && std::isfinite(resolved[2]);
                if (finite) {
                  g_recordWorld[0] = resolved[0];
                  g_recordWorld[1] = resolved[1];
                  g_recordWorld[2] = resolved[2];
                  g_recordWorldValid.store(true, std::memory_order_relaxed);
                  if (!g_recordWorldReported.exchange(true)) {
                    const double dx = static_cast<double>(resolved[0]) - recordCamera.eye[0];
                    const double dy = static_cast<double>(resolved[1]) - recordCamera.eye[1];
                    const double dz = static_cast<double>(resolved[2]) - recordCamera.eye[2];
                    log("ACZ record world: w=(%.2f,%.2f,%.2f) eye=(%.2f,%.2f,%.2f) eyeDist=%.2f depth=%.3f",
                      static_cast<double>(resolved[0]), static_cast<double>(resolved[1]),
                      static_cast<double>(resolved[2]),
                      static_cast<double>(recordCamera.eye[0]),
                      static_cast<double>(recordCamera.eye[1]),
                      static_cast<double>(recordCamera.eye[2]),
                      std::sqrt(dx * dx + dy * dy + dz * dz),
                      static_cast<double>(r[15]));
                  }
                }
              }
            }
          }
        }
        return;
      }
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
    g_currentVertexBuffer = vertexBuffer.BufferLocation;
    g_currentIndexCount = indexCount;
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
    // DrawACM_v reads its record through t10 as a StructuredBuffer<SrtData> whose
    // element is 416 bytes. A root SRV never appears in a descriptor table, so the
    // root slots are the only place it can be bound, and the object CBV holds the
    // eight-byte SrtInputData rather than a matrix.
    if (nativePipelineInfo.vertexShaderHash == 0xb6a7d1d096ce8fe7ull) {
      reportSrtDescriptor(graphicsDescriptorTables, 16u);
      verifyRecordBeforeVertexBuffer(vertexBuffer);
      // The record is bound to register t10, and a table that does not start at t0
      // hides which slot that is. A root descriptor is not in any table, so the
      // root SRV slots and the table pointers are reported once as well.
      static std::atomic<bool> reportedAircraftRootSrvs { false };
      if (!reportedAircraftRootSrvs.exchange(true)) {
        for (uint32_t slot = 0; slot < 16; ++slot) {
          const D3D12_GPU_VIRTUAL_ADDRESS address =
            graphicsRootSrvs ? graphicsRootSrvs[slot] : 0;
          const D3D12_GPU_VIRTUAL_ADDRESS table =
            graphicsDescriptorTables ? graphicsDescriptorTables[slot].ptr : 0;
          log("ACZ root bind: index=%u srv=0x%llx table=0x%llx", slot,
            static_cast<unsigned long long>(address),
            static_cast<unsigned long long>(table));
        }
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
    // Screen space families transform their vertices on the CPU already, so
    // submitting them as world geometry can only produce garbage. They are
    // dropped here, before any placement work, and counted so the routing itself
    // stays visible.
    const AczPlacementMode placementMode =
      aczPlacementModeFor(nativePipelineInfo.vertexShaderHash);
    if (placementMode == AczPlacementMode::kScreenSpace) {
      static std::atomic<uint32_t> screenSpaceLogs { 0 };
      const uint32_t index = screenSpaceLogs.fetch_add(1, std::memory_order_relaxed);

      return;
    }
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
      // A matrix without a translation is a rotation only: moving objects keep
      // their placement in another binding, so such a matrix is not a placement
      // and the remaining slots are probed instead of accepting it.
      const auto lacksPlacement = [&]() {
        if (!nativePipelineInfo.hasObjectToWorld)
          return true;
        const auto& m = nativePipelineInfo.objectToWorld;
        return isIdentityObjectTransform(m)
          || std::abs(m[12]) + std::abs(m[13]) + std::abs(m[14]) < 1.0f;
      };
      // Some pipelines bind their per-draw constants in roots 3..5 instead of
      // root 2, so those slots are probed before giving up on the transform.
      if (lacksPlacement()) {
        for (uint32_t slot = 3; slot <= 5; ++slot) {
          const auto address = currentGraphicsRootCbv(slot);
          if (!address)
            continue;
          std::array<float, 16> candidate {};
          bool found = false;
          for (uint32_t offset = 0; offset <= 0x3c0; offset += 0x40) {
            if (!readObjectTransform(address, candidate, offset)
             || isIdentityObjectTransform(candidate)
             || std::abs(candidate[12]) + std::abs(candidate[13]) + std::abs(candidate[14]) < 1.0f)
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

      // The model shaders read their matrices from a structured buffer whose
      // binding the draw trace cannot map, so the descriptors bound to this draw
      // are read straight out of the heap and every word that looks like a GPU
      // address is probed. A placement is accepted when it is a rigid matrix that
      // lands near the camera, which is what separates a real object matrix from
      // the material blocks and viewport tables that share the same heap.
      if (!fusedDecoded && indexCount >= 20
       && isModelVertexShader(nativePipelineInfo.vertexShaderHash)) {
        constexpr uint32_t kMaxProbes = 6;
        constexpr uint32_t kProbeBytes = 0x140;

        D3D12_GPU_VIRTUAL_ADDRESS probes[kMaxProbes] = {};
        uint32_t probeCount = 0;
        const auto addProbe = [&](D3D12_GPU_VIRTUAL_ADDRESS address) {
          if (!address || probeCount >= kMaxProbes)
            return;
          for (uint32_t i = 0; i < probeCount; ++i) {
            if (probes[i] == address)
              return;
          }
          probes[probeCount++] = address;
        };
        addProbe(objectTransformCbv);
        for (uint32_t slot = 0; slot <= 1; ++slot)
          addProbe(currentGraphicsRootCbv(slot));
        collectDescriptorAddresses(graphicsDescriptorTables, probes, probeCount, kMaxProbes);

        AczCameraMatrices camera {};
        const bool haveCamera = buildAczCamera(camera);
        if (!haveCamera) {
          // Without a camera nothing can be validated, so the probe is skipped
          // instead of spending reads on candidates that cannot be accepted.

          probeCount = 0;
        }

        // A placement is a rigid affine matrix near the camera.
        const auto rigidPlacement = [&](const std::array<float, 16>& matrix) {
          if (!haveCamera)
            return false;
          if (std::abs(matrix[3]) > 0.02f || std::abs(matrix[7]) > 0.02f
           || std::abs(matrix[11]) > 0.02f || std::abs(matrix[15] - 1.0f) > 0.02f)
            return false;
          const float r0 = std::sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2]);
          const float r1 = std::sqrt(matrix[4] * matrix[4] + matrix[5] * matrix[5] + matrix[6] * matrix[6]);
          const float r2 = std::sqrt(matrix[8] * matrix[8] + matrix[9] * matrix[9] + matrix[10] * matrix[10]);
          if (r0 < 1.0e-3f || r1 < 1.0e-3f || r2 < 1.0e-3f)
            return false;
          const float largest = std::max(r0, std::max(r1, r2));
          const float smallest = std::min(r0, std::min(r1, r2));
          if (largest > smallest * 1.02f)
            return false;
          const float dot01 = (matrix[0] * matrix[4] + matrix[1] * matrix[5] + matrix[2] * matrix[6]) / (r0 * r1);
          const float dot02 = (matrix[0] * matrix[8] + matrix[1] * matrix[9] + matrix[2] * matrix[10]) / (r0 * r2);
          const float dot12 = (matrix[4] * matrix[8] + matrix[5] * matrix[9] + matrix[6] * matrix[10]) / (r1 * r2);
          if (std::abs(dot01) > 0.02f || std::abs(dot02) > 0.02f || std::abs(dot12) > 0.02f)
            return false;
          const float dx = matrix[12] - camera.eye[0];
          const float dy = matrix[13] - camera.eye[1];
          const float dz = matrix[14] - camera.eye[2];
          const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
          return distance > 1.0f && distance < 3000.0f;
        };

        for (uint32_t i = 0; i < probeCount && !fusedDecoded; ++i) {
          std::vector<uint8_t> block;
          if (!readGpuBuffer(probes[i], kProbeBytes, block) || block.size() < kProbeBytes)
            continue;

          // Raw sample of what each candidate address holds, so the location of the
          // structured buffer can be told apart from the material blocks by content.
          {
            static std::atomic<uint32_t> blockLogs { 0 };
            if (blockLogs.fetch_add(1, std::memory_order_relaxed) < 12) {
              const auto* fused = reinterpret_cast<const float*>(block.data());
              const auto* object = reinterpret_cast<const float*>(block.data() + 0x100);
            }
          }

          // Mode one: the placement is the object matrix at 0x100.
          std::array<float, 16> objectMatrix {};
          std::memcpy(objectMatrix.data(), block.data() + 0x100, sizeof(float) * 16);
          if (rigidPlacement(objectMatrix)) {

            nativePipelineInfo.objectToWorld = objectMatrix;
            nativePipelineInfo.hasObjectToWorld = true;
            nativePipelineInfo.objectTransformSource = 6;
            nativePipelineInfoPtr = &nativePipelineInfo;
            fusedDecoded = true;
            break;
          }

          // Mode two: the placement is fused into the leading matrix.
          std::array<float, 16> fusedMatrix {};
          std::memcpy(fusedMatrix.data(), block.data(), sizeof(float) * 16);
          std::array<float, 16> decoded {};
          if (!composeFusedMatrix(fusedMatrix, basePlacement, decoded))
            continue;
          if (!rigidPlacement(decoded))
            continue;

          nativePipelineInfo.objectToWorld = decoded;
          nativePipelineInfo.hasObjectToWorld = true;
          nativePipelineInfo.objectTransformSource = 5;
          nativePipelineInfoPtr = &nativePipelineInfo;
          fusedDecoded = true;
        }



        // Periodic sample so the in-mission state is visible even after the first
        // few draws have spent the counters above. A draw counter is used rather
        // than the frame id because a frame id rarely lands on a model draw.

      }
    }
    // The model extent is sampled once for the player and once for another
    // aircraft. Equal extents mean the viewing distance is what is off, while a
    // larger player model means the record carries a scale this layer drops.
    {
      const bool isAircraft = nativePipelineInfo.vertexShaderHash == 0xb6a7d1d096ce8fe7ull;
      const bool zeroTranslation =
        std::abs(nativePipelineInfo.objectToWorld[12]) < 1.0f
        && std::abs(nativePipelineInfo.objectToWorld[13]) < 1.0f
        && std::abs(nativePipelineInfo.objectToWorld[14]) < 1.0f;
      static float measuredExtent[6] = {};
      const bool wanted = isAircraft && zeroTranslation;
      if (wanted && vertexBuffer.StrideInBytes) {
        uint32_t positionOffset = 0;
        for (const auto& element : nativePipelineInfo.inputElements) {
          if (element.slot != 0)
            continue;
          std::string semantic = element.semantic;
          std::transform(semantic.begin(), semantic.end(), semantic.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
          if (semantic == "POSITION") {
            positionOffset = element.offset;
            break;
          }
        }
        std::vector<uint8_t> bytes;
        const size_t wanted = static_cast<size_t>(vertexBuffer.StrideInBytes) * 256;
        if (readGpuBuffer(vertexBuffer.BufferLocation, wanted, bytes)) {
          const uint32_t stride = vertexBuffer.StrideInBytes;
          const uint32_t count = static_cast<uint32_t>(bytes.size() / stride);
          float low[3] = { 1.0e30f, 1.0e30f, 1.0e30f };
          float high[3] = { -1.0e30f, -1.0e30f, -1.0e30f };
          for (uint32_t i = 0; i < count; ++i) {
            const auto* point = reinterpret_cast<const float*>(
              bytes.data() + static_cast<size_t>(i) * stride + positionOffset);
            for (uint32_t axis = 0; axis < 3; ++axis) {
              low[axis] = std::min(low[axis], point[axis]);
              high[axis] = std::max(high[axis], point[axis]);
            }
          }
          // Reported on change rather than once, because the question this answers
          // is whether the vertex stream itself moves with the aircraft: a stream
          // the game has already rotated and offset is drawn correctly with the
          // camera position alone, while a static model space one is not.
          const float measured[6] = {
            low[0], low[1], low[2], high[0], high[1], high[2] };
          bool changed = false;
          for (uint32_t index = 0; index < 6; ++index)
            changed = changed || measured[index] != measuredExtent[index];
          if (changed) {
            std::memcpy(measuredExtent, measured, sizeof(measured));
            const auto* first = reinterpret_cast<const float*>(
              bytes.data() + positionOffset);
            log("ACZ model extent: idx=%u verts=%u stride=%u pos=%u min=(%.2f,%.2f,%.2f) max=(%.2f,%.2f,%.2f) v0=(%.2f,%.2f,%.2f)",
              indexCount, count, stride, positionOffset,
              static_cast<double>(low[0]), static_cast<double>(low[1]),
              static_cast<double>(low[2]),
              static_cast<double>(high[0]), static_cast<double>(high[1]),
              static_cast<double>(high[2]),
              static_cast<double>(first[0]), static_cast<double>(first[1]),
              static_cast<double>(first[2]));
          }
        }
      }
    }
    // The player aircraft is drawn with a record the game centres on the view, and this
    // layer reconstructs its placement instead of reading it, so the draw is admitted on
    // that record alone. Its object matrix cannot be the admission test: the matrix of a
    // sub-mesh carries that sub-mesh's own offset, so requiring a zero translation there
    // admitted only the parts that sit on the model origin, and the aircraft was drawn as
    // the single piece those parts make up.
    if (nativePipelineInfo.vertexShaderHash == 0xb6a7d1d096ce8fe7ull) {
      // Only the first submission of a mesh is kept: the highlight pass repeats vertices
      // that are already in the scene. A part drawn once and blended, such as glass or a
      // decal, has no earlier submission and is still submitted.
      if (nativePipelineInfo.blendEnabled
       && placedMeshSubmitted(g_currentVertexBuffer, indexCount)) {
        static std::unordered_set<uint64_t> loggedSkips;
        const uint64_t skipKey = (static_cast<uint64_t>(g_currentVertexBuffer) << 8)
          ^ static_cast<uint64_t>(indexCount);
        if (loggedSkips.emplace(skipKey).second) {
          log("ACZ aircraft skip: idx=%u vb=0x%llx", indexCount,
            static_cast<unsigned long long>(g_currentVertexBuffer));
        }
        return;
      }
      // The camera the scene is rendered with takes priority: the vertex stream
      // carries that camera's rotation, so undoing it needs the same matrix, not a
      // block read at a different point in the frame.
      AczCameraMatrices playerCamera {};
      bool haveCamera = false;
      float remixView[16] = {};
      float remixEye[3] = {};
      if (readD3D12CameraMatrices(remixView, remixEye)) {
        std::memcpy(playerCamera.view.data(), remixView, sizeof(remixView));
        for (uint32_t axis = 0; axis < 3; ++axis)
          playerCamera.eye[axis] = remixEye[axis];
        haveCamera = true;
      } else {
        haveCamera = buildAczCamera(playerCamera);
      }
      if (haveCamera
       && placeAircraftInCameraFrame(playerCamera, nativePipelineInfo.objectToWorld,
            nativePipelineInfo.objectTransformSource)) {
        g_placedMeshes.emplace_back(g_currentVertexBuffer, indexCount);
        nativePipelineInfo.hasObjectToWorld = true;
        nativePipelineInfoPtr = &nativePipelineInfo;
      }
    }
    // The structured buffer the model shaders read their transform from is dumped
    // once, so its layout can be compared with the block the object matrix was
    // taken from.
    {
      static std::atomic<uint32_t> srtDumps { 0 };
      const uint64_t srtAddress = g_aczSrtBufferAddress.load(std::memory_order_relaxed);
      if (srtAddress && srtDumps.fetch_add(1, std::memory_order_relaxed) < 2) {
        std::vector<uint8_t> block;
        if (readGpuBuffer(srtAddress, 0x1A0, block) && block.size() >= 0x1A0) {
          for (uint32_t row = 16; row < 25; ++row) {
            const auto* v = reinterpret_cast<const float*>(block.data() + row * 16);
          }
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
    // Model draws report a placement whose translation is not in world space:
    // it stays put while the camera crosses the map, so it is encoded relative
    // to the camera. Three conventions cover how that can be written, and the
    // draw's own vertices decide between them because only the right one puts
    // the geometry inside the frustum.
    D3D12NativePipelineInfo cameraFrameInfo;
    if (nativePipelineInfoPtr
     && nativePipelineInfoPtr->hasObjectToWorld
     && indexCount >= 20
     && (nativePipelineInfoPtr->objectTransformSource == 2
      || nativePipelineInfoPtr->objectTransformSource == 4)
     && placesInCameraRelativeFrame(nativePipelineInfoPtr->vertexShaderHash)) {
      AczCameraMatrices camera {};
      if (buildAczCamera(camera)) {
        const auto& stored = nativePipelineInfoPtr->objectToWorld;
        const uint64_t shaderHash = nativePipelineInfoPtr->vertexShaderHash;
        const uint32_t source = nativePipelineInfoPtr->objectTransformSource;

        // The first two vertices of the draw are what the candidates are tested
        // against, read through the pipeline's own position element because the
        // models name their attributes ATTRIBUTE0 and up.
        float vertices[6] = {};
        uint32_t vertexCount = 0;
        uint32_t positionOffset = 0;
        bool hasPosition = false;
        for (const auto& element : nativePipelineInfoPtr->inputElements) {
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
          if (!hasPosition || element.offset < positionOffset) {
            positionOffset = element.offset;
            hasPosition = true;
          }
        }
        if (hasPosition && vertexBuffer.StrideInBytes > positionOffset + sizeof(float) * 3) {
          std::vector<uint8_t> bytes;
          if (readGpuBuffer(vertexBuffer.BufferLocation,
                static_cast<size_t>(vertexBuffer.StrideInBytes) * 2, bytes)) {
            const uint32_t available = static_cast<uint32_t>(
              bytes.size() / vertexBuffer.StrideInBytes);
            vertexCount = std::min<uint32_t>(available, 2u);
            for (uint32_t i = 0; i < vertexCount; ++i) {
              const auto* position = reinterpret_cast<const float*>(
                bytes.data() + static_cast<size_t>(i) * vertexBuffer.StrideInBytes
                + positionOffset);
              vertices[i * 3 + 0] = position[0];
              vertices[i * 3 + 1] = position[1];
              vertices[i * 3 + 2] = position[2];
            }
          }
        }

        const auto inFrustum = [&](const std::array<float, 16>& matrix) {
          if (vertexCount == 0)
            return false;
          float centroidX = 0.0f;
          float centroidY = 0.0f;
          for (uint32_t i = 0; i < vertexCount; ++i) {
            const float* v = &vertices[i * 3];
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
          centroidX /= static_cast<float>(vertexCount);
          centroidY /= static_cast<float>(vertexCount);
          return std::abs(centroidX) <= 2.0f && std::abs(centroidY) <= 2.0f;
        };
        // A: the placement as stored. B: the translation lifted into world
        // space. C: the translation read in the placement's own basis, then
        // lifted.
        std::array<float, 16> placements[3] = { stored, stored, stored };
        placements[1][12] += camera.eye[0];
        placements[1][13] += camera.eye[1];
        placements[1][14] += camera.eye[2];
        const float basisX = stored[0] * stored[12] + stored[4] * stored[13] + stored[8] * stored[14];
        const float basisY = stored[1] * stored[12] + stored[5] * stored[13] + stored[9] * stored[14];
        const float basisZ = stored[2] * stored[12] + stored[6] * stored[13] + stored[10] * stored[14];
        placements[2][12] = basisX + camera.eye[0];
        placements[2][13] = basisY + camera.eye[1];
        placements[2][14] = basisZ + camera.eye[2];

        static const char* const kConventionNames[3] = { "stored", "lifted", "basis" };
        uint32_t chosen = 3;
        for (uint32_t i = 0; i < 3; ++i) {
          if (inFrustum(placements[i])) {
            chosen = i;
            break;
          }
        }
        if (chosen < 3) {
          cameraFrameInfo = *nativePipelineInfoPtr;
          cameraFrameInfo.objectToWorld = placements[chosen];
          nativePipelineInfoPtr = &cameraFrameInfo;
        }

        static std::atomic<uint32_t> frameLogs { 0 };
        const uint32_t frameLog = frameLogs.fetch_add(1, std::memory_order_relaxed);

      }
    }
    // Only the draws this task can act on are reported: one that sits next to the
    // camera, which is where the player aircraft belongs, and one whose transform
    // collapsed to the origin while it is large enough to be a hull. Both are
    // reported on every frame, so a moving aircraft is never missed.
    if (indexCount >= 200) {
      const auto& meshMatrix = nativePipelineInfoPtr->objectToWorld;
      const bool hasTransform = nativePipelineInfoPtr->hasObjectToWorld;
      AczCameraMatrices meshCamera {};
      float eyeDistance = 0.0f;
      const bool haveCamera = buildAczCamera(meshCamera);
      if (haveCamera) {
        const float dx = meshMatrix[12] - meshCamera.eye[0];
        const float dy = meshMatrix[13] - meshCamera.eye[1];
        const float dz = meshMatrix[14] - meshCamera.eye[2];
        eyeDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
      }
      const bool nearCamera = haveCamera && eyeDistance < 200.0f;
      const bool collapsedAtOrigin = hasTransform && indexCount >= 500
        && std::abs(meshMatrix[12]) < 1.0e-3f
        && std::abs(meshMatrix[13]) < 1.0e-3f
        && std::abs(meshMatrix[14]) < 1.0e-3f;
      // Once the player aircraft is placed in the camera frame every one of its
      // draws lands within the near window, so the report is keyed by what the draw
      // actually is rather than repeated per frame.
      if (nearCamera || collapsedAtOrigin) {
        static std::unordered_set<uint64_t> reportedMeshes;
        const uint64_t meshKey = (static_cast<uint64_t>(indexCount) << 8)
          | (static_cast<uint64_t>(nativePipelineInfo.objectTransformSource) << 1)
          | (nearCamera ? 1ull : 0ull);
        if (reportedMeshes.size() >= 256 || reportedMeshes.insert(meshKey).second) {
          log("ACZ mesh: vs=0x%016llx ps=0x%016llx idx=%u t=(%.1f,%.1f,%.1f) hasT=%d eyeDist=%.1f",
            static_cast<unsigned long long>(nativePipelineInfo.vertexShaderHash),
            static_cast<unsigned long long>(nativePipelineInfo.pixelShaderHash),
            indexCount,
            static_cast<double>(meshMatrix[12]), static_cast<double>(meshMatrix[13]),
            static_cast<double>(meshMatrix[14]),
            hasTransform ? 1 : 0, static_cast<double>(eyeDistance));
        }
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
    // The position this frame renders is the engine target the previous frame ended with.
    // Advancing it here keeps every submission of a frame on one value and still moves the
    // aircraft forward once per frame, including while the target itself stands still.
    if (g_targetSeenValid) {
      for (uint32_t index = 0; index < 3; ++index)
        g_cachedTarget[index] = g_targetSeen[index];
      g_cachedTargetValid = true;
    }

    g_placedMeshes.clear();
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


  void recordD3D12Present(void* swapChain, UINT syncInterval, UINT flags, HRESULT result) {
    initializeD3D12RuntimeAdapter();
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
          }
        } else {
          static uint32_t missingBlockLogs = 0;
          if (missingBlockLogs < 2) {
            ++missingBlockLogs;
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
          for (uint32_t row = 0; row < 8; ++row) {
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
