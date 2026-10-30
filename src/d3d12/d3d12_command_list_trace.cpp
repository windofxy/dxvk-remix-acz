#include "d3d12_command_list_trace.h"

#include "d3d12_geometry_bridge.h"
#include "d3d12_device_proxy.h"
#include "d3d12_resource_registry.h"
#include "d3d12_runtime_adapter.h"
#include "d3d12_native_backend.h"

#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <mutex>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace remix {

  // Root constant-buffer views bound for the most recent indexed draw.
  std::array<D3D12_GPU_VIRTUAL_ADDRESS, 16> g_currentRootCbvs {};

  // Cross-check counters: if the game binds vertex buffers far more often than
  // it issues draws through this interface, some draws bypass the trace.
  std::atomic<uint32_t> g_vertexBufferSets { 0 };
  std::atomic<uint32_t> g_indexedDraws { 0 };
  std::atomic<uint32_t> g_instancedDraws { 0 };
  // Module-relative return address of the most recent indexed draw.
  std::atomic<uint64_t> g_lastDrawCallerRva { 0 };

  D3D12_GPU_VIRTUAL_ADDRESS currentGraphicsRootCbv(uint32_t index) {
    return index < g_currentRootCbvs.size() ? g_currentRootCbvs[index] : 0;
  }

  uint64_t currentDrawCallerRva() {
    return g_lastDrawCallerRva.load(std::memory_order_relaxed);
  }

  namespace {

    // ID3D12GraphicsCommandList7 adds Barrier after the ID3D12GraphicsCommandList6
    // mesh-shader methods. Keep that slot when cloning newer interface pointers.
    constexpr size_t kGraphicsCommandListMethodCount = 81;
    constexpr size_t kQueryInterfaceIndex = 0;
    constexpr size_t kClearStateIndex = 11;
    constexpr size_t kCloseIndex = 9;
    constexpr size_t kResetIndex = 10;
    constexpr size_t kCopyBufferRegionIndex = 15;
    constexpr size_t kCopyTextureRegionIndex = 16;
    constexpr size_t kCopyResourceIndex = 17;
    constexpr size_t kResolveSubresourceIndex = 19;
    constexpr size_t kIASetPrimitiveTopologyIndex = 20;
    constexpr size_t kRSSetViewportsIndex = 21;
    constexpr size_t kRSSetScissorRectsIndex = 22;
    constexpr size_t kOMSetStencilRefIndex = 24;
    constexpr size_t kExecuteBundleIndex = 27;
    constexpr size_t kSetDescriptorHeapsIndex = 28;
    constexpr size_t kSetComputeRootSignatureIndex = 29;
    constexpr size_t kSetGraphicsRootSignatureIndex = 30;
    constexpr size_t kSetComputeRootDescriptorTableIndex = 31;
    constexpr size_t kSetGraphicsRootDescriptorTableIndex = 32;
    constexpr size_t kSetComputeRoot32BitConstantsIndex = 34;
    constexpr size_t kSetGraphicsRoot32BitConstantsIndex = 39;
    constexpr size_t kSetGraphicsRootConstantBufferViewIndex = 40;
    constexpr size_t kSetGraphicsRootShaderResourceViewIndex = 41;
    constexpr size_t kIASetIndexBufferIndex = 43;
    constexpr size_t kIASetVertexBuffersIndex = 44;
    constexpr size_t kOMSetRenderTargetsIndex = 46;
    constexpr size_t kClearDepthStencilViewIndex = 47;
    constexpr size_t kClearRenderTargetViewIndex = 48;
    constexpr size_t kDrawInstancedIndex = 12;
    constexpr size_t kDrawIndexedInstancedIndex = 13;
    constexpr size_t kDispatchIndex = 14;
    constexpr size_t kSetPipelineStateIndex = 25;
    constexpr size_t kResourceBarrierIndex = 26;
    constexpr size_t kExecuteIndirectIndex = 59;
    constexpr size_t kBeginRenderPassIndex = 68;
    constexpr size_t kEndRenderPassIndex = 69;
    constexpr size_t kInitializeMetaCommandIndex = 70;
    constexpr size_t kExecuteMetaCommandIndex = 71;
    constexpr size_t kBuildRaytracingAccelerationStructureIndex = 72;
    constexpr size_t kEmitRaytracingAccelerationStructurePostbuildInfoIndex = 73;
    constexpr size_t kCopyRaytracingAccelerationStructureIndex = 74;
    constexpr size_t kSetPipelineState1Index = 75;
    constexpr size_t kDispatchRaysIndex = 76;
    constexpr size_t kRSSetShadingRateIndex = 77;
    constexpr size_t kRSSetShadingRateImageIndex = 78;
    constexpr size_t kDispatchMeshIndex = 79;
    constexpr size_t kBarrierIndex = 80;

    constexpr IID kGraphicsCommandList7Iid = {
      0xB0B9E70E, 0x59CE, 0x4415,
      { 0x81, 0x16, 0xC8, 0xC8, 0x26, 0xB1, 0xE5, 0x99 }
    };

    struct TraceEntry {
      void** vtable;
      void* queryInterface;
      void* clearState;
      void* close;
      void* reset;
      void* copyBufferRegion;
      void* copyTextureRegion;
      void* copyResource;
      void* resolveSubresource;
      void* iaSetPrimitiveTopology;
      void* rsSetViewports;
      void* rsSetScissorRects;
      void* omSetStencilRef;
      void* executeBundle;
      void* setDescriptorHeaps;
      void* setComputeRootSignature;
      void* setGraphicsRootSignature;
      void* setComputeRootDescriptorTable;
      void* setGraphicsRootDescriptorTable;
      void* setComputeRoot32BitConstants;
      void* setGraphicsRoot32BitConstants;
      void* setGraphicsRootConstantBufferView;
      void* setGraphicsRootShaderResourceView;
      void* iaSetIndexBuffer;
      void* iaSetVertexBuffers;
      void* omSetRenderTargets;
      void* clearDepthStencilView;
      void* clearRenderTargetView;
      void* drawInstanced;
      void* drawIndexedInstanced;
      void* dispatch;
      void* setPipelineState;
      void* resourceBarrier;
      void* executeIndirect;
      void* beginRenderPass;
      void* endRenderPass;
      void* initializeMetaCommand;
      void* executeMetaCommand;
      void* buildRaytracingAccelerationStructure;
      void* emitRaytracingAccelerationStructurePostbuildInfo;
      void* copyRaytracingAccelerationStructure;
      void* setPipelineState1;
      void* dispatchRays;
      void* rsSetShadingRate;
      void* rsSetShadingRateImage;
      void* dispatchMesh;
      void* barrier;
      ID3D12PipelineState* currentPipelineState;
      D3D12_VERTEX_BUFFER_VIEW currentVertexBuffer;
      D3D12_INDEX_BUFFER_VIEW currentIndexBuffer;
      D3D12_PRIMITIVE_TOPOLOGY currentTopology;
      D3D12_GPU_DESCRIPTOR_HANDLE currentGraphicsDescriptorTables[16];
      D3D12_GPU_VIRTUAL_ADDRESS currentGraphicsRootCbvs[16];
      D3D12_GPU_VIRTUAL_ADDRESS currentGraphicsRootSrvs[16];
      uint32_t currentRootConstantCount[8];
      uint32_t currentRootConstants[8][16];
    };

    std::mutex g_traceMutex;
    std::unordered_map<ID3D12GraphicsCommandList*, TraceEntry> g_traceEntries;
    std::unordered_set<IUnknown*> g_interfaceProbes;
    std::unordered_set<uint64_t> g_loggedRootCbvData;


    void logRootCbvData(UINT rootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS bufferLocation) {
      // Root CBVs carry per-frame and per-object constants. Capture a
      // bounded set of aligned addresses so the camera block can be identified
      // without producing an unbounded trace file.
      if (rootParameterIndex > 7 || !bufferLocation)
        return;

      {
        std::lock_guard lock(g_traceMutex);
        if (g_loggedRootCbvData.size() >= 256 || !g_loggedRootCbvData.emplace(bufferLocation).second)
          return;
      }

      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(bufferLocation, 65536, bytes) || bytes.size() < 65536) {
        log("D3D12 root CBV data unavailable: index=%u gpu=0x%llx",
          rootParameterIndex, static_cast<unsigned long long>(bufferLocation));
        return;
      }

      const auto* values = reinterpret_cast<const float*>(bytes.data());
      log("D3D12 root CBV data: index=%u gpu=0x%llx",
        rootParameterIndex, static_cast<unsigned long long>(bufferLocation));
      for (unsigned row = 0; row < 64; ++row) {
        log("D3D12 root CBV data row%u: %.9g %.9g %.9g %.9g",
          row,
          static_cast<double>(values[row * 4 + 0]),
          static_cast<double>(values[row * 4 + 1]),
          static_cast<double>(values[row * 4 + 2]),
          static_cast<double>(values[row * 4 + 3]));
      }

      // Search the larger upload slice for affine/projection matrix blocks.
      // Root CBVs mix flags, colors, and object data, so matrix-shaped
      // blocks are more useful than assuming a fixed constant-buffer layout.
      for (size_t offset = 0; offset + 16 * sizeof(float) <= bytes.size(); offset += 16) {
        const float* matrix = reinterpret_cast<const float*>(bytes.data() + offset);
        bool finite = true;
        for (unsigned i = 0; i < 16; ++i)
          finite = finite && std::isfinite(matrix[i]) && std::abs(matrix[i]) < 1.0e8f;
        if (!finite || std::abs(matrix[15] - 1.0f) > 0.01f)
          continue;

        const float rowNorm0 = std::sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2]);
        const float rowNorm1 = std::sqrt(matrix[4] * matrix[4] + matrix[5] * matrix[5] + matrix[6] * matrix[6]);
        const float rowNorm2 = std::sqrt(matrix[8] * matrix[8] + matrix[9] * matrix[9] + matrix[10] * matrix[10]);
        if (rowNorm0 < 0.1f || rowNorm0 > 10.0f
         || rowNorm1 < 0.1f || rowNorm1 > 10.0f
         || rowNorm2 < 0.1f || rowNorm2 > 10.0f)
          continue;

        log("D3D12 root CBV matrix candidate: index=%u gpu=0x%llx offset=0x%zx rows=%.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g",
          rootParameterIndex, static_cast<unsigned long long>(bufferLocation), offset,
          static_cast<double>(matrix[0]), static_cast<double>(matrix[1]), static_cast<double>(matrix[2]), static_cast<double>(matrix[3]),
          static_cast<double>(matrix[4]), static_cast<double>(matrix[5]), static_cast<double>(matrix[6]), static_cast<double>(matrix[7]),
          static_cast<double>(matrix[8]), static_cast<double>(matrix[9]), static_cast<double>(matrix[10]), static_cast<double>(matrix[11]),
          static_cast<double>(matrix[12]), static_cast<double>(matrix[13]), static_cast<double>(matrix[14]), static_cast<double>(matrix[15]));
      }
    }

    void logRootSrvData(UINT rootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS bufferLocation) {
      if (rootParameterIndex > 7 || !bufferLocation || (bufferLocation & 0xFFFFFu) > 0x40u)
        return;

      {
        std::lock_guard lock(g_traceMutex);
        if (g_loggedRootCbvData.size() >= 64 || !g_loggedRootCbvData.emplace(bufferLocation).second)
          return;
      }

      std::vector<uint8_t> bytes;
      if (!readGpuBuffer(bufferLocation, 4096, bytes) || bytes.size() < 4096) {
        log("D3D12 root SRV data unavailable: index=%u gpu=0x%llx",
          rootParameterIndex, static_cast<unsigned long long>(bufferLocation));
        return;
      }

      const auto* values = reinterpret_cast<const float*>(bytes.data());
      log("D3D12 root SRV data: index=%u gpu=0x%llx",
        rootParameterIndex, static_cast<unsigned long long>(bufferLocation));
      for (unsigned row = 0; row < 64; ++row) {
        log("D3D12 root SRV data row%u: %.9g %.9g %.9g %.9g",
          row,
          static_cast<double>(values[row * 4 + 0]),
          static_cast<double>(values[row * 4 + 1]),
          static_cast<double>(values[row * 4 + 2]),
          static_cast<double>(values[row * 4 + 3]));
      }
    }

    TraceEntry* findEntry(ID3D12GraphicsCommandList* commandList) {
      std::lock_guard lock(g_traceMutex);
      auto entry = g_traceEntries.find(commandList);
      return entry == g_traceEntries.end() ? nullptr : &entry->second;
    }

    // QueryInterface and some driver paths can expose the native vtable again
    // after the first trace installation. Reapply the existing per-object clone
    // immediately before submission without probing every graphics interface.
    bool reapplyExistingTrace(IUnknown* object) {
      if (!object)
        return false;

      auto* commandList = reinterpret_cast<ID3D12GraphicsCommandList*>(object);
      void** clone = nullptr;
      {
        std::lock_guard lock(g_traceMutex);
        auto entry = g_traceEntries.find(commandList);
        if (entry == g_traceEntries.end())
          return false;
        clone = entry->second.vtable;
      }

      if (*reinterpret_cast<void***>(commandList) == clone)
        return true;

      DWORD protection = 0;
      if (!VirtualProtect(reinterpret_cast<void*>(commandList), sizeof(void*), PAGE_READWRITE, &protection)) {
        log("Command list trace reinstallation failed: VirtualProtect error=%lu", GetLastError());
        return false;
      }

      *reinterpret_cast<void***>(commandList) = clone;
      DWORD ignored = 0;
      VirtualProtect(reinterpret_cast<void*>(commandList), sizeof(void*), protection, &ignored);
      log("Reinstalled command list trace at %p", commandList);
      return true;
    }

    USHORT captureTraceFrames(void** frames, USHORT count) {
      return CaptureStackBackTrace(1, count, frames, nullptr);
    }

  }

  bool isGraphicsCommandListIid(REFIID riid) {
    return riid == __uuidof(ID3D12GraphicsCommandList)
      || riid == __uuidof(ID3D12GraphicsCommandList1)
      || riid == __uuidof(ID3D12GraphicsCommandList2)
      || riid == __uuidof(ID3D12GraphicsCommandList3)
      || riid == __uuidof(ID3D12GraphicsCommandList4)
      || riid == __uuidof(ID3D12GraphicsCommandList5)
      || riid == __uuidof(ID3D12GraphicsCommandList6)
      || riid == kGraphicsCommandList7Iid;
  }

  void installCommandListInterfaceTraces(IUnknown* commandList) {
    if (!commandList)
      return;

    bool firstProbe = false;
    {
      std::lock_guard lock(g_traceMutex);
      firstProbe = g_interfaceProbes.emplace(commandList).second;
    }

    if (!firstProbe) {
      reapplyExistingTrace(commandList);
      return;
    }

    // The game records through a newer graphics-command-list interface while the
    // queue exposes the object as ID3D12CommandList. Install the same prefix
    // hook on every supported graphics interface pointer before submission.
    const IID interfaces[] = {
      __uuidof(ID3D12GraphicsCommandList),
      __uuidof(ID3D12GraphicsCommandList1),
      __uuidof(ID3D12GraphicsCommandList2),
      __uuidof(ID3D12GraphicsCommandList3),
      __uuidof(ID3D12GraphicsCommandList4),
      __uuidof(ID3D12GraphicsCommandList5),
      __uuidof(ID3D12GraphicsCommandList6),
      kGraphicsCommandList7Iid,
    };

    for (const IID& iid : interfaces) {
      void* object = nullptr;
      HRESULT result = commandList->QueryInterface(iid, &object);
      bool installed = false;
      if (SUCCEEDED(result) && object) {
        installed = installCommandListTrace(
          reinterpret_cast<ID3D12GraphicsCommandList*>(object));
        reinterpret_cast<IUnknown*>(object)->Release();
      }
      log("Command list interface probe(riid=%s, result=0x%08lx, object=%p, traceInstalled=%u)",
        formatIid(iid), static_cast<unsigned long>(result), object,
        installed ? 1u : 0u);
    }
  }

  namespace {

    HRESULT STDMETHODCALLTYPE traceQueryInterface(ID3D12GraphicsCommandList* commandList, REFIID riid, void** object) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, REFIID, void**)>(entry->queryInterface);
      HRESULT result = function(commandList, riid, object);
      bool supported = isGraphicsCommandListIid(riid);
      bool installed = false;
      if (SUCCEEDED(result) && supported && object && *object)
        installed = installCommandListTrace(static_cast<ID3D12GraphicsCommandList*>(*object));
      log("ID3D12GraphicsCommandList::QueryInterface(riid=%s, result=0x%08lx, object=%p, traceSupported=%u, traceInstalled=%u)",
        formatIid(riid), static_cast<unsigned long>(result),
        object ? *object : nullptr, supported ? 1u : 0u, installed ? 1u : 0u);
      return result;
    }

    void STDMETHODCALLTYPE traceClearState(
        ID3D12GraphicsCommandList* commandList,
        ID3D12PipelineState* pipelineState) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12PipelineState*)>(entry->clearState);
      entry->currentPipelineState = pipelineState;
      entry->currentVertexBuffer = {};
      entry->currentIndexBuffer = {};
      entry->currentTopology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
      std::memset(entry->currentGraphicsDescriptorTables, 0, sizeof(entry->currentGraphicsDescriptorTables));
      std::memset(entry->currentGraphicsRootCbvs, 0, sizeof(entry->currentGraphicsRootCbvs));
      std::memset(entry->currentGraphicsRootSrvs, 0, sizeof(entry->currentGraphicsRootSrvs));
      log("ID3D12GraphicsCommandList::ClearState(list=%p, pipelineState=%p)",
        commandList, pipelineState);
      function(commandList, pipelineState);
    }

    HRESULT STDMETHODCALLTYPE traceClose(ID3D12GraphicsCommandList* commandList) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*)>(entry->close);
      HRESULT result = function(commandList);
      log("ID3D12GraphicsCommandList::Close(result=0x%08lx)", static_cast<unsigned long>(result));
      return result;
    }

    HRESULT STDMETHODCALLTYPE traceReset(ID3D12GraphicsCommandList* commandList, ID3D12CommandAllocator* allocator, ID3D12PipelineState* initialState) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*, ID3D12PipelineState*)>(entry->reset);
      HRESULT result = function(commandList, allocator, initialState);
      if (SUCCEEDED(result)) {
        entry->currentPipelineState = initialState;
        entry->currentVertexBuffer = {};
        entry->currentIndexBuffer = {};
        entry->currentTopology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
        std::memset(entry->currentGraphicsDescriptorTables, 0, sizeof(entry->currentGraphicsDescriptorTables));
        std::memset(entry->currentGraphicsRootCbvs, 0, sizeof(entry->currentGraphicsRootCbvs));
        std::memset(entry->currentGraphicsRootSrvs, 0, sizeof(entry->currentGraphicsRootSrvs));
        // Reset can swap the native command-list object to its recording-state
        // vtable. Restore the trace clone before the game records any draw calls.
        reapplyExistingTrace(reinterpret_cast<IUnknown*>(commandList));
      }
      log("ID3D12GraphicsCommandList::Reset(list=%p, allocator=%p, initialState=%p, result=0x%08lx)",
        commandList, allocator, initialState, static_cast<unsigned long>(result));
      return result;
    }

    void STDMETHODCALLTYPE traceCopyBufferRegion(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* destination,
        UINT64 destinationOffset,
        ID3D12Resource* source,
        UINT64 sourceOffset,
        UINT64 numBytes) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12Resource*, UINT64,
        ID3D12Resource*, UINT64, UINT64)>(entry->copyBufferRegion);
      log("ID3D12GraphicsCommandList::CopyBufferRegion(bytes=%llu)",
        static_cast<unsigned long long>(numBytes));
      recordCopyBufferRegion(destination, destinationOffset, source, sourceOffset, numBytes);
      function(commandList, destination, destinationOffset, source, sourceOffset, numBytes);
    }

    void STDMETHODCALLTYPE traceCopyTextureRegion(
        ID3D12GraphicsCommandList* commandList,
        const D3D12_TEXTURE_COPY_LOCATION* destination,
        UINT destinationX,
        UINT destinationY,
        UINT destinationZ,
        const D3D12_TEXTURE_COPY_LOCATION* source,
        const D3D12_BOX* sourceBox) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, const D3D12_TEXTURE_COPY_LOCATION*, UINT,
        UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION*, const D3D12_BOX*)>(
          entry->copyTextureRegion);
      log("ID3D12GraphicsCommandList::CopyTextureRegion(dst=%p, src=%p)",
        destination, source);
      recordD3D12CopyTextureRegion(destination, source, destinationX, destinationY, destinationZ, sourceBox);
      function(commandList, destination, destinationX, destinationY, destinationZ, source, sourceBox);
    }

    void STDMETHODCALLTYPE traceCopyResource(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* destination,
        ID3D12Resource* source) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12Resource*, ID3D12Resource*)>(entry->copyResource);
      log("ID3D12GraphicsCommandList::CopyResource(dst=%p, src=%p)", destination, source);
      if (destination && source
       && destination->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
       && source->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        recordCopyBufferRegion(destination, 0, source, 0,
          std::min(destination->GetDesc().Width, source->GetDesc().Width));
      }
      function(commandList, destination, source);
    }

    void STDMETHODCALLTYPE traceResolveSubresource(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* destination,
        UINT destinationSubresource,
        ID3D12Resource* source,
        UINT sourceSubresource,
        DXGI_FORMAT format) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12Resource*, UINT,
        ID3D12Resource*, UINT, DXGI_FORMAT)>(entry->resolveSubresource);
      log("ID3D12GraphicsCommandList::ResolveSubresource(dst=%p, src=%p, format=%u)",
        destination, source, static_cast<unsigned>(format));
      function(commandList, destination, destinationSubresource, source, sourceSubresource, format);
    }

    void STDMETHODCALLTYPE traceIASetPrimitiveTopology(
        ID3D12GraphicsCommandList* commandList,
        D3D12_PRIMITIVE_TOPOLOGY topology) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, D3D12_PRIMITIVE_TOPOLOGY)>(entry->iaSetPrimitiveTopology);
      entry->currentTopology = topology;
      log("ID3D12GraphicsCommandList::IASetPrimitiveTopology(topology=%u)",
        static_cast<unsigned>(topology));
      function(commandList, topology);
    }

    void STDMETHODCALLTYPE traceRSSetViewports(
        ID3D12GraphicsCommandList* commandList,
        UINT count,
        const D3D12_VIEWPORT* viewports) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, const D3D12_VIEWPORT*)>(
          entry->rsSetViewports);
      if (count && viewports) {
        log("ID3D12GraphicsCommandList::RSSetViewports(count=%u, first=%gx%g at %g,%g)",
          count, viewports[0].Width, viewports[0].Height,
          viewports[0].TopLeftX, viewports[0].TopLeftY);
      } else {
        log("ID3D12GraphicsCommandList::RSSetViewports(count=%u)", count);
      }
      function(commandList, count, viewports);
    }

    void STDMETHODCALLTYPE traceRSSetScissorRects(
        ID3D12GraphicsCommandList* commandList,
        UINT count,
        const D3D12_RECT* rects) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, const D3D12_RECT*)>(
          entry->rsSetScissorRects);
      if (count && rects) {
        log("ID3D12GraphicsCommandList::RSSetScissorRects(count=%u, first=%ld,%ld-%ld,%ld)",
          count, rects[0].left, rects[0].top, rects[0].right, rects[0].bottom);
      } else {
        log("ID3D12GraphicsCommandList::RSSetScissorRects(count=%u)", count);
      }
      function(commandList, count, rects);
    }

    void STDMETHODCALLTYPE traceOMSetStencilRef(
        ID3D12GraphicsCommandList* commandList,
        UINT stencilRef) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT)>(entry->omSetStencilRef);
      log("ID3D12GraphicsCommandList::OMSetStencilRef(value=%u)", stencilRef);
      function(commandList, stencilRef);
    }

    void STDMETHODCALLTYPE traceExecuteBundle(
        ID3D12GraphicsCommandList* commandList,
        ID3D12GraphicsCommandList* bundle) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList*)>(entry->executeBundle);
      log("ID3D12GraphicsCommandList::ExecuteBundle(bundle=%p)", bundle);
      function(commandList, bundle);
    }

    void STDMETHODCALLTYPE traceSetDescriptorHeaps(
        ID3D12GraphicsCommandList* commandList,
        UINT count,
        ID3D12DescriptorHeap* const* heaps) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, ID3D12DescriptorHeap* const*)>(entry->setDescriptorHeaps);
      log("ID3D12GraphicsCommandList::SetDescriptorHeaps(count=%u)", count);
      function(commandList, count, heaps);
    }

    void STDMETHODCALLTYPE traceSetComputeRootSignature(
        ID3D12GraphicsCommandList* commandList,
        ID3D12RootSignature* rootSignature) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12RootSignature*)>(entry->setComputeRootSignature);
      log("ID3D12GraphicsCommandList::SetComputeRootSignature(root=%p)", rootSignature);
      function(commandList, rootSignature);
    }

    void STDMETHODCALLTYPE traceSetGraphicsRootSignature(
        ID3D12GraphicsCommandList* commandList,
        ID3D12RootSignature* rootSignature) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12RootSignature*)>(entry->setGraphicsRootSignature);
      log("ID3D12GraphicsCommandList::SetGraphicsRootSignature(root=%p)", rootSignature);
      function(commandList, rootSignature);
    }

    void STDMETHODCALLTYPE traceSetComputeRootDescriptorTable(
        ID3D12GraphicsCommandList* commandList,
        UINT rootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE baseDescriptor) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE)>(entry->setComputeRootDescriptorTable);
      log("ID3D12GraphicsCommandList::SetComputeRootDescriptorTable(index=%u, gpu=0x%llx)",
        rootParameterIndex, static_cast<unsigned long long>(baseDescriptor.ptr));
      function(commandList, rootParameterIndex, baseDescriptor);
    }

    void STDMETHODCALLTYPE traceSetGraphicsRootDescriptorTable(
        ID3D12GraphicsCommandList* commandList,
        UINT rootParameterIndex,
        D3D12_GPU_DESCRIPTOR_HANDLE baseDescriptor) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE)>(entry->setGraphicsRootDescriptorTable);
      if (rootParameterIndex < ARRAYSIZE(entry->currentGraphicsDescriptorTables))
        entry->currentGraphicsDescriptorTables[rootParameterIndex] = baseDescriptor;
      log("ID3D12GraphicsCommandList::SetGraphicsRootDescriptorTable(index=%u, gpu=0x%llx)",
        rootParameterIndex, static_cast<unsigned long long>(baseDescriptor.ptr));
      function(commandList, rootParameterIndex, baseDescriptor);
    }

    void STDMETHODCALLTYPE traceSetComputeRoot32BitConstants(
        ID3D12GraphicsCommandList* commandList,
        UINT rootParameterIndex,
        UINT num32BitValuesToSet,
        const void* pSrcData,
        UINT destOffsetIn32BitValues) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, UINT, const void*, UINT)>(entry->setComputeRoot32BitConstants);
      log("ID3D12GraphicsCommandList::SetComputeRoot32BitConstants(index=%u, values=%u, dest=%u)",
        rootParameterIndex, num32BitValuesToSet, destOffsetIn32BitValues);
      function(commandList, rootParameterIndex, num32BitValuesToSet, pSrcData, destOffsetIn32BitValues);
    }

    void STDMETHODCALLTYPE traceSetGraphicsRoot32BitConstants(
        ID3D12GraphicsCommandList* commandList,
        UINT rootParameterIndex,
        UINT num32BitValuesToSet,
        const void* pSrcData,
        UINT destOffsetIn32BitValues) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, UINT, const void*, UINT)>(entry->setGraphicsRoot32BitConstants);
      log("ID3D12GraphicsCommandList::SetGraphicsRoot32BitConstants(index=%u, values=%u, dest=%u)",
        rootParameterIndex, num32BitValuesToSet, destOffsetIn32BitValues);
      if (rootParameterIndex < ARRAYSIZE(entry->currentRootConstants) && pSrcData) {
        const UINT copy = std::min<UINT>(num32BitValuesToSet, 16);
        std::memcpy(entry->currentRootConstants[rootParameterIndex],
          pSrcData, copy * sizeof(uint32_t));
        entry->currentRootConstantCount[rootParameterIndex] = copy;
      }
      function(commandList, rootParameterIndex, num32BitValuesToSet, pSrcData, destOffsetIn32BitValues);
    }

    void STDMETHODCALLTYPE traceSetGraphicsRootConstantBufferView(
        ID3D12GraphicsCommandList* commandList,
        UINT rootParameterIndex,
        D3D12_GPU_VIRTUAL_ADDRESS bufferLocation) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, D3D12_GPU_VIRTUAL_ADDRESS)>(entry->setGraphicsRootConstantBufferView);
      if (rootParameterIndex < ARRAYSIZE(entry->currentGraphicsRootCbvs))
        entry->currentGraphicsRootCbvs[rootParameterIndex] = bufferLocation;
      log("ID3D12GraphicsCommandList::SetGraphicsRootConstantBufferView(index=%u, gpu=0x%llx)",
        rootParameterIndex, static_cast<unsigned long long>(bufferLocation));
      logRootCbvData(rootParameterIndex, bufferLocation);
      function(commandList, rootParameterIndex, bufferLocation);
    }

    void STDMETHODCALLTYPE traceSetGraphicsRootShaderResourceView(
        ID3D12GraphicsCommandList* commandList,
        UINT rootParameterIndex,
        D3D12_GPU_VIRTUAL_ADDRESS bufferLocation) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, D3D12_GPU_VIRTUAL_ADDRESS)>(entry->setGraphicsRootShaderResourceView);
      if (rootParameterIndex < ARRAYSIZE(entry->currentGraphicsRootSrvs))
        entry->currentGraphicsRootSrvs[rootParameterIndex] = bufferLocation;
      log("ID3D12GraphicsCommandList::SetGraphicsRootShaderResourceView(index=%u, gpu=0x%llx)",
        rootParameterIndex, static_cast<unsigned long long>(bufferLocation));
      logRootSrvData(rootParameterIndex, bufferLocation);
      function(commandList, rootParameterIndex, bufferLocation);
    }

    void STDMETHODCALLTYPE traceIASetIndexBuffer(
        ID3D12GraphicsCommandList* commandList,
        const D3D12_INDEX_BUFFER_VIEW* view) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, const D3D12_INDEX_BUFFER_VIEW*)>(entry->iaSetIndexBuffer);
      entry->currentIndexBuffer = view ? *view : D3D12_INDEX_BUFFER_VIEW {};
      if (view) {
        log("ID3D12GraphicsCommandList::IASetIndexBuffer(view=%p, location=0x%llx, size=%u, format=%u)",
          view, static_cast<unsigned long long>(view->BufferLocation),
          view->SizeInBytes, static_cast<unsigned>(view->Format));
      } else {
        log("ID3D12GraphicsCommandList::IASetIndexBuffer(view=null)");
      }
      function(commandList, view);
    }

    void STDMETHODCALLTYPE traceIASetVertexBuffers(
        ID3D12GraphicsCommandList* commandList,
        UINT startSlot,
        UINT count,
        const D3D12_VERTEX_BUFFER_VIEW* views) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, UINT,
        const D3D12_VERTEX_BUFFER_VIEW*)>(entry->iaSetVertexBuffers);
      if (views && startSlot == 0 && count)
        entry->currentVertexBuffer = views[0];
      g_vertexBufferSets.fetch_add(1, std::memory_order_relaxed);
      if (views && count) {
        const auto& first = views[0];
        log("ID3D12GraphicsCommandList::IASetVertexBuffers(start=%u, count=%u, firstLocation=0x%llx, firstSize=%u, firstStride=%u)",
          startSlot, count, static_cast<unsigned long long>(first.BufferLocation),
          first.SizeInBytes, first.StrideInBytes);
      } else {
        log("ID3D12GraphicsCommandList::IASetVertexBuffers(start=%u, count=%u)", startSlot, count);
      }
      function(commandList, startSlot, count, views);
    }

    void STDMETHODCALLTYPE traceOMSetRenderTargets(
        ID3D12GraphicsCommandList* commandList,
        UINT count,
        const D3D12_CPU_DESCRIPTOR_HANDLE* renderTargets,
        BOOL singleHandleRange,
        const D3D12_CPU_DESCRIPTOR_HANDLE* depthStencil) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*,
        BOOL, const D3D12_CPU_DESCRIPTOR_HANDLE*)>(entry->omSetRenderTargets);
      log("ID3D12GraphicsCommandList::OMSetRenderTargets(count=%u, depth=%p)",
        count, depthStencil);
      function(commandList, count, renderTargets, singleHandleRange, depthStencil);
    }

    void STDMETHODCALLTYPE traceClearDepthStencilView(
        ID3D12GraphicsCommandList* commandList,
        D3D12_CPU_DESCRIPTOR_HANDLE depthStencilView,
        D3D12_CLEAR_FLAGS clearFlags,
        FLOAT depth,
        UINT8 stencil,
        UINT rectCount,
        const D3D12_RECT* rects) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_CLEAR_FLAGS, FLOAT, UINT8, UINT, const D3D12_RECT*)>(
          entry->clearDepthStencilView);
      log("ID3D12GraphicsCommandList::ClearDepthStencilView(rects=%u)", rectCount);
      function(commandList, depthStencilView, clearFlags, depth, stencil, rectCount, rects);
    }

    void STDMETHODCALLTYPE traceClearRenderTargetView(
        ID3D12GraphicsCommandList* commandList,
        D3D12_CPU_DESCRIPTOR_HANDLE renderTargetView,
        const FLOAT color[4],
        UINT rectCount,
        const D3D12_RECT* rects) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, D3D12_CPU_DESCRIPTOR_HANDLE,
        const FLOAT*, UINT, const D3D12_RECT*)>(entry->clearRenderTargetView);
      log("ID3D12GraphicsCommandList::ClearRenderTargetView(rects=%u)", rectCount);
      function(commandList, renderTargetView, color, rectCount, rects);
    }

    bool dumpFlagPresent() {
      // Rechecked at most once per second so a forgotten flag file cannot flood
      // the log on every draw.
      static std::atomic<uint64_t> lastCheckTick { 0 };
      static std::atomic<bool> present { false };
      const uint64_t now = GetTickCount64();
      if (now - lastCheckTick.load(std::memory_order_relaxed) < 1000)
        return present.load(std::memory_order_relaxed);
      lastCheckTick.store(now, std::memory_order_relaxed);

      wchar_t modulePath[MAX_PATH] = {};
      const DWORD length = GetModuleFileNameW(nullptr, modulePath, ARRAYSIZE(modulePath));
      std::wstring path(modulePath, length);
      const size_t separator = path.find_last_of(L'\\');
      if (separator != std::wstring::npos)
        path.resize(separator + 1);
      path += L"remix-dump-cbv.flag";

      const bool found = GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES;
      present.store(found, std::memory_order_relaxed);
      return found;
    }

    // Diagnostic for locating an engine's camera upload: while the flag file
    // exists, the root state of the first large indexed draws is written to the
    // log, which is where per-frame view constants are bound.
    void dumpDrawRootStateIfRequested(TraceEntry* entry, UINT indexCount) {
      static std::atomic<uint32_t> dumpedDraws { 0 };
      if (!entry || indexCount < 64 || dumpedDraws.load(std::memory_order_relaxed) >= 12)
        return;
      if (!dumpFlagPresent())
        return;
      ++dumpedDraws;

      for (uint32_t slot = 0; slot < 6; ++slot) {
        const D3D12_GPU_VIRTUAL_ADDRESS address = entry->currentGraphicsRootCbvs[slot];
        if (!address)
          continue;

        std::vector<uint8_t> bytes;
        if (!readGpuBuffer(address, 1024, bytes) || bytes.size() < 64 * sizeof(float))
          continue;

        const auto* values = reinterpret_cast<const float*>(bytes.data());
        log("D3D12 draw CBV dump: slot=%u gpu=0x%llx indices=%u", slot,
          static_cast<unsigned long long>(address), indexCount);
        for (uint32_t row = 0; row < 64; ++row) {
          log("D3D12 draw CBV dump row: slot=%u row=%02u %.9g %.9g %.9g %.9g", slot, row,
            static_cast<double>(values[row * 4 + 0]), static_cast<double>(values[row * 4 + 1]),
            static_cast<double>(values[row * 4 + 2]), static_cast<double>(values[row * 4 + 3]));
        }
      }

      for (uint32_t slot = 0; slot < 6; ++slot) {
        const uint32_t count = entry->currentRootConstantCount[slot];
        if (!count)
          continue;

        std::string line;
        char cell[48];
        for (uint32_t i = 0; i < count; ++i) {
          float value = 0.0f;
          std::memcpy(&value, &entry->currentRootConstants[slot][i], sizeof(value));
          std::snprintf(cell, sizeof(cell), "%g ", static_cast<double>(value));
          line += cell;
        }
        log("D3D12 draw root constants: slot=%u count=%u indices=%u values=%s",
          slot, count, indexCount, line.c_str());
      }

      // Descriptor tables are the other place a per-frame camera can live.
      for (uint32_t table = 0; table < 6; ++table) {
        const uint64_t tableGpu = entry->currentGraphicsDescriptorTables[table].ptr;
        if (!tableGpu)
          continue;

        log("D3D12 draw table: slot=%u gpu=0x%llx indices=%u", table,
          static_cast<unsigned long long>(tableGpu), indexCount);
        for (uint32_t descriptor = 0; descriptor < 4; ++descriptor) {
          std::vector<uint8_t> bytes;
          const char* kind = nullptr;
          uint64_t bufferGpu = 0;
          if (!readBoundBufferFromDescriptorTable(
              tableGpu, descriptor, 416, bytes, &kind, &bufferGpu))
            continue;
          if (bytes.size() < 64)
            continue;

          const auto* values = reinterpret_cast<const float*>(bytes.data());
          log("D3D12 draw table buffer: slot=%u descriptor=%u kind=%s gpu=0x%llx",
            table, descriptor, kind ? kind : "unknown",
            static_cast<unsigned long long>(bufferGpu));
          for (uint32_t row = 0; row < 26; ++row) {
            log("D3D12 draw table buffer row: slot=%u descriptor=%u row=%02u %.9g %.9g %.9g %.9g",
              table, descriptor, row,
              static_cast<double>(values[row * 4 + 0]), static_cast<double>(values[row * 4 + 1]),
              static_cast<double>(values[row * 4 + 2]), static_cast<double>(values[row * 4 + 3]));
          }
        }
      }
    }

    void STDMETHODCALLTYPE traceDrawInstanced(ID3D12GraphicsCommandList* commandList, UINT vertexCount, UINT instanceCount, UINT startVertex, UINT startInstance) {
      auto* entry = findEntry(commandList);
      g_instancedDraws.fetch_add(1, std::memory_order_relaxed);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT, UINT)>(entry->drawInstanced);
      void* frames[4] = {};
      USHORT frameCount = captureTraceFrames(frames, ARRAYSIZE(frames));
      log("ID3D12GraphicsCommandList::DrawInstanced(list=%p, vertices=%u, instances=%u, startVertex=%u, startInstance=%u, pipelineState=%p, caller=%p)",
        commandList, vertexCount, instanceCount, startVertex, startInstance,
        entry->currentPipelineState, _ReturnAddress());
      log("ID3D12GraphicsCommandList::DrawInstanced stack(count=%u, frame0=%p, frame1=%p, frame2=%p, frame3=%p)",
        frameCount, frames[0], frames[1], frames[2], frames[3]);
      // Non-indexed draws observed so far are flat screen-space quads used by
      // post-processing and HUD passes, so they are not submitted as world
      // geometry.
      function(commandList, vertexCount, instanceCount, startVertex, startInstance);
    }

    void STDMETHODCALLTYPE traceDrawIndexedInstanced(ID3D12GraphicsCommandList* commandList, UINT indexCount, UINT instanceCount, UINT startIndex, INT baseVertex, UINT startInstance) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT, INT, UINT)>(entry->drawIndexedInstanced);
      void* frames[4] = {};
      USHORT frameCount = captureTraceFrames(frames, ARRAYSIZE(frames));
      {
        const uint32_t drawIndex = g_indexedDraws.fetch_add(1, std::memory_order_relaxed);
        if ((drawIndex % 300) == 0) {
          log("D3D12 draw counters: vbSets=%u indexed=%u instanced=%u",
            g_vertexBufferSets.load(std::memory_order_relaxed),
            drawIndex,
            g_instancedDraws.load(std::memory_order_relaxed));
        }
      }
      log("ID3D12GraphicsCommandList::DrawIndexedInstanced(list=%p, indices=%u, instances=%u, startIndex=%u, baseVertex=%d, startInstance=%u, pipelineState=%p, caller=%p)",
        commandList, indexCount, instanceCount, startIndex, baseVertex, startInstance,
        entry->currentPipelineState, _ReturnAddress());
      log("ID3D12GraphicsCommandList::DrawIndexedInstanced stack(count=%u, frame0=%p, frame1=%p, frame2=%p, frame3=%p)",
        frameCount, frames[0], frames[1], frames[2], frames[3]);
      const auto drawCaller = reinterpret_cast<uintptr_t>(_ReturnAddress());
      const auto moduleBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
      g_lastDrawCallerRva.store(static_cast<uint64_t>(drawCaller - moduleBase),
        std::memory_order_relaxed);
      captureIndexedGeometry(
        commandList,
        &entry->currentVertexBuffer,
        &entry->currentIndexBuffer,
        entry->currentTopology,
        entry->currentPipelineState,
        indexCount,
        instanceCount,
        startIndex,
        baseVertex,
        startInstance);
      dumpDrawRootStateIfRequested(entry, indexCount);
      recordD3D12Draw(
        commandList,
        entry->currentVertexBuffer,
        entry->currentIndexBuffer,
        entry->currentGraphicsDescriptorTables,
        entry->currentTopology,
        entry->currentPipelineState,
        indexCount,
        instanceCount,
        startIndex,
        baseVertex,
        startInstance,
        entry->currentGraphicsRootCbvs[2],
        0,
        entry->currentGraphicsRootSrvs);
      for (size_t i = 0; i < g_currentRootCbvs.size(); ++i)
        g_currentRootCbvs[i] = entry->currentGraphicsRootCbvs[i];
      function(commandList, indexCount, instanceCount, startIndex, baseVertex, startInstance);
    }

    void STDMETHODCALLTYPE traceDispatch(ID3D12GraphicsCommandList* commandList, UINT x, UINT y, UINT z) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT)>(entry->dispatch);
      log("ID3D12GraphicsCommandList::Dispatch(x=%u, y=%u, z=%u)", x, y, z);
      function(commandList, x, y, z);
    }

    void STDMETHODCALLTYPE traceSetPipelineState(ID3D12GraphicsCommandList* commandList, ID3D12PipelineState* pipelineState) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*)>(entry->setPipelineState);
      entry->currentPipelineState = pipelineState;
      log("ID3D12GraphicsCommandList::SetPipelineState(list=%p, pipelineState=%p, caller=%p)",
        commandList, pipelineState, _ReturnAddress());
      function(commandList, pipelineState);
    }

    void STDMETHODCALLTYPE traceResourceBarrier(ID3D12GraphicsCommandList* commandList, UINT count, const D3D12_RESOURCE_BARRIER* barriers) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*)>(entry->resourceBarrier);
      log("ID3D12GraphicsCommandList::ResourceBarrier(count=%u)", count);
      recordD3D12ResourceBarrier(count, barriers);
      function(commandList, count, barriers);
    }

    void STDMETHODCALLTYPE traceExecuteIndirect(
        ID3D12GraphicsCommandList* commandList,
        ID3D12CommandSignature* commandSignature,
        UINT maxCommandCount,
        ID3D12Resource* argumentBuffer,
        UINT64 argumentBufferOffset,
        ID3D12Resource* countBuffer,
        UINT64 countBufferOffset) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12CommandSignature*, UINT,
        ID3D12Resource*, UINT64, ID3D12Resource*, UINT64)>(entry->executeIndirect);
      log("ID3D12GraphicsCommandList::ExecuteIndirect(maxCommands=%u, signature=%p, argumentBuffer=%p, countBuffer=%p)",
        maxCommandCount, commandSignature, argumentBuffer, countBuffer);
      function(commandList, commandSignature, maxCommandCount, argumentBuffer,
        argumentBufferOffset, countBuffer, countBufferOffset);
    }

    void STDMETHODCALLTYPE traceBeginRenderPass(
        ID3D12GraphicsCommandList* commandList,
        UINT renderTargetCount,
        const D3D12_RENDER_PASS_RENDER_TARGET_DESC* renderTargets,
        const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC* depthStencil,
        D3D12_RENDER_PASS_FLAGS flags) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT,
        const D3D12_RENDER_PASS_RENDER_TARGET_DESC*,
        const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC*,
        D3D12_RENDER_PASS_FLAGS)>(entry->beginRenderPass);
      log("ID3D12GraphicsCommandList4::BeginRenderPass(renderTargets=%u, depth=%p, flags=0x%x)",
        renderTargetCount, depthStencil, static_cast<unsigned>(flags));
      function(commandList, renderTargetCount, renderTargets, depthStencil, flags);
    }

    void STDMETHODCALLTYPE traceEndRenderPass(ID3D12GraphicsCommandList* commandList) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*)>(entry->endRenderPass);
      log("ID3D12GraphicsCommandList4::EndRenderPass()");
      function(commandList);
    }

    void STDMETHODCALLTYPE traceInitializeMetaCommand(
        ID3D12GraphicsCommandList* commandList,
        ID3D12MetaCommand* metaCommand,
        const void* initializationParametersData,
        SIZE_T initializationParametersDataSizeInBytes) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12MetaCommand*, const void*, SIZE_T)>(
          entry->initializeMetaCommand);
      log("ID3D12GraphicsCommandList4::InitializeMetaCommand(meta=%p, size=%zu)",
        metaCommand, initializationParametersDataSizeInBytes);
      function(commandList, metaCommand, initializationParametersData,
        initializationParametersDataSizeInBytes);
    }

    void STDMETHODCALLTYPE traceExecuteMetaCommand(
        ID3D12GraphicsCommandList* commandList,
        ID3D12MetaCommand* metaCommand,
        const void* executionParametersData,
        SIZE_T executionParametersDataSizeInBytes) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12MetaCommand*, const void*, SIZE_T)>(
          entry->executeMetaCommand);
      log("ID3D12GraphicsCommandList4::ExecuteMetaCommand(meta=%p, size=%zu)",
        metaCommand, executionParametersDataSizeInBytes);
      function(commandList, metaCommand, executionParametersData,
        executionParametersDataSizeInBytes);
    }

    void STDMETHODCALLTYPE traceBuildRaytracingAccelerationStructure(
        ID3D12GraphicsCommandList* commandList,
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc,
        UINT numPostbuildInfoDescs,
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* postbuildInfoDescs) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*,
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*, UINT,
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*)>(
          entry->buildRaytracingAccelerationStructure);
      log("ID3D12GraphicsCommandList4::BuildRaytracingAccelerationStructure(postbuild=%u)",
        numPostbuildInfoDescs);
      function(commandList, desc, numPostbuildInfoDescs, postbuildInfoDescs);
    }

    void STDMETHODCALLTYPE traceEmitRaytracingAccelerationStructurePostbuildInfo(
        ID3D12GraphicsCommandList* commandList,
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* desc,
        UINT numSourceAccelerationStructures,
        const D3D12_GPU_VIRTUAL_ADDRESS* sourceAccelerationStructureData) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*,
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*, UINT,
        const D3D12_GPU_VIRTUAL_ADDRESS*)>(
          entry->emitRaytracingAccelerationStructurePostbuildInfo);
      log("ID3D12GraphicsCommandList4::EmitRaytracingAccelerationStructurePostbuildInfo(count=%u)",
        numSourceAccelerationStructures);
      function(commandList, desc, numSourceAccelerationStructures,
        sourceAccelerationStructureData);
    }

    void STDMETHODCALLTYPE traceCopyRaytracingAccelerationStructure(
        ID3D12GraphicsCommandList* commandList,
        D3D12_GPU_VIRTUAL_ADDRESS destination,
        D3D12_GPU_VIRTUAL_ADDRESS source,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE mode) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, D3D12_GPU_VIRTUAL_ADDRESS,
        D3D12_GPU_VIRTUAL_ADDRESS,
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE)>(
          entry->copyRaytracingAccelerationStructure);
      log("ID3D12GraphicsCommandList4::CopyRaytracingAccelerationStructure(mode=%u)",
        static_cast<unsigned>(mode));
      function(commandList, destination, source, mode);
    }

    void STDMETHODCALLTYPE traceSetPipelineState1(
        ID3D12GraphicsCommandList* commandList,
        ID3D12StateObject* stateObject) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12StateObject*)>(entry->setPipelineState1);
      log("ID3D12GraphicsCommandList4::SetPipelineState1(state=%p)", stateObject);
      function(commandList, stateObject);
    }

    void STDMETHODCALLTYPE traceDispatchRays(
        ID3D12GraphicsCommandList* commandList,
        const D3D12_DISPATCH_RAYS_DESC* desc) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, const D3D12_DISPATCH_RAYS_DESC*)>(
          entry->dispatchRays);
      log("ID3D12GraphicsCommandList4::DispatchRays(desc=%p)", desc);
      function(commandList, desc);
    }

    void STDMETHODCALLTYPE traceRSSetShadingRate(
        ID3D12GraphicsCommandList* commandList,
        D3D12_SHADING_RATE shadingRate,
        const D3D12_SHADING_RATE_COMBINER* combiners) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, D3D12_SHADING_RATE,
        const D3D12_SHADING_RATE_COMBINER*)>(entry->rsSetShadingRate);
      log("ID3D12GraphicsCommandList5::RSSetShadingRate(rate=%u)",
        static_cast<unsigned>(shadingRate));
      function(commandList, shadingRate, combiners);
    }

    void STDMETHODCALLTYPE traceRSSetShadingRateImage(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* shadingRateImage) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, ID3D12Resource*)>(entry->rsSetShadingRateImage);
      log("ID3D12GraphicsCommandList5::RSSetShadingRateImage(resource=%p)",
        shadingRateImage);
      function(commandList, shadingRateImage);
    }

    void STDMETHODCALLTYPE traceDispatchMesh(
        ID3D12GraphicsCommandList* commandList,
        UINT threadGroupCountX,
        UINT threadGroupCountY,
        UINT threadGroupCountZ) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, UINT, UINT)>(entry->dispatchMesh);
      log("ID3D12GraphicsCommandList6::DispatchMesh(x=%u, y=%u, z=%u)",
        threadGroupCountX, threadGroupCountY, threadGroupCountZ);
      function(commandList, threadGroupCountX, threadGroupCountY, threadGroupCountZ);
    }

    void STDMETHODCALLTYPE traceBarrier(
        ID3D12GraphicsCommandList* commandList,
        UINT groupCount,
        const void* groups) {
      auto* entry = findEntry(commandList);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12GraphicsCommandList*, UINT, const void*)>(entry->barrier);
      log("ID3D12GraphicsCommandList7::Barrier(groups=%u, data=%p)",
        groupCount, groups);
      function(commandList, groupCount, groups);
    }

  }

  bool installCommandListTrace(ID3D12GraphicsCommandList* commandList) {
    if (!commandList) {
      log("Command list trace installation failed: null command list");
      return false;
    }

    void** originalVtable = *reinterpret_cast<void***>(commandList);
    {
      std::lock_guard lock(g_traceMutex);
      auto entry = g_traceEntries.find(commandList);
      if (entry != g_traceEntries.end()) {
        void** clone = entry->second.vtable;
        if (originalVtable != clone) {
          DWORD protection = 0;
          if (!VirtualProtect(reinterpret_cast<void*>(commandList), sizeof(void*), PAGE_READWRITE, &protection)) {
            log("Command list trace reinstallation failed: VirtualProtect error=%lu", GetLastError());
            return false;
          }

          *reinterpret_cast<void***>(commandList) = clone;
          DWORD ignored = 0;
          VirtualProtect(reinterpret_cast<void*>(commandList), sizeof(void*), protection, &ignored);
          log("Reinstalled command list trace at %p", commandList);
        }
        return true;
      }
    }

    auto* clone = static_cast<void**>(VirtualAlloc(
      nullptr,
      kGraphicsCommandListMethodCount * sizeof(void*),
      MEM_COMMIT | MEM_RESERVE,
      PAGE_READWRITE));
    if (!clone) {
      log("Command list trace installation failed: VirtualAlloc error=%lu", GetLastError());
      return false;
    }

    std::memcpy(clone, originalVtable, kGraphicsCommandListMethodCount * sizeof(void*));

    TraceEntry entry {
      clone,
      clone[kQueryInterfaceIndex],
      clone[kClearStateIndex],
      clone[kCloseIndex],
      clone[kResetIndex],
      clone[kCopyBufferRegionIndex],
      clone[kCopyTextureRegionIndex],
      clone[kCopyResourceIndex],
      clone[kResolveSubresourceIndex],
      clone[kIASetPrimitiveTopologyIndex],
      clone[kRSSetViewportsIndex],
      clone[kRSSetScissorRectsIndex],
      clone[kOMSetStencilRefIndex],
      clone[kExecuteBundleIndex],
      clone[kSetDescriptorHeapsIndex],
      clone[kSetComputeRootSignatureIndex],
      clone[kSetGraphicsRootSignatureIndex],
      clone[kSetComputeRootDescriptorTableIndex],
      clone[kSetGraphicsRootDescriptorTableIndex],
      clone[kSetComputeRoot32BitConstantsIndex],
      clone[kSetGraphicsRoot32BitConstantsIndex],
      clone[kSetGraphicsRootConstantBufferViewIndex],
      clone[kSetGraphicsRootShaderResourceViewIndex],
      clone[kIASetIndexBufferIndex],
      clone[kIASetVertexBuffersIndex],
      clone[kOMSetRenderTargetsIndex],
      clone[kClearDepthStencilViewIndex],
      clone[kClearRenderTargetViewIndex],
      clone[kDrawInstancedIndex],
      clone[kDrawIndexedInstancedIndex],
      clone[kDispatchIndex],
      clone[kSetPipelineStateIndex],
      clone[kResourceBarrierIndex],
      clone[kExecuteIndirectIndex],
      clone[kBeginRenderPassIndex],
      clone[kEndRenderPassIndex],
      clone[kInitializeMetaCommandIndex],
      clone[kExecuteMetaCommandIndex],
      clone[kBuildRaytracingAccelerationStructureIndex],
      clone[kEmitRaytracingAccelerationStructurePostbuildInfoIndex],
      clone[kCopyRaytracingAccelerationStructureIndex],
      clone[kSetPipelineState1Index],
      clone[kDispatchRaysIndex],
      clone[kRSSetShadingRateIndex],
      clone[kRSSetShadingRateImageIndex],
      clone[kDispatchMeshIndex],
      clone[kBarrierIndex],
      nullptr,
      {},
      {},
      D3D_PRIMITIVE_TOPOLOGY_UNDEFINED,
      {},
      {},
    };

    clone[kQueryInterfaceIndex] = reinterpret_cast<void*>(&traceQueryInterface);
    clone[kClearStateIndex] = reinterpret_cast<void*>(&traceClearState);
    clone[kCloseIndex] = reinterpret_cast<void*>(&traceClose);
    clone[kResetIndex] = reinterpret_cast<void*>(&traceReset);
    clone[kCopyBufferRegionIndex] = reinterpret_cast<void*>(&traceCopyBufferRegion);
    clone[kCopyTextureRegionIndex] = reinterpret_cast<void*>(&traceCopyTextureRegion);
    clone[kCopyResourceIndex] = reinterpret_cast<void*>(&traceCopyResource);
    clone[kResolveSubresourceIndex] = reinterpret_cast<void*>(&traceResolveSubresource);
    clone[kIASetPrimitiveTopologyIndex] = reinterpret_cast<void*>(&traceIASetPrimitiveTopology);
    clone[kRSSetViewportsIndex] = reinterpret_cast<void*>(&traceRSSetViewports);
    clone[kRSSetScissorRectsIndex] = reinterpret_cast<void*>(&traceRSSetScissorRects);
    clone[kOMSetStencilRefIndex] = reinterpret_cast<void*>(&traceOMSetStencilRef);
    clone[kExecuteBundleIndex] = reinterpret_cast<void*>(&traceExecuteBundle);
    clone[kSetDescriptorHeapsIndex] = reinterpret_cast<void*>(&traceSetDescriptorHeaps);
    clone[kSetComputeRootSignatureIndex] = reinterpret_cast<void*>(&traceSetComputeRootSignature);
    clone[kSetGraphicsRootSignatureIndex] = reinterpret_cast<void*>(&traceSetGraphicsRootSignature);
    clone[kSetComputeRootDescriptorTableIndex] = reinterpret_cast<void*>(&traceSetComputeRootDescriptorTable);
    clone[kSetGraphicsRootDescriptorTableIndex] = reinterpret_cast<void*>(&traceSetGraphicsRootDescriptorTable);
    clone[kSetComputeRoot32BitConstantsIndex] = reinterpret_cast<void*>(&traceSetComputeRoot32BitConstants);
    clone[kSetGraphicsRoot32BitConstantsIndex] = reinterpret_cast<void*>(&traceSetGraphicsRoot32BitConstants);
    clone[kSetGraphicsRootConstantBufferViewIndex] = reinterpret_cast<void*>(&traceSetGraphicsRootConstantBufferView);
    clone[kSetGraphicsRootShaderResourceViewIndex] = reinterpret_cast<void*>(&traceSetGraphicsRootShaderResourceView);
    clone[kIASetIndexBufferIndex] = reinterpret_cast<void*>(&traceIASetIndexBuffer);
    clone[kIASetVertexBuffersIndex] = reinterpret_cast<void*>(&traceIASetVertexBuffers);
    clone[kOMSetRenderTargetsIndex] = reinterpret_cast<void*>(&traceOMSetRenderTargets);
    clone[kClearDepthStencilViewIndex] = reinterpret_cast<void*>(&traceClearDepthStencilView);
    clone[kClearRenderTargetViewIndex] = reinterpret_cast<void*>(&traceClearRenderTargetView);
    clone[kDrawInstancedIndex] = reinterpret_cast<void*>(&traceDrawInstanced);
    clone[kDrawIndexedInstancedIndex] = reinterpret_cast<void*>(&traceDrawIndexedInstanced);
    clone[kDispatchIndex] = reinterpret_cast<void*>(&traceDispatch);
    clone[kSetPipelineStateIndex] = reinterpret_cast<void*>(&traceSetPipelineState);
    clone[kResourceBarrierIndex] = reinterpret_cast<void*>(&traceResourceBarrier);
    clone[kExecuteIndirectIndex] = reinterpret_cast<void*>(&traceExecuteIndirect);
    clone[kBeginRenderPassIndex] = reinterpret_cast<void*>(&traceBeginRenderPass);
    clone[kEndRenderPassIndex] = reinterpret_cast<void*>(&traceEndRenderPass);
    clone[kInitializeMetaCommandIndex] = reinterpret_cast<void*>(&traceInitializeMetaCommand);
    clone[kExecuteMetaCommandIndex] = reinterpret_cast<void*>(&traceExecuteMetaCommand);
    clone[kBuildRaytracingAccelerationStructureIndex] = reinterpret_cast<void*>(&traceBuildRaytracingAccelerationStructure);
    clone[kEmitRaytracingAccelerationStructurePostbuildInfoIndex] = reinterpret_cast<void*>(&traceEmitRaytracingAccelerationStructurePostbuildInfo);
    clone[kCopyRaytracingAccelerationStructureIndex] = reinterpret_cast<void*>(&traceCopyRaytracingAccelerationStructure);
    clone[kSetPipelineState1Index] = reinterpret_cast<void*>(&traceSetPipelineState1);
    clone[kDispatchRaysIndex] = reinterpret_cast<void*>(&traceDispatchRays);
    clone[kRSSetShadingRateIndex] = reinterpret_cast<void*>(&traceRSSetShadingRate);
    clone[kRSSetShadingRateImageIndex] = reinterpret_cast<void*>(&traceRSSetShadingRateImage);
    clone[kDispatchMeshIndex] = reinterpret_cast<void*>(&traceDispatchMesh);
    clone[kBarrierIndex] = reinterpret_cast<void*>(&traceBarrier);

    {
      std::lock_guard lock(g_traceMutex);
      g_traceEntries.emplace(commandList, entry);
    }

    DWORD protection = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(commandList), sizeof(void*), PAGE_READWRITE, &protection)) {
      log("Command list trace installation failed: VirtualProtect error=%lu", GetLastError());
      VirtualFree(clone, 0, MEM_RELEASE);
      return false;
    }

    *reinterpret_cast<void***>(commandList) = clone;
    VirtualProtect(reinterpret_cast<void*>(commandList), sizeof(void*), protection, &protection);
    log("Installed command list trace at %p", commandList);
    return true;
  }

}
