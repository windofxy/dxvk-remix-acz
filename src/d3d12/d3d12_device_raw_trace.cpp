#include "d3d12_device_raw_trace.h"

#include "d3d12_command_list_trace.h"
#include "d3d12_device_proxy.h"
#include "d3d12_runtime_adapter.h"

#include <windows.h>

#include <mutex>
#include <unordered_map>

namespace remix {

  namespace {

    // Vtable slots of ID3D12Device. Newer interface versions append methods, so
    // patching the native vtable observes every device interface at once, which
    // is what keeps capture complete when the game asks for ID3D12Device3+.
    constexpr size_t kCreateCommandListIndex = 12;
    constexpr size_t kCreateDescriptorHeapIndex = 14;
    constexpr size_t kCreateConstantBufferViewIndex = 17;
    constexpr size_t kCreateShaderResourceViewIndex = 18;
    constexpr size_t kCreateUnorderedAccessViewIndex = 19;
    constexpr size_t kCreateSamplerIndex = 22;
    constexpr size_t kCopyDescriptorsIndex = 23;
    constexpr size_t kCopyDescriptorsSimpleIndex = 24;
    constexpr size_t kCreateCommittedResourceIndex = 27;

    using CreateCommandListFunction = HRESULT (STDMETHODCALLTYPE*)(
      ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE,
      ID3D12CommandAllocator*, ID3D12PipelineState*, REFIID, void**);
    using CreateDescriptorHeapFunction = HRESULT (STDMETHODCALLTYPE*)(
      ID3D12Device*, const D3D12_DESCRIPTOR_HEAP_DESC*, REFIID, void**);
    using CreateConstantBufferViewFunction = void (STDMETHODCALLTYPE*)(
      ID3D12Device*, const D3D12_CONSTANT_BUFFER_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
    using CreateShaderResourceViewFunction = void (STDMETHODCALLTYPE*)(
      ID3D12Device*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
    using CreateUnorderedAccessViewFunction = void (STDMETHODCALLTYPE*)(
      ID3D12Device*, ID3D12Resource*, ID3D12Resource*,
      const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
    using CreateSamplerFunction = void (STDMETHODCALLTYPE*)(
      ID3D12Device*, const D3D12_SAMPLER_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
    using CopyDescriptorsFunction = void (STDMETHODCALLTYPE*)(
      ID3D12Device*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*,
      UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, D3D12_DESCRIPTOR_HEAP_TYPE);
    using CopyDescriptorsSimpleFunction = void (STDMETHODCALLTYPE*)(
      ID3D12Device*, UINT, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE,
      D3D12_DESCRIPTOR_HEAP_TYPE);
    using CreateCommittedResourceFunction = HRESULT (STDMETHODCALLTYPE*)(
      ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
      const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
      REFIID, void**);

    struct RawDeviceTrace {
      CreateCommandListFunction createCommandList = nullptr;
      CreateDescriptorHeapFunction createDescriptorHeap = nullptr;
      CreateConstantBufferViewFunction createConstantBufferView = nullptr;
      CreateShaderResourceViewFunction createShaderResourceView = nullptr;
      CreateUnorderedAccessViewFunction createUnorderedAccessView = nullptr;
      CreateSamplerFunction createSampler = nullptr;
      CopyDescriptorsFunction copyDescriptors = nullptr;
      CopyDescriptorsSimpleFunction copyDescriptorsSimple = nullptr;
      CreateCommittedResourceFunction createCommittedResource = nullptr;
    };

    std::mutex g_traceMutex;
    std::unordered_map<void**, RawDeviceTrace> g_patchedVtables;

    RawDeviceTrace* findTrace(void* device) {
      if (!device)
        return nullptr;
      void** vtable = *reinterpret_cast<void***>(device);
      std::lock_guard lock(g_traceMutex);
      auto entry = g_patchedVtables.find(vtable);
      return entry == g_patchedVtables.end() ? nullptr : &entry->second;
    }

    HRESULT STDMETHODCALLTYPE traceCreateCommandList(
        ID3D12Device* device,
        UINT nodeMask,
        D3D12_COMMAND_LIST_TYPE type,
        ID3D12CommandAllocator* commandAllocator,
        ID3D12PipelineState* initialState,
        REFIID riid,
        void** commandList) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->createCommandList) {
        log("Raw ID3D12Device::CreateCommandList missing vtable entry (device=%p)", device);
        return E_UNEXPECTED;
      }

      HRESULT result = trace->createCommandList(
        device, nodeMask, type, commandAllocator, initialState, riid, commandList);

      bool traceSupported = isGraphicsCommandListIid(riid);
      bool traceInstalled = false;
      if (SUCCEEDED(result) && traceSupported && commandList && *commandList)
        traceInstalled = installCommandListTrace(
          static_cast<ID3D12GraphicsCommandList*>(*commandList));

      log("Raw ID3D12Device::CreateCommandList(node=%u, type=%u, riid=%s, result=0x%08lx, object=%p, traceInstalled=%u)",
        nodeMask,
        static_cast<unsigned>(type),
        formatIid(riid),
        static_cast<unsigned long>(result),
        commandList ? *commandList : nullptr,
        traceInstalled ? 1u : 0u);
      return result;
    }

    HRESULT STDMETHODCALLTYPE traceCreateDescriptorHeap(
        ID3D12Device* device,
        const D3D12_DESCRIPTOR_HEAP_DESC* description,
        REFIID riid,
        void** heap) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->createDescriptorHeap)
        return E_UNEXPECTED;

      const HRESULT result = trace->createDescriptorHeap(device, description, riid, heap);
      if (SUCCEEDED(result) && heap && *heap && description) {
        auto* created = static_cast<ID3D12DescriptorHeap*>(*heap);
        const D3D12_CPU_DESCRIPTOR_HANDLE cpuStart = created->GetCPUDescriptorHandleForHeapStart();
        const D3D12_GPU_DESCRIPTOR_HANDLE gpuStart = created->GetGPUDescriptorHandleForHeapStart();
        const UINT increment =
          device->GetDescriptorHandleIncrementSize(description->Type);
        recordD3D12DescriptorHeap(created, *description, cpuStart, gpuStart, increment);
      }
      return result;
    }

    void STDMETHODCALLTYPE traceCreateConstantBufferView(
        ID3D12Device* device,
        const D3D12_CONSTANT_BUFFER_VIEW_DESC* description,
        D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->createConstantBufferView)
        return;
      trace->createConstantBufferView(device, description, destDescriptor);
      recordD3D12ConstantBufferView(destDescriptor, description);
    }

    void STDMETHODCALLTYPE traceCreateShaderResourceView(
        ID3D12Device* device,
        ID3D12Resource* resource,
        const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
        D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->createShaderResourceView)
        return;
      trace->createShaderResourceView(device, resource, desc, destDescriptor);
      recordD3D12View("SRV", destDescriptor, resource,
        desc ? desc->Format : DXGI_FORMAT_UNKNOWN, desc, desc ? sizeof(*desc) : 0);
    }

    void STDMETHODCALLTYPE traceCreateUnorderedAccessView(
        ID3D12Device* device,
        ID3D12Resource* resource,
        ID3D12Resource* counterResource,
        const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
        D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->createUnorderedAccessView)
        return;
      trace->createUnorderedAccessView(device, resource, counterResource, desc, destDescriptor);
      recordD3D12View("UAV", destDescriptor, resource,
        desc ? desc->Format : DXGI_FORMAT_UNKNOWN, desc, desc ? sizeof(*desc) : 0);
    }

    void STDMETHODCALLTYPE traceCreateSampler(
        ID3D12Device* device,
        const D3D12_SAMPLER_DESC* description,
        D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->createSampler)
        return;
      trace->createSampler(device, description, destDescriptor);
      recordD3D12Sampler(destDescriptor, description);
    }

    void STDMETHODCALLTYPE traceCopyDescriptors(
        ID3D12Device* device,
        UINT numDestDescriptorRanges,
        const D3D12_CPU_DESCRIPTOR_HANDLE* destDescriptorRangeStarts,
        const UINT* destDescriptorRangeSizes,
        UINT numSrcDescriptorRanges,
        const D3D12_CPU_DESCRIPTOR_HANDLE* srcDescriptorRangeStarts,
        const UINT* srcDescriptorRangeSizes,
        D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapsType) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->copyDescriptors)
        return;
      trace->copyDescriptors(device, numDestDescriptorRanges, destDescriptorRangeStarts,
        destDescriptorRangeSizes, numSrcDescriptorRanges, srcDescriptorRangeStarts,
        srcDescriptorRangeSizes, descriptorHeapsType);
      recordD3D12CopyDescriptors(
        numDestDescriptorRanges, destDescriptorRangeStarts, destDescriptorRangeSizes,
        numSrcDescriptorRanges, srcDescriptorRangeStarts, srcDescriptorRangeSizes,
        device->GetDescriptorHandleIncrementSize(descriptorHeapsType));
    }

    void STDMETHODCALLTYPE traceCopyDescriptorsSimple(
        ID3D12Device* device,
        UINT numDescriptors,
        D3D12_CPU_DESCRIPTOR_HANDLE destDescriptorRangeStart,
        D3D12_CPU_DESCRIPTOR_HANDLE srcDescriptorRangeStart,
        D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapsType) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->copyDescriptorsSimple)
        return;
      trace->copyDescriptorsSimple(device, numDescriptors, destDescriptorRangeStart,
        srcDescriptorRangeStart, descriptorHeapsType);
      recordD3D12CopyDescriptorsSimple(numDescriptors, destDescriptorRangeStart,
        srcDescriptorRangeStart, device->GetDescriptorHandleIncrementSize(descriptorHeapsType));
    }

    HRESULT STDMETHODCALLTYPE traceCreateCommittedResource(
        ID3D12Device* device,
        const D3D12_HEAP_PROPERTIES* heapProperties,
        D3D12_HEAP_FLAGS heapFlags,
        const D3D12_RESOURCE_DESC* description,
        D3D12_RESOURCE_STATES initialState,
        const D3D12_CLEAR_VALUE* optimizedClearValue,
        REFIID riidResource,
        void** resource) {
      RawDeviceTrace* trace = findTrace(device);
      if (!trace || !trace->createCommittedResource)
        return E_UNEXPECTED;

      const HRESULT result = trace->createCommittedResource(device, heapProperties, heapFlags,
        description, initialState, optimizedClearValue, riidResource, resource);
      if (SUCCEEDED(result) && resource && *resource) {
        recordD3D12Resource(heapProperties, description,
          static_cast<ID3D12Resource*>(*resource), result);
      }
      return result;
    }

  }

  bool installRawDeviceTrace(ID3D12Device* device) {
    if (!device) {
      log("Raw device trace installation failed: null device");
      return false;
    }

    void** vtable = *reinterpret_cast<void***>(device);
    {
      std::lock_guard lock(g_traceMutex);
      if (g_patchedVtables.find(vtable) != g_patchedVtables.end())
        return true;
    }

    const bool required =
      vtable[kCreateCommandListIndex] && vtable[kCreateShaderResourceViewIndex];
    if (!required) {
      log("Raw device trace installation failed: null required slot (device=%p)", device);
      return false;
    }

    RawDeviceTrace trace {
      reinterpret_cast<CreateCommandListFunction>(vtable[kCreateCommandListIndex]),
      reinterpret_cast<CreateDescriptorHeapFunction>(vtable[kCreateDescriptorHeapIndex]),
      reinterpret_cast<CreateConstantBufferViewFunction>(vtable[kCreateConstantBufferViewIndex]),
      reinterpret_cast<CreateShaderResourceViewFunction>(vtable[kCreateShaderResourceViewIndex]),
      reinterpret_cast<CreateUnorderedAccessViewFunction>(vtable[kCreateUnorderedAccessViewIndex]),
      reinterpret_cast<CreateSamplerFunction>(vtable[kCreateSamplerIndex]),
      reinterpret_cast<CopyDescriptorsFunction>(vtable[kCopyDescriptorsIndex]),
      reinterpret_cast<CopyDescriptorsSimpleFunction>(vtable[kCopyDescriptorsSimpleIndex]),
      reinterpret_cast<CreateCommittedResourceFunction>(vtable[kCreateCommittedResourceIndex]),
    };

    DWORD protection = 0;
    if (!VirtualProtect(vtable, sizeof(void*) * 40, PAGE_READWRITE, &protection)) {
      log("Raw device trace installation failed: VirtualProtect error=%lu", GetLastError());
      return false;
    }

    vtable[kCreateCommandListIndex] = reinterpret_cast<void*>(&traceCreateCommandList);
    vtable[kCreateDescriptorHeapIndex] = reinterpret_cast<void*>(&traceCreateDescriptorHeap);
    vtable[kCreateConstantBufferViewIndex] = reinterpret_cast<void*>(&traceCreateConstantBufferView);
    vtable[kCreateShaderResourceViewIndex] = reinterpret_cast<void*>(&traceCreateShaderResourceView);
    vtable[kCreateUnorderedAccessViewIndex] = reinterpret_cast<void*>(&traceCreateUnorderedAccessView);
    vtable[kCreateSamplerIndex] = reinterpret_cast<void*>(&traceCreateSampler);
    vtable[kCopyDescriptorsIndex] = reinterpret_cast<void*>(&traceCopyDescriptors);
    vtable[kCopyDescriptorsSimpleIndex] = reinterpret_cast<void*>(&traceCopyDescriptorsSimple);
    vtable[kCreateCommittedResourceIndex] = reinterpret_cast<void*>(&traceCreateCommittedResource);

    DWORD ignored = 0;
    VirtualProtect(vtable, sizeof(void*) * 40, protection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), vtable, sizeof(void*) * 40);

    {
      std::lock_guard lock(g_traceMutex);
      g_patchedVtables.emplace(vtable, trace);
    }

    log("Installed raw ID3D12Device vtable capture at device=%p, vtable=%p", device, vtable);
    return true;
  }

}
