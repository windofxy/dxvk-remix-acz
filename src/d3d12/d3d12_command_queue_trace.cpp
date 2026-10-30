#include "d3d12_command_queue_trace.h"

#include "d3d12_command_list_trace.h"
#include "d3d12_device_extension_trace.h"
#include "d3d12_device_proxy.h"
#include "d3d12_device_raw_trace.h"
#include "d3d12_runtime_adapter.h"

#include <windows.h>

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace remix {

  namespace {

    // IUnknown + ID3D12Object + ID3D12DeviceChild + queue methods.
    constexpr size_t kCommandQueueMethodCount = 19;
    // ID3D12DeviceChild::GetDevice precedes the command-queue-specific methods.
    constexpr size_t kGetDeviceIndex = 7;
    constexpr size_t kExecuteCommandListsIndex = 10;
    constexpr size_t kSignalIndex = 14;
    constexpr size_t kWaitIndex = 15;

    struct TraceEntry {
      void** vtable;
      void* getDevice;
      void* executeCommandLists;
      void* signal;
      void* wait;
    };

    std::mutex g_traceMutex;
    std::unordered_map<ID3D12CommandQueue*, TraceEntry> g_traceEntries;

    TraceEntry* findEntry(ID3D12CommandQueue* commandQueue) {
      std::lock_guard lock(g_traceMutex);
      auto entry = g_traceEntries.find(commandQueue);
      return entry == g_traceEntries.end() ? nullptr : &entry->second;
    }

    void STDMETHODCALLTYPE traceExecuteCommandLists(
        ID3D12CommandQueue* commandQueue,
        UINT count,
        ID3D12CommandList* const* commandLists) {
      auto* entry = findEntry(commandQueue);
      auto function = reinterpret_cast<void (STDMETHODCALLTYPE*)(
        ID3D12CommandQueue*, UINT, ID3D12CommandList* const*)>(entry->executeCommandLists);

      // Command lists can be created through an unwrapped device interface and
      // only become observable when they are submitted. Install tracing here
      // so those lists are covered without replacing the queue or device.
      if (commandLists) {
        for (UINT i = 0; i < count; ++i) {
          auto* commandList = commandLists[i];
          if (!commandList)
            continue;

          installCommandListInterfaceTraces(commandList);
        }
      }

      log("ID3D12CommandQueue::ExecuteCommandLists(count=%u, first=%p)",
        count, count && commandLists ? commandLists[0] : nullptr);
      recordD3D12Submit(commandQueue, count, commandLists);
      function(commandQueue, count, commandLists);
    }

    HRESULT STDMETHODCALLTYPE traceGetDevice(
        ID3D12CommandQueue* commandQueue,
        REFIID riid,
        void** device) {
      auto* entry = findEntry(commandQueue);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(
        ID3D12CommandQueue*, REFIID, void**)>(entry->getDevice);
      HRESULT result = function(commandQueue, riid, device);

      bool deviceInterface = riid == __uuidof(ID3D12Device)
        || riid == __uuidof(ID3D12Device1)
        || riid == __uuidof(ID3D12Device2)
        || riid == __uuidof(ID3D12Device3)
        || isDevice4OrLaterIid(riid);
      bool traceInstalled = false;
      if (SUCCEEDED(result) && deviceInterface && device && *device)
        traceInstalled = installRawDeviceTrace(
          reinterpret_cast<ID3D12Device*>(*device));

      log("ID3D12CommandQueue::GetDevice(riid=%s, result=0x%08lx, device=%p, traceInstalled=%u)",
        formatIid(riid), static_cast<unsigned long>(result),
        device ? *device : nullptr, traceInstalled ? 1u : 0u);
      return result;
    }

    HRESULT STDMETHODCALLTYPE traceSignal(
        ID3D12CommandQueue* commandQueue,
        ID3D12Fence* fence,
        UINT64 value) {
      auto* entry = findEntry(commandQueue);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(
        ID3D12CommandQueue*, ID3D12Fence*, UINT64)>(entry->signal);
      HRESULT result = function(commandQueue, fence, value);
      log("ID3D12CommandQueue::Signal(fence=%p, value=%llu, result=0x%08lx)",
        fence, value, static_cast<unsigned long>(result));
      recordD3D12FenceSignal(commandQueue, fence, value, result);
      return result;
    }

    HRESULT STDMETHODCALLTYPE traceWait(
        ID3D12CommandQueue* commandQueue,
        ID3D12Fence* fence,
        UINT64 value) {
      auto* entry = findEntry(commandQueue);
      auto function = reinterpret_cast<HRESULT (STDMETHODCALLTYPE*)(
        ID3D12CommandQueue*, ID3D12Fence*, UINT64)>(entry->wait);
      HRESULT result = function(commandQueue, fence, value);
      log("ID3D12CommandQueue::Wait(fence=%p, value=%llu, result=0x%08lx)",
        fence, value, static_cast<unsigned long>(result));
      recordD3D12FenceWait(commandQueue, fence, value, result);
      return result;
    }

  }

  bool installCommandQueueTrace(ID3D12CommandQueue* commandQueue) {
    if (!commandQueue) {
      log("Command queue trace installation failed: null command queue");
      return false;
    }

    {
      std::lock_guard lock(g_traceMutex);
      if (g_traceEntries.find(commandQueue) != g_traceEntries.end())
        return true;
    }

    void** originalVtable = *reinterpret_cast<void***>(commandQueue);
    auto* clone = static_cast<void**>(VirtualAlloc(
      nullptr,
      kCommandQueueMethodCount * sizeof(void*),
      MEM_COMMIT | MEM_RESERVE,
      PAGE_READWRITE));
    if (!clone) {
      log("Command queue trace installation failed: VirtualAlloc error=%lu", GetLastError());
      return false;
    }

    std::memcpy(clone, originalVtable, kCommandQueueMethodCount * sizeof(void*));
    TraceEntry entry {
      clone,
      clone[kGetDeviceIndex],
      clone[kExecuteCommandListsIndex],
      clone[kSignalIndex],
      clone[kWaitIndex],
    };
    clone[kGetDeviceIndex] = reinterpret_cast<void*>(&traceGetDevice);
    clone[kExecuteCommandListsIndex] = reinterpret_cast<void*>(&traceExecuteCommandLists);
    clone[kSignalIndex] = reinterpret_cast<void*>(&traceSignal);
    clone[kWaitIndex] = reinterpret_cast<void*>(&traceWait);

    {
      std::lock_guard lock(g_traceMutex);
      g_traceEntries.emplace(commandQueue, entry);
    }

    DWORD protection = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(commandQueue), sizeof(void*), PAGE_READWRITE, &protection)) {
      log("Command queue trace installation failed: VirtualProtect error=%lu", GetLastError());
      VirtualFree(clone, 0, MEM_RELEASE);
      return false;
    }

    *reinterpret_cast<void***>(commandQueue) = clone;
    VirtualProtect(reinterpret_cast<void*>(commandQueue), sizeof(void*), protection, &protection);
    log("Installed command queue trace at %p", commandQueue);
    return true;
  }

}
