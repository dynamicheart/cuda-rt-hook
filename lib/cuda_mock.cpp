#include "cuda_mock.h"

#include <dlfcn.h>
#include <csetjmp>
#include <string.h>

#include "backtrace.h"
#include "cuda_op_tracer.h"
#include "hook.h"
#include <stdint.h>
#include <map>
#include <mutex>

#include "backtrace.h"
#include "cuda_op_tracer.h"
#include "hook.h"
#include "logger/logger.h"

std::jmp_buf log_jump_buffer = {{}};

#ifdef __cplusplus

namespace {

typedef int (*XpuMallocPtr)(void**, uint64_t, int);
typedef int (*XpuFreePtr)(void*);
typedef int (*XpuCurrentDevicePtr)(int* devid);
XpuMallocPtr xpu_malloc_fn_ptr = nullptr;
XpuFreePtr xpu_free_fn_ptr = nullptr;
XpuCurrentDevicePtr xpu_current_device_fn_ptr = nullptr;

std::mutex xpu_malloc_mutex;
std::vector<uint64_t> gm_allocated(8, 0);
std::vector<uint64_t> l3_allocated(8, 0);
std::vector<uint64_t> gm_allocated_peak(8, 0);
std::vector<uint64_t> l3_allocated_peak(8, 0);

std::vector<std::map<void*, std::pair<int, uint64_t>>> allocated_map(8);

int recorded_xpu_malloc(void** pdevptr, uint64_t sz, int kind) {
    std::lock_guard<std::mutex> lock(xpu_malloc_mutex);
    int current_devid = 0;
    int r = xpu_current_device_fn_ptr(&current_devid);
    if (r != 0) {
        return r;
    }

    r = xpu_malloc_fn_ptr(pdevptr, sz, kind);
    if (r != 0) {
        LOG(WARN) << "Device " << current_devid << " malloc failed, size: " << sz << ", kind: " << kind << " gm allocated: " << gm_allocated[current_devid];
        return r;
    }

    if (kind == 0) {
        gm_allocated[current_devid] += sz;
        gm_allocated_peak[current_devid] = std::max(gm_allocated_peak[current_devid], gm_allocated[current_devid]);
    } else if (kind == 1) {
        l3_allocated[current_devid] += sz;
        l3_allocated_peak[current_devid] = std::max(l3_allocated_peak[current_devid], l3_allocated[current_devid]);
    }

    allocated_map[current_devid][*pdevptr] = std::make_pair(current_devid, sz);

    LOG(WARN) << "Device " << current_devid << " malloc size: " << sz << ", kind: " << kind << ", peak: " << gm_allocated_peak[current_devid];

    return r;
}

int recorded_xpu_free(void* devptr) {
    std::lock_guard<std::mutex> lock(xpu_malloc_mutex);
    int current_devid = 0;
    int r = xpu_current_device_fn_ptr(&current_devid);
    if (r != 0) {
        return r;
    }

    r = xpu_free_fn_ptr(devptr);
    if (r != 0) {
        return r;
    }

    auto it = allocated_map[current_devid].find(devptr);
    if (it == allocated_map[current_devid].end()) {
        return r;
    }

    int type = it->second.first;
    uint64_t sz = it->second.second;

    if (type == 0) {
        gm_allocated[current_devid] -= sz;
    } else if (type == 1){
        l3_allocated[current_devid] -= sz;
    }
    allocated_map[current_devid].erase(it);
    return r;
}

struct DeviceMallocFreeHook : public hook::HookInstallerWrap<DeviceMallocFreeHook> {
    bool targetLib(const char* name) {
        // return strstr(name, "libpaddle.so") || strstr(name, "libxpuapi.so") || strstr(name, "libphi.so");
        return !strstr(name, "libxpurt.so.1") && !strstr(name, "libxpurt.so");
    }
    bool targetSym(const char* name) {
        return strstr(name, "xpu_malloc") || strstr(name, "xpu_free") || strstr(name, "xpu_current_device") /*|| strstr(name, "xpu_wait")*/;
    }
    void* newFuncPtr(const hook::OriginalInfo& info) {
        if (strstr(curSymName(), "xpu_malloc")) {
            LOG(WARN) << info.libName << '\t' << "xpu_malloc";
            if (!xpu_malloc_fn_ptr) {
                xpu_malloc_fn_ptr = (XpuMallocPtr)info.oldFuncPtr;
            }
            return reinterpret_cast<void*>(&recorded_xpu_malloc);
        } else if (strstr(curSymName(), "xpu_free")) {
            LOG(WARN) << info.libName << '\t' << "xpu_free";
            if (!xpu_free_fn_ptr) {
                xpu_free_fn_ptr = (XpuFreePtr)info.oldFuncPtr;
            }
            return reinterpret_cast<void*>(&recorded_xpu_free);
        } else if (strstr(curSymName(), "xpu_current_device")) {
            LOG(WARN) << info.libName << '\t' << "xpu_current_device";
            if (!xpu_current_device_fn_ptr) {
                xpu_current_device_fn_ptr = (XpuCurrentDevicePtr)info.oldFuncPtr;
            }
            return info.oldFuncPtr;
        }
        CHECK(0, "error name");
        return nullptr;
    }

    void onSuccess() {}
};

}

extern "C" {

void dh_initialize() {
    LOG(INFO) << "initialize";
    hook::HookInstaller hookInstaller = trace::getHookInstaller();
    hook::install_hook(hookInstaller);
    // hook::install_hook();
    static auto install_wrap = std::make_shared<DeviceMallocFreeHook>();
    install_wrap->install();
}

static void* oldFuncAddr = nullptr;

void log_router() {
    LOG(INFO) << __func__ << ":" << oldFuncAddr;
    // sometime will crash
    // trace::BackTraceCollection::CallStackInfo tracer({});
    // tracer.snapshot();
    // LOG(WARN) << tracer;
    longjmp(log_jump_buffer, 1);
}

void __any_mock_func__() {
// Conditional code for x86_64 architecture
#if defined(__x86_64__)
    asm volatile("pop %rbp");
    asm volatile("push %rax");
    asm volatile("push %rdi");
    if (!setjmp(log_jump_buffer)) {
        log_router();
    }
    asm volatile("pop %rdi");
    asm volatile("pop %rax");
    asm volatile("add    $0x8,%rsp");
    asm volatile("jmp *%0" : : "r"(oldFuncAddr));

// Conditional code for aarch64 architecture
#else
    // asm volatile("push {r0}");
    // if (!setjmp(log_jump_buffer)) {
    //     log_router();
    // }
    // asm volatile("pop {r0}");
#endif
}

int builtin_printf(int flag, const char* fmt, va_list argp) {
    constexpr size_t kMax = 1024;
    char buf[kMax] = {"myprintf "};
    snprintf(buf + strlen(buf), kMax - strlen(buf), fmt, argp);
    LOG(INFO) << buf;
    return 0;
}

static std::unordered_map<std::string, void*> gBuiltinFuncs = {
    {"__printf_chk", reinterpret_cast<void*>(&builtin_printf)},
};

void dh_internal_install_hook(const char* srcLib, const char* targetLib,
                              const char* symbolName, const char* hookerLibPath,
                              const char* hookerSymbolName) {
    LOG(INFO) << "initialize srcLib:" << srcLib << " targetLib:" << targetLib
              << " symbolName:" << symbolName;
    auto iter = gBuiltinFuncs.find(symbolName);
    auto hookerAddr = iter == gBuiltinFuncs.end()
                          ? reinterpret_cast<void*>(&__any_mock_func__)
                          : iter->second;
    if (hookerLibPath) {
        auto handle = dlopen(hookerLibPath, RTLD_LAZY);
        CHECK(handle, "can't not dlopen {}", hookerLibPath);
        hookerAddr =
            dlsym(handle, hookerSymbolName ? hookerSymbolName : symbolName);
    }
    CHECK(hookerAddr, "hookerAddr can't be empty!");
    hook::HookInstaller hookInstaller =
        trace::getHookInstaller(trace::HookerInfo{.srcLib = srcLib,
                                                  .targeLib = targetLib,
                                                  .symbolName = symbolName,
                                                  .newFuncPtr = hookerAddr});
    hookInstaller.onSuccess = [&]() {
        oldFuncAddr =
            trace::CudaInfoCollection::instance().getSymbolAddr(symbolName);
        LOG(INFO) << __func__ << ":" << oldFuncAddr;
    };
    hook::install_hook(hookInstaller);
}
}

#endif
