#include "xpu_mock.h"

#include <Python.h>
#include <dlfcn.h>  // dladdr
#include <execinfo.h>
#include <frameobject.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "backtrace.h"
#include "hook.h"
#include "logger/logger.h"
#include "support.h"

namespace {

void cppBacktrace() {
    constexpr int kMaxStackDeep = 512;
    void* call_stack[kMaxStackDeep] = {0};
    char** symbols = nullptr;
    int num = backtrace(call_stack, kMaxStackDeep);
    CHECK(num > 0, "Expect frams num {} > 0!", num);
    CHECK(num <= kMaxStackDeep, "Expect frams num {} <= 512!", num);
    symbols = backtrace_symbols(call_stack, num);
    if (symbols == nullptr) {
        return;
    }

    LOG(WARN) << "stack_deep=" << num;
    Dl_info info;
    for (int j = 0; j < num; j++) {
        if (dladdr(call_stack[j], &info) && info.dli_sname) {
            auto demangled = __support__demangle(info.dli_sname);
            std::string path(info.dli_fname);
            LOG(WARN) << "    frame " << j << path << ":" << demangled;
        } else {
            // filtering useless print
            // LOG(WARN) << "    frame " << j << call_stack[j];
        }
    }
    free(symbols);
}

void pythonBacktrace() {
    // Acquire the Global Interpreter Lock (GIL) before calling Python C API
    // functions from non-Python threads.
    PyGILState_STATE gstate = PyGILState_Ensure();

    LOG(WARN) << "Python stack trace:";
    // https://stackoverflow.com/questions/1796510/accessing-a-python-traceback-from-the-c-api
    PyThreadState* tstate = PyThreadState_GET();
    if (NULL != tstate && NULL != tstate->frame) {
        PyFrameObject* frame = tstate->frame;

        while (NULL != frame) {
            // int line = frame->f_lineno;
            /*
            frame->f_lineno will not always return the correct line number
            you need to call PyCode_Addr2Line().
            */
            int line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
            const char* filename = PyUnicode_AsUTF8(frame->f_code->co_filename);
            const char* funcname = PyUnicode_AsUTF8(frame->f_code->co_name);
            LOG(WARN) << "    " << filename << "(" << line << "): " << funcname;
            frame = frame->f_back;
        }
    }
    PyGILState_Release(gstate);
}

class Timer {
   public:
    Timer(std::chrono::time_point<std::chrono::steady_clock> tp =
              std::chrono::steady_clock::now())
        : _start_time(tp) {}

    template <typename T>
    bool Timeout(double count) const {
        return Passed<T>() >= count;
    }

    double Passed() const { return Passed<std::chrono::duration<double>>(); }

    double PassedSec() const { return Passed<std::chrono::seconds>(); }

    double PassedMicro() const { return Passed<std::chrono::microseconds>(); }

    double PassedNano() const { return Passed<std::chrono::nanoseconds>(); }

    template <typename T>
    double Passed() const {
        return Passed<T>(std::chrono::steady_clock::now());
    }

    template <typename T>
    double Passed(std::chrono::time_point<std::chrono::steady_clock> tp) const {
        const auto elapsed = std::chrono::duration_cast<T>(tp - _start_time);
        return elapsed.count();
    }

    uint64_t TimePointMicro() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   _start_time.time_since_epoch())
            .count();
    }

    void Reset() { _start_time = std::chrono::steady_clock::now(); }

   private:
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
};

#define TIMING_CALL(dur, r, func) \
    {                             \
        Timer t;                  \
        r = func;                 \
        dur = t.Passed();         \
    }

class XpuRuntimeApiHook;

class XpuRuntimeWrapApi {
   public:
    static constexpr int kMaxXpuDeviceNum = 8;

    static XpuRuntimeWrapApi& instance();
    XpuRuntimeWrapApi();
    static int xpuMalloc(void** pDevPtr, uint64_t size, int kind);
    static int xpuFree(void* devPtr);
    static int xpuWait(void* devStream);
    static int xpuMemcpy(void* dst, const void* src, uint64_t size, int kind);
    static int xpuLaunchArgumentSet(const void* arg, uint64_t size,
                                    uint64_t offset);
    static int xpuLaunchAsync(void* func);
    static int xpuLaunchConfig(int nclusters, int ncores, void* stream);
    static int xpuSetDevice(int devId);
    static int xpuCurrentDevice(int* devId);

   private:
    std::function<int(void**, uint64_t, int)> raw_xpu_malloc_;
    std::function<int(void*)> raw_xpu_free_;
    std::function<int(void*)> raw_xpu_wait_;
    std::function<int(void*, const void*, uint64_t, int)> raw_xpu_memcpy_;
    std::function<int(const void*, uint64_t, uint64_t)>
        raw_xpu_launch_argument_set_;
    std::function<int(void*)> raw_xpu_launch_async_;
    std::function<int(int, int, void*)> raw_xpu_launch_config_;
    std::function<int(int)> raw_xpu_set_device_;
    std::function<int(int*)> raw_xpu_current_device_;

    enum class XpuMemKind { GLOBAL_MEMORY = 0, L3_MEMORY };

    struct XpuDataPtr {
        void* data_ptr;
        uint64_t size;
        XpuMemKind kind;
    };

    std::mutex memory_api_mutex_;
    std::vector<std::map<void*, XpuDataPtr>> allocated_ptr_map_;
    std::vector<uint64_t> allocated_gm_size_;
    std::vector<uint64_t> allocated_l3_size_;
    std::vector<uint64_t> peak_gm_size_;
    std::vector<uint64_t> peak_l3_size_;

    friend class XpuRuntimeApiHook;
};

XpuRuntimeWrapApi& XpuRuntimeWrapApi::instance() {
    static XpuRuntimeWrapApi instance;
    return instance;
}

XpuRuntimeWrapApi::XpuRuntimeWrapApi()
    : allocated_ptr_map_(kMaxXpuDeviceNum),
      allocated_gm_size_(kMaxXpuDeviceNum, 0),
      allocated_l3_size_(kMaxXpuDeviceNum, 0),
      peak_gm_size_(kMaxXpuDeviceNum, 0),
      peak_l3_size_(kMaxXpuDeviceNum, 0) {}

int XpuRuntimeWrapApi::xpuMalloc(void** pDevPtr, uint64_t size, int kind) {
    double dur = 0;
    int r = 0;
    int devId = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_malloc_, "xpu_free not binded");

    // make malloc/free sequential to obtain a trusted memory usage footprint
    std::lock_guard<std::mutex> lock(
        XpuRuntimeWrapApi::instance().memory_api_mutex_);

    r = XpuRuntimeWrapApi::instance().raw_xpu_current_device_(&devId);
    if (r != 0) {
        return r;
    }
    CHECK_LT(devId, kMaxXpuDeviceNum,
             "devId({}) must less than kMaxXpuDeviceNum({})", devId,
             kMaxXpuDeviceNum);

    TIMING_CALL(
        dur, r,
        XpuRuntimeWrapApi::instance().raw_xpu_malloc_(pDevPtr, size, kind));
    if (r != 0) {
        LOG(WARN) << "[XpuRuntimeWrapApi xpuMalloc][failed] "
                  << "devId=" << devId << ","
                  << "size=" << size << ","
                  << "kind=" << kind << ","
                  << "gm_allocated="
                  << XpuRuntimeWrapApi::instance().allocated_gm_size_[devId]
                  << ","
                  << "gm_peak="
                  << XpuRuntimeWrapApi::instance().peak_gm_size_[devId] << ","
                  << "duration=" << dur;
        return r;
    }

    if (kind == (int)XpuMemKind::GLOBAL_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_gm_size_[devId] += size;
        XpuRuntimeWrapApi::instance().peak_gm_size_[devId] =
            std::max(XpuRuntimeWrapApi::instance().peak_gm_size_[devId],
                     XpuRuntimeWrapApi::instance().allocated_gm_size_[devId]);
    } else if (kind == (int)XpuMemKind::L3_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_l3_size_[devId] += size;
        XpuRuntimeWrapApi::instance().peak_l3_size_[devId] =
            std::max(XpuRuntimeWrapApi::instance().peak_l3_size_[devId],
                     XpuRuntimeWrapApi::instance().allocated_l3_size_[devId]);
    }

    XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId][*pDevPtr] = {
        *pDevPtr, size, (XpuMemKind)kind};

    LOG(WARN) << "[XpuRuntimeWrapApi xpuMalloc][success] "
              << "devId=" << devId << ","
              << "size=" << size << ","
              << "kind=" << kind << ","
              << "gm_allocated="
              << XpuRuntimeWrapApi::instance().allocated_gm_size_[devId] << ","
              << "gm_peak="
              << XpuRuntimeWrapApi::instance().peak_gm_size_[devId] << ","
              << "duration=" << dur;
    if (std::getenv("BACKTRACE_XPU_MALLOC")) {
        cppBacktrace();
        pythonBacktrace();
    }
    return r;
}

int XpuRuntimeWrapApi::xpuFree(void* devPtr) {
    double dur = 0;
    int r = 0;
    int devId = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_free_, "xpu_free not binded");

    // make malloc/free sequential to obtain a trusted memory usage footprint
    std::lock_guard<std::mutex> lock(
        XpuRuntimeWrapApi::instance().memory_api_mutex_);

    r = XpuRuntimeWrapApi::instance().raw_xpu_current_device_(&devId);
    if (r != 0) {
        return r;
    }
    CHECK_LT(devId, kMaxXpuDeviceNum,
             "devId({}) must less than kMaxXpuDeviceNum({})", devId,
             kMaxXpuDeviceNum);

    TIMING_CALL(dur, r, XpuRuntimeWrapApi::instance().raw_xpu_free_(devPtr));
    LOG(WARN) << "[XpuRuntimeWrapApi xpuFree]: duration=" << dur;
    if (r != 0) {
        return r;
    }

    auto it =
        XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId].find(devPtr);
    if (it == XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId].end()) {
        return r;
    }

    XpuDataPtr dataPtr = it->second;

    if (dataPtr.kind == XpuMemKind::GLOBAL_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_gm_size_[devId] -= dataPtr.size;
    } else if (dataPtr.kind == XpuMemKind::L3_MEMORY) {
        XpuRuntimeWrapApi::instance().allocated_l3_size_[devId] -= dataPtr.size;
    }

    XpuRuntimeWrapApi::instance().allocated_ptr_map_[devId].erase(it);
    return r;
}

int XpuRuntimeWrapApi::xpuWait(void* devStream) {
    double dur = 0;
    int r = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_wait_, "xpu_wait not binded");
    TIMING_CALL(dur, r, XpuRuntimeWrapApi::instance().raw_xpu_wait_(devStream));

    if (!std::getenv("MOCK_XPU_WAIT")) {
        LOG(WARN) << "[XpuRuntimeWrapApi xpuWait]:"
                  << "duration=" << dur;
        return r;
    }

    cppBacktrace();
    return r;
}

int XpuRuntimeWrapApi::xpuMemcpy(void* dst, const void* src, uint64_t size,
                                 int kind) {
    double dur = 0;
    int r = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_memcpy_,
          "xpu_memcpy not binded");
    TIMING_CALL(
        dur, r,
        XpuRuntimeWrapApi::instance().raw_xpu_memcpy_(dst, src, size, kind));

    if (!std::getenv("MOCK_XPU_MEMCPY")) {
        LOG(WARN) << "[XpuRuntimeWrapApi xpuMemcpy]"
                  << "duration=" << dur;
        return r;
    }

    pythonBacktrace();
    return r;
}

int XpuRuntimeWrapApi::xpuLaunchArgumentSet(const void* arg, uint64_t size,
                                            uint64_t offset) {
    double dur = 0;
    int r = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_launch_argument_set_,
          "xpu_launch_argument_set not binded");
    TIMING_CALL(dur, r,
                XpuRuntimeWrapApi::instance().raw_xpu_launch_argument_set_(
                    arg, size, offset));

    LOG(WARN) << "[XpuRuntimeWrapApi xpuLaunchArgumentSet]"
              << "duration=" << dur;
    return r;
}

int XpuRuntimeWrapApi::xpuLaunchAsync(void* func) {
    double dur = 0;
    int r = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_launch_async_,
          "xpu_launch_async not binded");
    TIMING_CALL(dur, r,
                XpuRuntimeWrapApi::instance().raw_xpu_launch_async_(func));

    LOG(WARN) << "[XpuRuntimeWrapApi xpuLaunchAsync]"
              << "duration=" << dur;
    return r;
}

int XpuRuntimeWrapApi::xpuLaunchConfig(int nclusters, int ncores,
                                       void* stream) {
    double dur = 0;
    int r = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_launch_config_,
          "xpu_launch_config not binded");
    TIMING_CALL(dur, r,
                XpuRuntimeWrapApi::instance().raw_xpu_launch_config_(
                    nclusters, ncores, stream));

    LOG(WARN) << "[XpuRuntimeWrapApi xpuLaunchConfig]"
              << "duration=" << dur;
    return r;
}

int XpuRuntimeWrapApi::xpuSetDevice(int devId) {
    double dur = 0;
    int r = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_set_device_,
          "xpu_set_device not binded");
    TIMING_CALL(dur, r,
                XpuRuntimeWrapApi::instance().raw_xpu_set_device_(devId));

    LOG(WARN) << "[XpuRuntimeWrapApi xpuSetDevice]"
              << "devId=" << devId << ","
              << "duration=" << dur;

    if (!std::getenv("MOCK_XPU_SET_DEVICE")) {
        return r;
    }

    cppBacktrace();
    pythonBacktrace();
    return r;
}

int XpuRuntimeWrapApi::xpuCurrentDevice(int* devId) {
    double dur = 0;
    int r = 0;

    CHECK(XpuRuntimeWrapApi::instance().raw_xpu_current_device_,
          "xpu_current_device not binded");
    TIMING_CALL(dur, r,
                XpuRuntimeWrapApi::instance().raw_xpu_current_device_(devId));

    LOG(WARN) << "[XpuRuntimeWrapApi xpuCurrentDevice]"
              << "devId=" << *devId << ","
              << "duration=" << dur;

    if (!std::getenv("MOCK_XPU_CURRENT_DEVICE")) {
        return r;
    }

    cppBacktrace();
    pythonBacktrace();
    return r;
}

struct XpuRuntimeApiHook : public hook::HookInstallerWrap<XpuRuntimeApiHook> {
    bool targetLib(const char* name) {
        return !strstr(name, "libxpurt.so.1") && !strstr(name, "libxpurt.so");
    }

    bool targetSym(const char* name) {
        return strstr(name, "xpu_malloc") || strstr(name, "xpu_free") ||
               strstr(name, "xpu_wait") || strstr(name, "xpu_memcpy") ||
               strstr(name, "xpu_launch_argument_set") ||
               strstr(name, "xpu_launch_async") ||
               strstr(name, "xpu_launch_config") ||
               strstr(name, "xpu_set_device") ||
               strstr(name, "xpu_current_device");
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        if (strstr(curSymName(), "xpu_malloc")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_malloc]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_malloc_) {
                XpuRuntimeWrapApi::instance().raw_xpu_malloc_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuMalloc);
        } else if (strstr(curSymName(), "xpu_free")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_free]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_free_) {
                XpuRuntimeWrapApi::instance().raw_xpu_free_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuFree);
        } else if (strstr(curSymName(), "xpu_wait")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_wait]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_wait_) {
                XpuRuntimeWrapApi::instance().raw_xpu_wait_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuWait);
        } else if (strstr(curSymName(), "xpu_memcpy")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_memcpy]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_memcpy_) {
                XpuRuntimeWrapApi::instance().raw_xpu_memcpy_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3, std::placeholders::_4);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuMemcpy);
        } else if (strstr(curSymName(), "xpu_launch_argument_set")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_launch_argument_set]:"
                      << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_launch_argument_set_) {
                XpuRuntimeWrapApi::instance().raw_xpu_launch_argument_set_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3);
            }
            return reinterpret_cast<void*>(
                &XpuRuntimeWrapApi::xpuLaunchArgumentSet);
        } else if (strstr(curSymName(), "xpu_launch_async")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_launch_async]:"
                      << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_launch_async_) {
                XpuRuntimeWrapApi::instance().raw_xpu_launch_async_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuLaunchAsync);
        } else if (strstr(curSymName(), "xpu_launch_config")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_launch_config]:"
                      << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_launch_config_) {
                XpuRuntimeWrapApi::instance().raw_xpu_launch_config_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuLaunchConfig);
        } else if (strstr(curSymName(), "xpu_set_device")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_set_device]:" << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_set_device_) {
                XpuRuntimeWrapApi::instance().raw_xpu_set_device_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(&XpuRuntimeWrapApi::xpuSetDevice);
        } else if (strstr(curSymName(), "xpu_current_device")) {
            LOG(WARN) << "[XpuRuntimeApiHook][xpu_current_device]:"
                      << info.libName;
            if (!XpuRuntimeWrapApi::instance().raw_xpu_current_device_) {
                XpuRuntimeWrapApi::instance().raw_xpu_current_device_ =
                    std::bind(reinterpret_cast<int((*)(...))>(info.oldFuncPtr),
                              std::placeholders::_1);
            }
            return reinterpret_cast<void*>(
                &XpuRuntimeWrapApi::xpuCurrentDevice);
        }
        CHECK(0, "capture wrong function: {}", curSymName());
        return nullptr;
    }

    void onSuccess() {}
};

}  // namespace

extern "C" {

void xpu_dh_initialize() {
    static auto install_wrap = std::make_shared<XpuRuntimeApiHook>();
    install_wrap->install();
}
}