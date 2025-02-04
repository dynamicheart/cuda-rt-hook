#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trace {

class BackTraceCollection {
   public:
    class CallStackInfo {
       public:
        static constexpr size_t kMaxStackDeep = 256;

        explicit CallStackInfo(
            const std::function<const void*(const std::string&)>& getBaseAddr)
            : getBaseAddr_(getBaseAddr) {
            backtrace_addrs_.reserve(kMaxStackDeep);
            backtrace_.reserve(kMaxStackDeep);
            // test_feed_and_parse();
        }

        bool snapshot();

        bool parse();

        void test_feed_and_parse();

        friend std::ostream& operator<<(std::ostream&, const CallStackInfo&);

       private:
        // every call function's address, we need not this, just check
        std::vector<const void*> backtrace_addrs_;
        std::vector<std::string> backtrace_;
        std::function<const void*(const std::string&)> getBaseAddr_;
    };

    static BackTraceCollection& instance();

    void collect_backtrace(const void* func_ptr);
    void dump();

    void setBaseAddr(const char* libName, const void* addr) {
        base_addrs_.emplace(libName, addr);
    }

    const void* getBaseAddr(const std::string& name);

    void parse_link_map();

    ~BackTraceCollection() { dump(); }

   private:
    std::vector<std::tuple<CallStackInfo, size_t>> backtraces_;
    std::unordered_map<const void*, size_t> cached_map_;
    std::unordered_map<std::string, const void*> base_addrs_;

    // range-range permission xxx xxx xxx libname
    std::vector<std::string> link_maps_;
};

}  // namespace trace
