#include "logger.h"

#include <atomic>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>

namespace logger {

class LogConsumer {
   public:
    LogConsumer() : th_(&LogConsumer::print, this), exit_(false) {}

    void pushLog(std::stringstream& ss) {
        auto str = ss.str();
        {
            std::lock_guard<std::mutex> guard{mtx_};
            buf_.push_back(std::move(str));
        }

        ss.clear();
        ss.str("");
    }

    void print() {
        while (!exit_) {
            if (buf_.empty()) {
                std::this_thread::yield();
                continue;
            } else {
                mtx_.lock();
                std::string str = std::move(buf_.front());
                buf_.pop_front();
                mtx_.unlock();

                std::cout << str << std::endl;
            }
            std::this_thread::yield();
        }
    }

    ~LogConsumer() {
        exit_.store(true);
        th_.join();
    }

   private:
    std::mutex mtx_;
    std::deque<std::string> buf_;
    std::thread th_;
    std::atomic<bool> exit_;
};

size_t StringLiteralBase::MN = 0;

static LogConsumer gLogConsumer;

LogStream& LogStream::instance() {
    static thread_local LogStream __instance;
    return __instance;
}

void LogStream::flush() {
    gLogConsumer.pushLog(ss_);
}

}  // namespace logger