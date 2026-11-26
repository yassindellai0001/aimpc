#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <mutex>
#include <chrono>
#include <thread>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <climits>
#include <vector>
#include <utility>
#include <windows.h>

enum class LogLevel { Info, Warning, Error, Fatal };

class Logger {
public:
    static constexpr std::uintmax_t kMaxLogSize        = 10ull * 1024 * 1024;
    static constexpr int            kMaxBackupCount    = 3;
    static constexpr std::uintmax_t kSizeCheckInterval = 256ull * 1024;

    // Leaked singleton — never destructed → no static-destruction UAF
    static Logger& Get() {
        static Logger* inst = new Logger();
        return *inst;
    }

    void Shutdown() {
        std::lock_guard<std::mutex> lock(mtx);
        if (m_closed) return;
        m_closed = true;
        if (file.is_open()) {
            try { file.flush(); } catch (...) {}
            try { file.close(); } catch (...) {}
        }
        m_fileInitSucceeded = false;
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void Init(const std::filesystem::path& filePath) {
        std::lock_guard<std::mutex> lock(mtx);

        if (!m_consoleInitAttempted) {
            m_consoleInitAttempted = true;
            InitializeConsole();
        }

        if (filePath.empty()) return;

        if (m_currentPath != filePath) {
            if (file.is_open()) {
                try { file.flush(); } catch (...) {}
                try { file.close(); } catch (...) {}
            }
            m_fileInitSucceeded = false;
            m_currentPath  = filePath;
            m_logPath      = filePath;
        } else if (m_fileInitSucceeded && file.is_open()) {
            return;
        }

        m_fileInitAttempted = true;
        m_closed            = false;
        m_fileInitSucceeded = InitializeFile();
    }

    void Log(LogLevel level, const std::string& component,
             const std::string& msg, long hr = 0) {
        try {
            std::lock_guard<std::mutex> lock(mtx);

            if (m_closed) {
                if (level >= LogLevel::Error) {
                    std::wstring w = Utf8ToWide(
                        "[" + component + "] " + msg + "\n");
                    if (!w.empty()) {
                        try { OutputDebugStringW(w.c_str()); } catch (...) {}
                    }
                }
                return;
            }

            const std::string& logStr = FormatLog(level, component, msg, hr);

            bool fileWrote = false;
            if (file.is_open()) fileWrote = TryWriteToFile(logStr, level);

            bool consoleWrote = false;
            if (m_consoleAvailable && level >= LogLevel::Warning)
                consoleWrote = TryWriteToConsole(logStr);

            if (level >= LogLevel::Error) WriteToDebug(logStr);

            if (level >= LogLevel::Warning && !fileWrote && !consoleWrote) {
                try { std::cerr << logStr; } catch (...) {}
            }
        } catch (...) {}
    }

    bool IsFileOpen()         const { std::lock_guard<std::mutex> l(mtx); return file.is_open(); }
    bool IsConsoleAvailable() const { std::lock_guard<std::mutex> l(mtx); return m_consoleAvailable; }
    bool IsClosed()           const { std::lock_guard<std::mutex> l(mtx); return m_closed; }

    void Flush() {
        std::lock_guard<std::mutex> lock(mtx);
        if (file.is_open()) { try { file.flush(); } catch (...) {} }
    }

    static void Info (const std::string& c, const std::string& m)              { Get().Log(LogLevel::Info,    c, m); }
    static void Warn (const std::string& c, const std::string& m)              { Get().Log(LogLevel::Warning, c, m); }
    static void Error(const std::string& c, const std::string& m, long hr = 0) { Get().Log(LogLevel::Error,   c, m, hr); }
    static void Fatal(const std::string& c, const std::string& m, long hr = 0) { Get().Log(LogLevel::Fatal,   c, m, hr); }

private:
    Logger() = default;

    bool InitializeFile() {
        namespace fs = std::filesystem;
        if (m_logPath.empty()) return false;

        try {
            if (m_logPath.has_parent_path() && !m_logPath.parent_path().empty()) {
                std::error_code ec;
                fs::create_directories(m_logPath.parent_path(), ec);
                if (ec) {
                    OutputDebugStringA("[Logger] Cannot create log dir\n");
                    return false;
                }
            }
        } catch (...) { return false; }

        bool needRotate = false;
        {
            std::error_code ec;
            if (fs::exists(m_logPath, ec) && !ec &&
                fs::is_regular_file(m_logPath, ec) && !ec) {
                auto size = fs::file_size(m_logPath, ec);
                if (!ec && size >= kMaxLogSize) needRotate = true;
            }
        }

        if (needRotate) {
            if (!RotateBackups()) {
                OutputDebugStringA(
                    "[Logger] Rotation failed at init — appending\n");
                return OpenFileForWriting(/*append=*/true);
            }
        }

        return OpenFileForWriting(/*append=*/false);
    }

    bool RotateBackups() {
        namespace fs = std::filesystem;
        std::error_code ec;

        std::vector<std::pair<fs::path, fs::path>> moves;

        auto tryMove = [&](const fs::path& src, const fs::path& dst) -> bool {
            fs::rename(src, dst, ec);
            if (ec) return false;
            moves.emplace_back(src, dst);
            return true;
        };

        auto rollback = [&]() {
            for (auto it = moves.rbegin(); it != moves.rend(); ++it) {
                std::error_code rec;
                fs::rename(it->second, it->first, rec);
            }
            moves.clear();
        };

        fs::path oldest = m_logPath;
        oldest += "." + std::to_string(kMaxBackupCount);
        fs::path oldestPending = oldest;
        oldestPending += ".pending";
        fs::remove(oldestPending, ec);

        bool oldestExisted = fs::exists(oldest, ec) && !ec;
        if (oldestExisted) {
            fs::rename(oldest, oldestPending, ec);
            if (ec) return false;
        }

        for (int i = kMaxBackupCount - 1; i >= 1; --i) {
            fs::path src = m_logPath; src += "." + std::to_string(i);
            fs::path dst = m_logPath; dst += "." + std::to_string(i + 1);
            if (fs::exists(src, ec) && !ec) {
                if (!tryMove(src, dst)) {
                    rollback();
                    if (oldestExisted) {
                        std::error_code rc;
                        fs::rename(oldestPending, oldest, rc);
                    }
                    return false;
                }
            }
        }

        fs::path dst = m_logPath; dst += ".1";
        if (!tryMove(m_logPath, dst)) {
            rollback();
            if (oldestExisted) {
                std::error_code rc;
                fs::rename(oldestPending, oldest, rc);
            }
            return false;
        }

        if (oldestExisted) fs::remove(oldestPending, ec);
        return true;
    }

    bool OpenFileForWriting(bool append) {
        file.clear();
        auto mode = std::ios::out | (append ? std::ios::app : std::ios::trunc);
        file.open(m_logPath, mode);
        if (!file.is_open()) return false;
        if (!append) {
            file << "\xEF\xBB\xBF";
            file.flush();
            if (file.fail()) { file.close(); return false; }
        }
        m_bytesSinceSizeCheck = 0;
        return true;
    }

    void InitializeConsole() {
        HWND consoleWnd = GetConsoleWindow();
        bool newlyAllocated = false;

        if (consoleWnd != nullptr) {
            m_consoleOwned = false;
        } else {
            if (!AllocConsole()) { m_consoleAvailable = false; return; }
            m_consoleOwned = true;
            newlyAllocated = true;
        }

        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (hOut == INVALID_HANDLE_VALUE || hOut == nullptr ||
            !GetConsoleMode(hOut, &mode)) {
            if (newlyAllocated) { FreeConsole(); m_consoleOwned = false; }
            m_consoleAvailable = false;
            return;
        }
        m_consoleAvailable = true;

        if (!m_stdoutRedirected) {
            FILE* dummy = nullptr;
            if (freopen_s(&dummy, "CONOUT$", "w", stdout) == 0) m_stdoutRedirected = true;
        }
        if (!m_stderrRedirected) {
            FILE* dummy = nullptr;
            if (freopen_s(&dummy, "CONOUT$", "w", stderr) == 0) m_stderrRedirected = true;
        }
        if (m_stdoutRedirected || m_stderrRedirected) std::ios::sync_with_stdio(true);
    }

    const std::string& FormatLog(LogLevel level, const std::string& component,
                                 const std::string& msg, long hr) const {
        static thread_local std::string s_tid;
        if (s_tid.empty()) {
            std::ostringstream oss; oss << std::this_thread::get_id();
            s_tid = oss.str();
        }

        char timeBuf[16] = "00:00:00";
        {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            struct tm tm_buf{};
            if (localtime_s(&tm_buf, &t) == 0)
                std::strftime(timeBuf, sizeof(timeBuf), "%H:%M:%S", &tm_buf);
        }

        const char* lvl = "INFO ";
        switch (level) {
            case LogLevel::Info:    lvl = "INFO "; break;
            case LogLevel::Warning: lvl = "WARN "; break;
            case LogLevel::Error:   lvl = "ERROR"; break;
            case LogLevel::Fatal:   lvl = "FATAL"; break;
            default:                lvl = "UNKWN"; break;
        }

        static thread_local std::string s_buf;
        s_buf.clear();
        s_buf.reserve(256);
        s_buf.append("[").append(timeBuf).append("] [")
             .append(lvl).append("] [").append(component)
             .append("] [TID:").append(s_tid).append("]");
        if (hr != 0) {
            char hrBuf[32];
            std::snprintf(hrBuf, sizeof(hrBuf), " [HR:0x%lX]", hr);
            s_buf.append(hrBuf);
        }
        s_buf.append(" ").append(msg).append("\n");
        return s_buf;
    }

    bool TryWriteToFile(const std::string& logStr, LogLevel level) {
        if (!file.is_open()) return false;

        if (m_bytesSinceSizeCheck + logStr.size() >= kSizeCheckInterval) {
            m_bytesSinceSizeCheck = 0;
            if (!CheckAndRotate()) return RetryAfterWriteFailure(logStr, level);
            if (!file.is_open()) return false;
        }

        try { file.write(logStr.data(), static_cast<std::streamsize>(logStr.size())); }
        catch (...) { return RetryAfterWriteFailure(logStr, level); }
        if (file.fail()) return RetryAfterWriteFailure(logStr, level);
        m_bytesSinceSizeCheck += logStr.size();

        if (level >= LogLevel::Error) {
            bool flushOk = true;
            try { file.flush(); flushOk = !file.fail(); } catch (...) { flushOk = false; }
            if (!flushOk) {
                try { file.clear(); file.close(); } catch (...) {}
                if (!m_logPath.empty()) {
                    file.clear();
                    file.open(m_logPath, std::ios::out | std::ios::app);
                }
                return false;
            }
        }
        return true;
    }

    bool CheckAndRotate() {
        if (!file.is_open()) return false;

        namespace fs = std::filesystem;
        std::error_code ec;
        auto size = fs::file_size(m_logPath, ec);
        if (ec || size < kMaxLogSize) return true;

        bool flushOk = true;
        try { file.flush(); flushOk = !file.fail(); } catch (...) { flushOk = false; }
        try { file.close(); } catch (...) {}

        if (!flushOk) return OpenFileForWriting(/*append=*/true);

        bool rotated = RotateBackups();
        return OpenFileForWriting(/*append=*/!rotated);
    }

    bool RetryAfterWriteFailure(const std::string& logStr, LogLevel level) {
        try { file.clear(); file.flush(); } catch (...) {}
        try { file.close(); } catch (...) {}
        if (m_logPath.empty()) return false;

        file.clear();
        file.open(m_logPath, std::ios::out | std::ios::app);
        if (!file.is_open()) return false;

        bool ok = false;
        try {
            file.write(logStr.data(), static_cast<std::streamsize>(logStr.size()));
            if (file.fail()) ok = false;
            else if (level >= LogLevel::Error) { file.flush(); ok = !file.fail(); }
            else ok = true;
        } catch (...) { ok = false; }
        if (!ok) { try { file.close(); } catch (...) {} return false; }
        m_bytesSinceSizeCheck = 0;
        return true;
    }

    bool TryWriteToConsole(const std::string& logStr) {
        if (!m_consoleAvailable) return false;
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hStdout == INVALID_HANDLE_VALUE || hStdout == nullptr) return false;
        DWORD mode = 0;
        if (!GetConsoleMode(hStdout, &mode)) return false;
        std::wstring wStr = Utf8ToWide(logStr);
        if (wStr.empty()) return false;
        DWORD written = 0;
        if (!WriteConsoleW(hStdout, wStr.c_str(), (DWORD)wStr.size(), &written, nullptr))
            return false;
        return written == (DWORD)wStr.size();
    }

    void WriteToDebug(const std::string& logStr) {
        try {
            std::wstring wStr = Utf8ToWide(logStr);
            if (!wStr.empty()) OutputDebugStringW(wStr.c_str());
        } catch (...) {}
    }

    static std::wstring Utf8ToWide(std::string_view s) {
        if (s.empty()) return std::wstring();
        if (s.size() > static_cast<size_t>(INT_MAX) / 2)
            s = s.substr(0, static_cast<size_t>(INT_MAX) / 2);
        const int len = static_cast<int>(s.size());

        int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                         s.data(), len, nullptr, 0);
        if (needed > 0) {
            std::wstring out(static_cast<size_t>(needed), L'\0');
            int got = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                          s.data(), len, out.data(), needed);
            if (got > 0) { out.resize(static_cast<size_t>(got)); return out; }
        }
        needed = MultiByteToWideChar(CP_UTF8, 0, s.data(), len, nullptr, 0);
        if (needed > 0) {
            std::wstring out(static_cast<size_t>(needed), L'\0');
            int got = MultiByteToWideChar(CP_UTF8, 0, s.data(), len, out.data(), needed);
            if (got > 0) { out.resize(static_cast<size_t>(got)); return out; }
        }
        return L"<invalid-utf8>";
    }

    mutable std::mutex      mtx;
    std::ofstream           file;
    std::filesystem::path   m_logPath;
    std::filesystem::path   m_currentPath;

    bool m_closed               = false;
    bool m_fileInitAttempted    = false;
    bool m_fileInitSucceeded    = false;
    bool m_consoleInitAttempted = false;
    bool m_consoleOwned         = false;
    bool m_consoleAvailable     = false;
    bool m_stdoutRedirected     = false;
    bool m_stderrRedirected     = false;

    std::uintmax_t m_bytesSinceSizeCheck = 0;
};
