#include <windows.h>
#include <mmsystem.h>
#include <shcore.h>
#include <tlhelp32.h>

#include <opencv2/core.hpp>

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <cstdint>
#include <climits>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <utility>
#include <string>
#include <system_error>
#include <exception>
#include <optional>

#include "Logger.h"
#include "Config.h"
#include "DXGICapture.h"
#include "YoloDetector.h"
#include "Overlay.h"

#pragma comment(lib, "shcore.lib")
#pragma comment(lib, "winmm.lib")

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::microseconds;
using std::chrono::duration_cast;
namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════
//  log_safe — noexcept logging for cleanup paths
// ═══════════════════════════════════════════════════════════════
namespace log_safe {
    inline void info(const char* c, const char* m) noexcept {
        try { Logger::Info(c ? c : "?", m ? m : "?"); } catch (...) {}
    }
    inline void warn(const char* c, const char* m) noexcept {
        try { Logger::Warn(c ? c : "?", m ? m : "?"); } catch (...) {}
    }
    inline void error(const char* c, const char* m, long hr = 0) noexcept {
        try { Logger::Error(c ? c : "?", m ? m : "?", hr); } catch (...) {}
    }
    inline void fatal(const char* c, const char* m, long hr = 0) noexcept {
        try { Logger::Fatal(c ? c : "?", m ? m : "?", hr); } catch (...) {
            OutputDebugStringA("[Logger failed in fatal path]\n");
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  TimerResolutionGuard
// ═══════════════════════════════════════════════════════════════
struct TimerResolutionGuard {
    bool m_active = false;
    TimerResolutionGuard() noexcept {
        if (timeBeginPeriod(1) == TIMERR_NOERROR) m_active = true;
    }
    ~TimerResolutionGuard() noexcept { if (m_active) timeEndPeriod(1); }
    TimerResolutionGuard(const TimerResolutionGuard&) = delete;
    TimerResolutionGuard& operator=(const TimerResolutionGuard&) = delete;
};

// ═══════════════════════════════════════════════════════════════
//  ScopeGuard
// ═══════════════════════════════════════════════════════════════
template<typename F>
struct ScopeGuard {
    F fn;
    bool active = true;

    explicit ScopeGuard(F f) : fn(std::move(f)) {}
    ~ScopeGuard() noexcept { if (active) { try { fn(); } catch (...) {} } }
    void dismiss() noexcept { active = false; }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&&) = delete;
    ScopeGuard& operator=(ScopeGuard&&) = delete;
};
template<typename F> ScopeGuard(F) -> ScopeGuard<F>;

// ═══════════════════════════════════════════════════════════════
//  TempFileGuard — arms only after file created
// ═══════════════════════════════════════════════════════════════
struct TempFileGuard {
    fs::path path;
    bool active = false;

    explicit TempFileGuard(fs::path p) : path(std::move(p)), active(true) {}
    TempFileGuard(TempFileGuard&& o) noexcept
        : path(std::move(o.path)), active(o.active) {
        o.active = false;
    }
    ~TempFileGuard() noexcept {
        if (!active) return;
        try {
            std::error_code ec;
            fs::remove(path, ec);
        } catch (...) {}
    }
    TempFileGuard(const TempFileGuard&) = delete;
    TempFileGuard& operator=(const TempFileGuard&) = delete;
    TempFileGuard& operator=(TempFileGuard&&) = delete;
};

// ═══════════════════════════════════════════════════════════════
//  SnapshotHandle
// ═══════════════════════════════════════════════════════════════
struct SnapshotHandle {
    HANDLE h = INVALID_HANDLE_VALUE;
    SnapshotHandle() noexcept = default;
    ~SnapshotHandle() noexcept { reset(); }
    SnapshotHandle(SnapshotHandle&& o) noexcept : h(o.h) { o.h = INVALID_HANDLE_VALUE; }
    SnapshotHandle& operator=(SnapshotHandle&& o) noexcept {
        if (this != &o) { reset(); h = o.h; o.h = INVALID_HANDLE_VALUE; }
        return *this;
    }
    SnapshotHandle(const SnapshotHandle&) = delete;
    SnapshotHandle& operator=(const SnapshotHandle&) = delete;
    bool valid() const noexcept { return h != INVALID_HANDLE_VALUE && h != nullptr; }
    void reset() noexcept {
        if (h == INVALID_HANDLE_VALUE || h == nullptr) return;
        if (!CloseHandle(h)) {
            try { OutputDebugStringA("[SnapshotHandle] CloseHandle failed\n"); }
            catch (...) {}
        }
        h = INVALID_HANDLE_VALUE;
    }
};

// ═══════════════════════════════════════════════════════════════
//  Thread lifecycle
// ═══════════════════════════════════════════════════════════════
static std::atomic<bool>       g_captureExited{true};
static std::atomic<bool>       g_aimExited{true};
static std::mutex              g_threadExitMtx;
static std::condition_variable g_threadExitCv;

static void SafeJoin(std::thread& t, std::atomic<bool>& exitedFlag,
                     const char* name) noexcept {
    if (!t.joinable()) return;

    {
        std::unique_lock<std::mutex> lock(g_threadExitMtx);
        g_threadExitCv.wait_for(lock, std::chrono::seconds(5),
            [&]() { return exitedFlag.load(); });
    }

    if (!exitedFlag.load()) {
        try {
            Logger::Fatal("Cleanup",
                std::string(name) + " did not signal exit in 5s; "
                "join may block until syscall returns.");
        } catch (...) {
            OutputDebugStringA("[Cleanup] thread stuck in syscall\n");
        }
    }

    try {
        t.join();
    } catch (const std::system_error& e) {
        try {
            Logger::Error("Cleanup", std::string(name) + " join threw: " + e.what());
        } catch (...) {}
    } catch (...) {}
}

// ═══════════════════════════════════════════════════════════════
//  HasInteractiveDesktop
// ═══════════════════════════════════════════════════════════════
static bool HasInteractiveDesktop() noexcept {
    HDESK desk = OpenInputDesktop(0, FALSE, DESKTOP_READOBJECTS);
    if (!desk) return false;
    CloseDesktop(desk);
    return true;
}

// ═══════════════════════════════════════════════════════════════
//  Globals
// ═══════════════════════════════════════════════════════════════
static std::atomic<bool> g_running(true);
static std::atomic<bool> g_altPressed(false);
static std::atomic<bool> g_isFiring(false);
static std::atomic<int>  g_headOffsetX(0);
static std::atomic<int>  g_headOffsetY(0);
static std::atomic<int>  g_crosshairMonX(0);
static std::atomic<int>  g_crosshairMonY(0);
static std::mutex g_targetMutex;
static Overlay g_overlay;

static std::atomic<int> g_moveFailCount(0);
static std::atomic<int> g_leftDownFailCount(0);
static std::atomic<int> g_leftUpFailCount(0);

static int  g_oldMouseParams[3] = {0, 0, 0};
static bool g_accelSaved = false;

// ═══════════════════════════════════════════════════════════════
//  Mouse acceleration
// ═══════════════════════════════════════════════════════════════
static void DisableMouseAcceleration() noexcept {
    try {
        if (!SystemParametersInfoW(SPI_GETMOUSE, 0, g_oldMouseParams, 0)) {
            log_safe::warn("Input", "SPI_GETMOUSE failed");
            return;
        }
        g_accelSaved = true;

        if (g_oldMouseParams[2] == 0) {
            log_safe::info("Input", "Mouse acceleration already disabled");
            return;
        }
        int flat[3] = {0, 0, 0};
        if (!SystemParametersInfoW(SPI_SETMOUSE, 0, flat, SPIF_SENDCHANGE)) {
            log_safe::warn("Input", "SPI_SETMOUSE failed");
            return;
        }
        log_safe::info("Input", "Mouse acceleration disabled");
    } catch (...) {}
}

static void RestoreMouseAcceleration() noexcept {
    if (!g_accelSaved) return;
    g_accelSaved = false;
    try {
        SystemParametersInfoW(SPI_SETMOUSE, 0, g_oldMouseParams, SPIF_SENDCHANGE);
    } catch (...) {}
}

// ═══════════════════════════════════════════════════════════════
//  TargetData
// ═══════════════════════════════════════════════════════════════
struct TargetData {
    bool found = false;
    int screenX = 0, screenY = 0;
    float vx = 0.0f, vy = 0.0f;
    float conf = 0.0f;
    cv::Rect boxMonitorLocal;
    Clock::time_point detectionTime;
};
static TargetData g_target;

// ═══════════════════════════════════════════════════════════════
//  UTF-8 → UTF-16 (strict)
// ═══════════════════════════════════════════════════════════════
static std::wstring StringToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    if (str.size() > static_cast<size_t>(INT_MAX)) {
        log_safe::error("Util", "String too long for UTF-8→UTF-16");
        return std::wstring();
    }
    const int len = static_cast<int>(str.size());

    int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                      str.data(), len, nullptr, 0);
    if (needed <= 0) {
        log_safe::warn("Util", "StringToWString: invalid UTF-8");
        return std::wstring();
    }

    std::wstring out(static_cast<size_t>(needed), L'\0');
    int got = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                   str.data(), len, out.data(), needed);
    if (got <= 0) return std::wstring();
    out.resize(static_cast<size_t>(got));
    return out;
}

// ═══════════════════════════════════════════════════════════════
//  FindGameWindow — prefers foreground, then largest
// ═══════════════════════════════════════════════════════════════
static HWND FindGameWindow(const std::string& forcedProcess) {
    std::vector<std::wstring> targets;
    if (!forcedProcess.empty()) {
        std::wstring w = StringToWString(forcedProcess);
        if (w.empty()) return nullptr;
        targets.push_back(std::move(w));
    } else {
        targets = { L"HD-Player.exe", L"LdVBoxHeadless.exe", L"MEmuHeadless.exe",
                    L"Nox.exe", L"AndroidEmulator.exe" };
    }

    SnapshotHandle snap;
    snap.h = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (!snap.valid()) {
        log_safe::warn("Main", "CreateToolhelp32Snapshot failed");
        return nullptr;
    }

    PROCESSENTRY32W pe = { sizeof(PROCESSENTRY32W) };
    DWORD pid = 0;
    if (Process32FirstW(snap.h, &pe)) {
        do {
            for (const auto& t : targets) {
                if (_wcsicmp(pe.szExeFile, t.c_str()) == 0) {
                    pid = pe.th32ProcessID;
                    break;
                }
            }
            if (pid != 0) break;
        } while (Process32NextW(snap.h, &pe));
    }
    if (pid == 0) return nullptr;

    // Prefer foreground window
    HWND foreground = GetForegroundWindow();
    if (foreground) {
        DWORD fgPid = 0;
        GetWindowThreadProcessId(foreground, &fgPid);
        if (fgPid == pid && IsWindowVisible(foreground)) {
            RECT r;
            if (GetWindowRect(foreground, &r) &&
                (r.right - r.left) > 200 && (r.bottom - r.top) > 200) {
                return foreground;
            }
        }
    }

    // Otherwise largest visible window
    struct Data { DWORD pid; HWND hwnd; LONG bestArea; };
    Data data = { pid, nullptr, 0 };

    EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
        auto* d = reinterpret_cast<Data*>(lParam);
        DWORD wpid;
        GetWindowThreadProcessId(hwnd, &wpid);
        if (wpid != d->pid || !IsWindowVisible(hwnd)) return TRUE;

        RECT r;
        if (!GetWindowRect(hwnd, &r)) return TRUE;

        LONG w = r.right - r.left;
        LONG h = r.bottom - r.top;
        if (w <= 200 || h <= 200) return TRUE;

        LONG area = w * h;
        if (area > d->bestArea) {
            d->hwnd = hwnd;
            d->bestArea = area;
        }
        return TRUE;
    }, reinterpret_cast<LPARAM>(&data));

    return data.hwnd;
}

// ═══════════════════════════════════════════════════════════════
//  Input helpers
// ═══════════════════════════════════════════════════════════════
static void MoveMouse(int dx, int dy) {
    INPUT in = {};
    in.type = INPUT_MOUSE;
    in.mi.dx = dx;
    in.mi.dy = dy;
    in.mi.dwFlags = MOUSEEVENTF_MOVE;
    if (SendInput(1, &in, sizeof(INPUT)) != 1) {
        int c = g_moveFailCount.fetch_add(1) + 1;
        if (c == 1 || (c % 200) == 0) {
            log_safe::error("Input", "MoveMouse SendInput failed");
        }
    } else {
        if (g_moveFailCount.exchange(0) > 0)
            log_safe::info("Input", "MoveMouse recovered");
    }
}

static bool SendMouseButton(bool down) {
    INPUT in = {};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = down ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP;
    std::atomic<int>& cnt = down ? g_leftDownFailCount : g_leftUpFailCount;

    if (SendInput(1, &in, sizeof(INPUT)) == 1) {
        if (cnt.exchange(0) > 0)
            log_safe::info("Input", down ? "LEFTDOWN recovered" : "LEFTUP recovered");
        return true;
    }
    int c = cnt.fetch_add(1) + 1;
    if (c == 1 || (c % 200) == 0) {
        log_safe::error("Input", down ? "LEFTDOWN failed" : "LEFTUP failed");
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════
//  CaptureThread
// ═══════════════════════════════════════════════════════════════
static void CaptureThread(DXGICapture* cap, YoloDetector* det, Config* cfg) {
    try {
        cv::Mat frame;
        auto lastTargetTime = Clock::now();
        bool lastLoggedState = false;
        int consecutiveErrors = 0;

        constexpr int kTransientThreshold   = 3;
        constexpr int kSustainedThreshold   = 30;
        constexpr int kCriticalReinitRetries = 10;

        while (g_running.load(std::memory_order_acquire)) {
            try {
                DXGICapture::Status st = DXGICapture::Status::Error;
                try { st = cap->GrabFrame(frame); }
                catch (...) {}

                if (st == DXGICapture::Status::Success) {
                    if (frame.empty() ||
                        frame.type() != CV_8UC3 ||
                        frame.cols <= 0 || frame.rows <= 0) {
                        log_safe::warn("DXGI", "GrabFrame returned invalid frame");
                        consecutiveErrors = 0;
                        continue;
                    }

                    consecutiveErrors = 0;
                    int radius     = cfg->circleRadius;
                    int crosshairX = g_crosshairMonX.load();
                    int crosshairY = g_crosshairMonY.load();

                    int cropSize = radius + cfg->cropMargin;
                    int cropX = crosshairX - cropSize, cropY = crosshairY - cropSize;
                    int cropW = cropSize * 2, cropH = cropSize * 2;

                    if (cropX < 0) { cropW += cropX; cropX = 0; }
                    if (cropY < 0) { cropH += cropY; cropY = 0; }
                    if (cropX + cropW > frame.cols) cropW = frame.cols - cropX;
                    if (cropY + cropH > frame.rows) cropH = frame.rows - cropY;
                    if (cropW <= 0 || cropH <= 0) continue;

                    cv::Mat roi = frame(cv::Rect(cropX, cropY, cropW, cropH));
                    std::vector<Detection> detections;
                    det->Detect(roi, detections);
                    auto now = Clock::now();

                    const int localCrossX = crosshairX - cropX;
                    const int localCrossY = crosshairY - cropY;

                    bool found = false;
                    double minDist = 1e9;
                    Detection best;
                    for (auto& d : detections) {
                        double dist = std::hypot(
                            (d.box.x + d.box.width / 2.0) - localCrossX,
                            (d.box.y + d.box.height / 2.0) - localCrossY);
                        if (dist > radius) continue;
                        if (dist < minDist) { minDist = dist; best = d; found = true; }
                    }

                    {
                        std::lock_guard<std::mutex> lock(g_targetMutex);
                        if (found) {
                            int tx = cropX + best.box.x + best.box.width  / 2;
                            int ty = cropY + best.box.y + best.box.height / 2;
                            auto dt = duration_cast<microseconds>(now - lastTargetTime).count();
                            if (g_target.found && dt > 1000 && dt < 200000) {
                                g_target.vx = g_target.vx * 0.5f +
                                              (float)(tx - g_target.screenX) / dt * 0.5f;
                                g_target.vy = g_target.vy * 0.5f +
                                              (float)(ty - g_target.screenY) / dt * 0.5f;
                            } else {
                                g_target.vx = 0.0f;
                                g_target.vy = 0.0f;
                            }
                            g_target.screenX = tx;
                            g_target.screenY = ty;
                            g_target.boxMonitorLocal = cv::Rect(
                                cropX + best.box.x, cropY + best.box.y,
                                best.box.width, best.box.height);
                            g_target.conf = best.confidence;
                            g_target.detectionTime = now;
                        }
                        g_target.found = found;
                    }

                    if (found) lastTargetTime = now;

                    if (found != lastLoggedState) {
                        if (found) log_safe::info("Capture", "Target acquired");
                        else       log_safe::info("Capture", "Target lost");
                        lastLoggedState = found;
                    }

                    OverlayTarget ot;
                    ot.found   = found;
                    ot.firing  = g_isFiring.load();
                    ot.circleX = g_crosshairMonX.load();
                    ot.circleY = g_crosshairMonY.load();
                    if (found) {
                        ot.boxX = cropX + best.box.x;
                        ot.boxY = cropY + best.box.y;
                        ot.boxW = best.box.width;
                        ot.boxH = best.box.height;
                    }
                    g_overlay.UpdateTarget(ot, radius);
                    continue;
                }

                if (st == DXGICapture::Status::Timeout) {
                    consecutiveErrors = 0;
                    std::this_thread::sleep_for(milliseconds(2));
                    continue;
                }

                if (st == DXGICapture::Status::Error) {
                    ++consecutiveErrors;
                    if (consecutiveErrors <= kTransientThreshold) {
                        if (consecutiveErrors == 1)
                            log_safe::warn("DXGI", "Transient GrabFrame error");
                        std::this_thread::sleep_for(milliseconds(10));
                        continue;
                    }
                    if (consecutiveErrors <= kSustainedThreshold) {
                        if (consecutiveErrors % 5 == 0)
                            log_safe::warn("DXGI", "Sustained GrabFrame errors");
                        std::this_thread::sleep_for(milliseconds(100));
                        continue;
                    }
                    log_safe::error("DXGI", "Critical errors; escalating");
                    st = DXGICapture::Status::AccessLost;
                }

                if (st == DXGICapture::Status::AccessLost ||
                    st == DXGICapture::Status::DeviceRemoved ||
                    st == DXGICapture::Status::SessionDisconnected) {

                    if (st == DXGICapture::Status::AccessLost) {
                        if (cap->RecreateDuplication()) {
                            consecutiveErrors = 0;
                            continue;
                        }
                        log_safe::warn("DXGI", "Fast recreate failed; full reinit");
                    } else {
                        log_safe::warn("DXGI", "Capture reset; full reinit");
                    }

                    int savedAdapter = cap->GetAdapterIdx();
                    int savedOutput  = cap->GetOutputIdx();
                    cap->Cleanup();

                    bool reinitSuccess = false;
                    for (int retry = 1; retry <= kCriticalReinitRetries &&
                                         g_running.load(); ++retry) {
                        std::this_thread::sleep_for(milliseconds(1000));
                        if (cap->Initialize(savedAdapter, savedOutput)) {
                            reinitSuccess = true;
                            break;
                        }
                    }
                    if (!reinitSuccess) {
                        log_safe::fatal("DXGI", "Reinit failed; shutting down");
                        g_running = false;
                        break;
                    }
                    consecutiveErrors = 0;
                    continue;
                }

                std::this_thread::sleep_for(milliseconds(50));
            }
            catch (const std::exception& e) {
                log_safe::error("Thread", ("Frame loop: " + std::string(e.what())).c_str());
                std::this_thread::sleep_for(milliseconds(100));
            }
            catch (...) {
                log_safe::error("Thread", "Frame loop: unknown");
                std::this_thread::sleep_for(milliseconds(100));
            }
        }
    }
    catch (...) {
        log_safe::fatal("Thread", "Capture fatal");
        g_running = false;
    }

    g_captureExited.store(true, std::memory_order_release);
    { std::lock_guard<std::mutex> lock(g_threadExitMtx); }
    g_threadExitCv.notify_all();
}

// ═══════════════════════════════════════════════════════════════
//  AimThread
// ═══════════════════════════════════════════════════════════════
static void AimThread(Config* cfg, RECT captureBounds) {
    try {
        bool altWasDown = false;
        constexpr auto kStaleAfter = std::chrono::milliseconds(150);

        while (g_running.load(std::memory_order_acquire)) {
            try {
                auto now = Clock::now();
                bool altDown = g_altPressed.load();

                if (altDown) {
                    TargetData t;
                    { std::lock_guard<std::mutex> lock(g_targetMutex); t = g_target; }

                    if (t.found && (now - t.detectionTime) > kStaleAfter) {
                        t.found = false;
                    }

                    if (!altWasDown && !t.found)
                        log_safe::info("Aim", "ALT: No target");

                    if (t.found) {
                        long long latency = duration_cast<microseconds>(
                            now - t.detectionTime).count();
                        latency = std::clamp<long long>(latency, 0, 200000);

                        float predX = t.screenX + t.vx * latency * cfg->predictionFactor
                                    + g_headOffsetX.load();
                        float predY = t.screenY + t.vy * latency * cfg->predictionFactor
                                    + g_headOffsetY.load();

                        if (std::isfinite(predX) && std::isfinite(predY)) {
                            POINT cur{};
                            if (GetCursorPos(&cur)) {
                                int desiredX = captureBounds.left + (int)std::lround(predX);
                                int desiredY = captureBounds.top  + (int)std::lround(predY);
                                int remX = desiredX - static_cast<int>(cur.x);
                                int remY = desiredY - static_cast<int>(cur.y);
                                int dist = (int)std::lround(
                                    std::hypot((double)remX, (double)remY));

                                if (dist <= (int)cfg->fireThresholdPx) {
                                    if (!g_isFiring) {
                                        if (SendMouseButton(true)) g_isFiring = true;
                                    }
                                } else {
                                    if (g_isFiring &&
                                        dist > (int)(cfg->fireThresholdPx * 2.0f)) {
                                        if (SendMouseButton(false)) g_isFiring = false;
                                    }
                                    int dx = (int)std::lround(remX * cfg->aimSpeed);
                                    int dy = (int)std::lround(remY * cfg->aimSpeed);
                                    if (dx != 0 || dy != 0) MoveMouse(dx, dy);
                                }
                            }
                        }
                    } else {
                        if (g_isFiring) {
                            if (SendMouseButton(false)) g_isFiring = false;
                        }
                    }
                } else if (g_isFiring) {
                    if (SendMouseButton(false)) g_isFiring = false;
                }

                altWasDown = altDown;
                std::this_thread::sleep_for(microseconds(500));
            }
            catch (const std::exception& e) {
                log_safe::error("Thread", ("Aim loop: " + std::string(e.what())).c_str());
                std::this_thread::sleep_for(milliseconds(100));
            }
            catch (...) {
                log_safe::error("Thread", "Aim loop: unknown");
                std::this_thread::sleep_for(milliseconds(100));
            }
        }
        if (g_isFiring) SendMouseButton(false);
    }
    catch (...) {
        log_safe::fatal("Thread", "Aim fatal");
        g_running = false;
    }

    g_aimExited.store(true, std::memory_order_release);
    { std::lock_guard<std::mutex> lock(g_threadExitMtx); }
    g_threadExitCv.notify_all();
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════
static void TestLog(const char* what, const char* detail) {
    try {
        Logger::Info("Test", std::string("[PASS] ") + what +
            (detail && *detail ? std::string(" — ") + detail : std::string()));
    } catch (...) {}
}
static void TestFail(const char* what, const char* detail) {
    try {
        Logger::Error("Test", std::string("[FAIL] ") + what + ": " +
            (detail ? detail : "?"));
    } catch (...) {}
}
static void TestSkip(const char* what, const char* why) {
    try {
        Logger::Info("Test", std::string("[SKIP] ") + what + ": " +
            (why ? why : "?"));
    } catch (...) {}
}

static bool Check(bool cond, const char* step, const char* detail) {
    if (cond) return true;
    TestFail(step, detail);
    return false;
}

static fs::path MakeTempPath(const std::string& baseName) {
    static std::atomic<uint64_t> counter{0};
    std::error_code ec;
    fs::path dir = fs::temp_directory_path(ec);
    if (ec) dir = fs::current_path(ec);
    if (ec) dir = ".";

    std::string stem, ext;
    try {
        fs::path orig = fs::u8path(baseName);
        stem = orig.stem().u8string();
        ext  = orig.extension().u8string();
    } catch (...) { stem = "tmp"; ext = ""; }

    auto pid = GetCurrentProcessId();
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();

    for (int attempt = 0; attempt < 100; ++attempt) {
        auto id = counter.fetch_add(1);
        std::string name = stem + "" + std::to_string(pid) + "" +
                          std::to_string(now) + "_" + std::to_string(id) + ext;
        fs::path p = dir / fs::u8path(name);
        std::error_code ec2;
        if (!fs::exists(p, ec2) || ec2) return p;
    }
    auto id = counter.fetch_add(1);
    return dir / fs::u8path("fallback_" + std::to_string(id) + ext);
}

static std::optional<TempFileGuard> CreateFileAndGuard(
    const fs::path& path, const char* data, size_t size) {
    try {
        std::ofstream f(path, std::ios::trunc | std::ios::binary);
        if (!f.is_open()) return std::nullopt;
        if (size > 0) f.write(data, static_cast<std::streamsize>(size));
        f.close();
        if (f.fail()) {
            std::error_code ec;
            fs::remove(path, ec);
            return std::nullopt;
        }
    } catch (...) {
        std::error_code ec;
        fs::remove(path, ec);
        return std::nullopt;
    }
    return TempFileGuard(path);
}

static std::string BuildTestIni(const fs::path& modelPath) {
    std::string s;
    s.reserve(256);
    s += "[AI]\n";
    s += "model = ";
    try { s += modelPath.u8string(); } catch (...) { s += modelPath.string(); }
    s += "\ninput_size = 320\n";
    s += "confidence = 0.5\n";
    return s;
}

static bool RunComponentTests(int& passed, int& skipped) {
    log_safe::info("Test", "=== Component Tests ===");

    {
        Config tc;
        if (!Check(tc.ParseInt("123", tc.inputSize), "Config parsing", "ParseInt('123')")) return false;
        if (!Check(tc.inputSize == 123, "Config parsing", "wrong value")) return false;
        if (!Check(!tc.ParseInt("123abc", tc.inputSize), "Config parsing", "'123abc' should fail")) return false;
        if (!Check(tc.ParseFloat("0.45", tc.nmsThreshold), "Config parsing", "ParseFloat('0.45')")) return false;
        if (!Check(!tc.ParseFloat("NaN", tc.nmsThreshold), "Config parsing", "'NaN' should fail")) return false;
        if (!Check(!tc.ParseFloat("1,5", tc.nmsThreshold), "Config parsing", "'1,5' should fail")) return false;
        bool b = false;
        if (!Check(tc.ParseBool("true", b) && b, "Config parsing", "bool parse")) return false;
        TestLog("Config parsing", nullptr);
        ++passed;
    }

    {
        cv::Mat valid(std::vector<int>{1, 84, 8400}, CV_32F, cv::Scalar(0.0f));
        TensorLayout l = YoloDetector::AnalyzeTensorStatic(valid, "yolov8", 80);
        if (!Check(l.valid && l.featureMajor && l.features == 84,
                   "Tensor analysis", "valid tensor rejected")) return false;
        cv::Mat empty(std::vector<int>{1, 84, 0}, CV_32F, cv::Scalar(0.0f));
        if (!Check(!YoloDetector::AnalyzeTensorStatic(empty, "yolov8", 80).valid,
                   "Tensor analysis", "empty accepted")) return false;
        TestLog("Tensor analysis", nullptr);
        ++passed;
    }

    {
        fs::path iniPath   = MakeTempPath("test_config.ini");
        fs::path modelPath = MakeTempPath("dummy.onnx");

        std::string iniContent = BuildTestIni(modelPath);
        auto guardIni = CreateFileAndGuard(iniPath,
            iniContent.data(), iniContent.size());
        if (!Check(guardIni.has_value(), "Config file loading", "write ini failed")) return false;

        std::vector<char> zeros(2048, 0);
        auto guardModel = CreateFileAndGuard(modelPath, zeros.data(), zeros.size());
        if (!Check(guardModel.has_value(), "Config file loading", "write model failed")) return false;

        Config tcfg;
        if (!Check(tcfg.Load(iniPath), "Config file loading", "Load failed")) return false;
        if (!Check(tcfg.confThreshold == 0.5f, "Config file loading", "threshold wrong")) return false;
        TestLog("Config file loading", "not model loading");
        ++passed;
    }

    {
        fs::path iniPath   = MakeTempPath("test_config_neg.ini");
        fs::path modelPath = MakeTempPath("dummy_neg.onnx");

        std::string iniContent = BuildTestIni(modelPath);
        auto guardIni = CreateFileAndGuard(iniPath,
            iniContent.data(), iniContent.size());
        if (!Check(guardIni.has_value(), "Reject invalid ONNX", "write ini")) return false;

        std::vector<char> zeros(2048, 0);
        auto guardModel = CreateFileAndGuard(modelPath, zeros.data(), zeros.size());
        if (!Check(guardModel.has_value(), "Reject invalid ONNX", "write model")) return false;

        Config tcfg;
        if (!Check(tcfg.Load(iniPath), "Reject invalid ONNX", "Load failed")) return false;

        YoloDetector det;
        if (!Check(!det.Load(tcfg), "Reject invalid ONNX", "accepted invalid")) return false;

        cv::Mat dummyFrame(320, 320, CV_8UC3, cv::Scalar(0));
        std::vector<Detection> results;
        det.Detect(dummyFrame, results);
        if (!Check(results.empty(), "Reject invalid ONNX", "Detect after failed Load")) return false;

        if (!Check(!det.Load(tcfg), "Reject invalid ONNX", "accepted on 2nd Load")) return false;
        TestLog("YoloDetector rejects invalid ONNX", nullptr);
        ++passed;
    }

    if (!HasInteractiveDesktop()) {
        TestSkip("Overlay lifecycle", "no interactive desktop");
        ++skipped;
    } else {
        Overlay o;
        RECT invalid = { 0, 0, 0, 0 };
        if (!Check(!o.Start(invalid, 150), "Overlay lifecycle", "accepted invalid bounds")) return false;
        o.Stop();
        o.Stop();

        RECT valid = { 0, 0, 1920, 1080 };
        if (!Check(o.Start(valid, 150), "Overlay lifecycle", "rejected valid")) return false;
        if (!Check(o.IsAlive(), "Overlay lifecycle", "not alive after Start")) { o.Stop(); return false; }
        o.Stop();
        if (!Check(!o.IsAlive(), "Overlay lifecycle", "alive after Stop")) return false;

        if (!Check(o.Start(valid, 150), "Overlay lifecycle", "rejected restart")) return false;
        o.Stop();
        TestLog("Overlay lifecycle", nullptr);
        ++passed;
    }

    return true;
}

static bool ShouldRunTests(int argc, char** argv) {
#if defined(_DEBUG) || defined(AIMBOT_RUN_TESTS)
    const bool defaultRun = true;
#else
    const bool defaultRun = false;
#endif
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test") == 0)     return true;
        if (std::strcmp(argv[i], "--no-test") == 0)  return false;
    }
    return defaultRun;
}

// ═══════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    // ✅ 1) DPI أولاً — قبل أي نافذة (MSDN: call early)
    if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        DWORD err = GetLastError();
        if (err != ERROR_ACCESS_DENIED) {
            OutputDebugStringA("[Main] SetProcessDpiAwarenessContext failed\n");
        }
    }

    // ✅ 2) Console CP قبل أي إخراج
    if (!SetConsoleOutputCP(CP_UTF8)) {
        OutputDebugStringA("[Main] SetConsoleOutputCP failed\n");
    }

    // ✅ 3) Logger آخراً (ينشئ Console بالدقة الصحيحة)
    Logger::Get().Init(fs::path("aimbot_log.txt"));
    if (!Logger::Get().IsFileOpen()) {
        OutputDebugStringA("[Main] Log file unavailable\n");
        std::cerr << "WARNING: Log file not opened (logging to console only)\n";
    }

    Logger::Info("Main", "=== Aimbot Enterprise v140 ===");

    // ✅ Logger RAII — يضمن Shutdown في كل مسار
    ScopeGuard loggerGuard([]() noexcept {
        try { Logger::Get().Shutdown(); } catch (...) {}
    });

    // ✅ Conditional tests
    if (ShouldRunTests(argc, argv)) {
        int passed = 0, skipped = 0;
        if (!RunComponentTests(passed, skipped)) {
            std::cout << "Self-tests failed. Press Enter to exit...\n";
            std::cin.get();
            return 1;
        }
        try {
            Logger::Info("Test",
                "=== " + std::to_string(passed) + " passed, " +
                std::to_string(skipped) + " skipped ===");
        } catch (...) {}
    }

    Config cfg;
    if (!cfg.Load("config.ini")) {
        std::cout << "Config failed. Press Enter to exit...\n";
        std::cin.get();
        return 1;
    }

    auto monitors = DXGICapture::EnumerateMonitors();
    if (monitors.empty()) {
        log_safe::fatal("Main", "No monitors found");
        std::cout << "Press Enter to exit...\n";
        std::cin.get();
        return 1;
    }

    RECT captureBounds = { 0, 0, 0, 0 };
    int selAdapter = 0, selOutput = 0;
    bool foundMonitor = false;

    if (cfg.monitorMode == 0) {
        HWND gameWin = FindGameWindow(cfg.gameProcess);
        if (gameWin) {
            RECT r;
            if (GetWindowRect(gameWin, &r)) {
                POINT pt = { (r.left + r.right) / 2, (r.top + r.bottom) / 2 };
                if (auto target = DXGICapture::FindMonitorAt(monitors, pt)) {
                    selAdapter = target->adapterIdx;
                    selOutput  = target->outputIdx;
                    captureBounds = target->bounds;
                    foundMonitor = true;
                }
            }
        }
    }
    if (!foundMonitor) {
        for (const auto& m : monitors) {
            if (m.isPrimary) {
                selAdapter = m.adapterIdx;
                selOutput  = m.outputIdx;
                captureBounds = m.bounds;
                foundMonitor = true;
                break;
            }
        }
    }
    if (!foundMonitor) {
        log_safe::fatal("Main", "No valid monitor");
        std::cout << "Press Enter to exit...\n";
        std::cin.get();
        return 1;
    }

    {
        const int w = captureBounds.right  - captureBounds.left;
        const int h = captureBounds.bottom - captureBounds.top;
        if (w <= 0 || h <= 0 || w > 16384 || h > 16384) {
            log_safe::fatal("Main",
                ("Invalid capture bounds: " + std::to_string(w) + "x" +
                 std::to_string(h)).c_str());
            std::cout << "Press Enter to exit...\n";
            std::cin.get();
            return 1;
        }
    }

    DXGICapture cap;
    if (!cap.Initialize(selAdapter, selOutput)) {
        log_safe::fatal("Main", "DXGI failed");
        std::cout << "Press Enter to exit...\n";
        std::cin.get();
        return 1;
    }

    YoloDetector det;
    if (!det.Load(cfg)) {
        log_safe::fatal("Main", "YOLO failed");
        std::cout << "Press Enter to exit...\n";
        std::cin.get();
        return 1;
    }

    if (!g_overlay.Start(captureBounds, cfg.circleRadius)) {
        log_safe::fatal("Main", "Overlay failed");
        std::cout << "Press Enter to exit...\n";
        std::cin.get();
        return 1;
    }

    POINT initCur{};
    if (GetCursorPos(&initCur)) {
        const int W = captureBounds.right  - captureBounds.left;
        const int H = captureBounds.bottom - captureBounds.top;
        int mx = std::clamp(static_cast<int>(initCur.x - captureBounds.left), 0, W - 1);
        int my = std::clamp(static_cast<int>(initCur.y - captureBounds.top),  0, H - 1);
        g_crosshairMonX.store(mx);
        g_crosshairMonY.store(my);
    }

    DisableMouseAcceleration();

    TimerResolutionGuard timerGuard;

    std::thread tCap, tAim;

    g_captureExited.store(false, std::memory_order_release);
    g_aimExited.store(false, std::memory_order_release);

    auto doCleanup = [&]() noexcept {
        try { g_running = false; } catch (...) {}
        SafeJoin(tCap, g_captureExited, "CaptureThread");
        SafeJoin(tAim, g_aimExited, "AimThread");
        if (g_isFiring) {
            try { SendMouseButton(false); g_isFiring = false; } catch (...) {}
        }
        try { g_overlay.Stop(); } catch (...) {}
        RestoreMouseAcceleration();
    };

    {
        ScopeGuard cleanupGuard(doCleanup);

        try {
            tCap = std::thread(CaptureThread, &cap, &det, &cfg);
            try {
                tAim = std::thread(AimThread, &cfg, captureBounds);
            } catch (const std::system_error& e) {
                log_safe::fatal("Main", ("Failed to create AimThread: " +
                    std::string(e.what())).c_str());
                return 1;
            }

            Logger::Info("Main", "Running. F8 to exit.");

            HWND trackedWindow = nullptr;
            auto lastWindowCheck = Clock::now();
            bool warnedOffMonitor = false;

            while (g_running.load(std::memory_order_acquire)) {
                POINT cur{};
                if (GetCursorPos(&cur)) {
                    const int W = captureBounds.right  - captureBounds.left;
                    const int H = captureBounds.bottom - captureBounds.top;
                    int mx = std::clamp(static_cast<int>(cur.x - captureBounds.left), 0, W - 1);
                    int my = std::clamp(static_cast<int>(cur.y - captureBounds.top),  0, H - 1);
                    g_crosshairMonX.store(mx);
                    g_crosshairMonY.store(my);
                }

                if (Clock::now() - lastWindowCheck > milliseconds(500)) {
                    lastWindowCheck = Clock::now();
                    if (cfg.monitorMode == 0) {
                        HWND h = FindGameWindow(cfg.gameProcess);
                        if (h != trackedWindow) {
                            trackedWindow = h;
                            if (h) log_safe::info("Main", "Game window tracked");
                            else   log_safe::warn("Main", "Game window lost");
                        }
                        if (h && !warnedOffMonitor) {
                            RECT cr{};
                            if (GetClientRect(h, &cr)) {
                                POINT tl{cr.left, cr.top}, br{cr.right, cr.bottom};
                                ClientToScreen(h, &tl);
                                ClientToScreen(h, &br);
                                if (tl.x < captureBounds.left || br.x > captureBounds.right ||
                                    tl.y < captureBounds.top  || br.y > captureBounds.bottom) {
                                    log_safe::warn("Main", "Game window beyond monitor");
                                    warnedOffMonitor = true;
                                }
                            }
                        }
                    }
                }

                g_altPressed.store((GetAsyncKeyState(VK_MENU) & 0x8000) != 0);
                if (GetAsyncKeyState(VK_F8) & 0x8000) break;
                if (GetAsyncKeyState(VK_UP) & 0x8000)    { g_headOffsetY--; Sleep(50); }
                if (GetAsyncKeyState(VK_DOWN) & 0x8000)  { g_headOffsetY++; Sleep(50); }
                if (GetAsyncKeyState(VK_LEFT) & 0x8000)  { g_headOffsetX--; Sleep(50); }
                if (GetAsyncKeyState(VK_RIGHT) & 0x8000) { g_headOffsetX++; Sleep(50); }
                std::this_thread::sleep_for(milliseconds(15));
            }
        }
        catch (const std::exception& e) {
            log_safe::fatal("Main", ("Exception: " + std::string(e.what())).c_str());
        }
        catch (...) {
            log_safe::fatal("Main", "Unknown exception");
        }
    } // cleanupGuard destructs here — threads joined, overlay stopped

    Logger::Info("Main", "Shutdown complete.");
    return 0;
}
