#pragma once
#include <windows.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <climits>
#include <string>
#include <exception>
#include "Logger.h"
#include "Config.h"

struct OverlayTarget {
    bool found = false;
    int boxX = 0, boxY = 0, boxW = 0, boxH = 0;
    int circleX = 0, circleY = 0;
    bool firing = false;
};

class Overlay {
public:
    enum class State { Idle, Starting, Running, Stopping, Stopped };

    bool Start(RECT bounds, int initialRadius) {
        {
            std::lock_guard<std::mutex> lock(m_lifecycleMtx);

            if (m_thread.joinable()) {
                if (m_threadExited.load(std::memory_order_acquire)) {
                    SafeJoinLocked("Start-stale");
                    m_state.store(State::Idle, std::memory_order_release);
                } else {
                    Logger::Error("Overlay", "Start(): previous thread still alive");
                    return false;
                }
            }

            State cur = m_state.load(std::memory_order_acquire);
            if (cur != State::Idle && cur != State::Stopped) {
                Logger::Error("Overlay",
                    "Start(): invalid state " + std::to_string(static_cast<int>(cur)));
                return false;
            }

            m_state.store(State::Starting, std::memory_order_release);

            {
                std::lock_guard<std::mutex> tlock(m_mtx);
                m_Bounds = bounds;
                m_circleRadius = std::clamp(initialRadius, 10, 1000);
                m_prevDirty.valid = false;
            }

            static std::atomic<int> s_instanceCounter{0};
            m_className = L"OverlayV140_" +
                          std::to_wstring(GetCurrentProcessId()) + L"_" +
                          std::to_wstring(s_instanceCounter.fetch_add(1));

            m_running.store(true, std::memory_order_release);
            m_initSuccess.store(false);
            m_initDone.store(false);
            m_threadExited.store(false);

            try {
                m_thread = std::thread(&Overlay::RenderThread, this);
            } catch (...) {
                Logger::Error("Overlay", "Thread creation failed");
                m_running.store(false);
                m_state.store(State::Idle);
                return false;
            }
        }

        bool initOk = false;
        {
            std::unique_lock<std::mutex> initLock(m_initMtx);
            const bool completed = m_initCv.wait_for(
                initLock, std::chrono::seconds(3),
                [this] { return m_initDone.load(); });
            initOk = completed && m_initSuccess.load();
        }

        if (!initOk) {
            Logger::Error("Overlay", "Initialization failed or timed out");
            Stop();
            return false;
        }

        State expectStarting = State::Starting;
        if (!m_state.compare_exchange_strong(expectStarting, State::Running,
                                              std::memory_order_acq_rel)) {
            Logger::Error("Overlay",
                "Start(): state changed during init (now " +
                std::to_string(static_cast<int>(m_state.load())) +
                ") — Stop() raced us");
            return false;
        }

        if (m_threadExited.load(std::memory_order_acquire)) {
            Logger::Error("Overlay", "Thread exited immediately after init");
            State expectRunning = State::Running;
            m_state.compare_exchange_strong(expectRunning, State::Stopped);
            return false;
        }

        return true;
    }

    void Stop() {
        std::lock_guard<std::mutex> lock(m_lifecycleMtx);

        if (!m_thread.joinable()) {
            State cur = m_state.load();
            if (cur == State::Idle || cur == State::Stopped) return;
            m_state.store(State::Stopped);
            return;
        }

        {
            std::lock_guard<std::mutex> waitLock(m_waitMtx);
            m_running.store(false, std::memory_order_release);
        }
        m_stopCv.notify_all();

        HWND hwnd = m_hwnd.load(std::memory_order_acquire);
        if (hwnd && IsWindow(hwnd)) PostMessageW(hwnd, WM_CLOSE, 0, 0);

        if (std::this_thread::get_id() == m_thread.get_id()) {
            Logger::Warn("Overlay",
                "Stop() from render thread — external Stop() required");
            return;
        }

        State expectStart = State::Starting;
        if (!m_state.compare_exchange_strong(expectStart, State::Stopping)) {
            State expectRun = State::Running;
            m_state.compare_exchange_strong(expectRun, State::Stopping);
        }

        bool exited = false;
        {
            std::unique_lock<std::mutex> exitLock(m_exitMtx);
            exited = m_exitCv.wait_for(exitLock, std::chrono::seconds(5),
                [this] { return m_threadExited.load(std::memory_order_acquire); });
        }

        if (!exited) {
            Logger::Fatal("Overlay",
                "RenderThread did not signal exit in 5s. "
                "Thread is blocked in UpdateLayeredWindow or PeekMessage. "
                "join() will now block until that syscall returns. "
                "No safe alternative exists (detach causes UAF).");
        }

        SafeJoinLocked("Stop");
        m_state.store(State::Stopped, std::memory_order_release);
    }

    void UpdateTarget(const OverlayTarget& t, int radius) {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_target = t;
        m_circleRadius = std::clamp(radius, 10, 1000);

        const int64_t w64 = static_cast<int64_t>(m_Bounds.right)
                          - static_cast<int64_t>(m_Bounds.left);
        const int64_t h64 = static_cast<int64_t>(m_Bounds.bottom)
                          - static_cast<int64_t>(m_Bounds.top);
        if (w64 <= 0 || h64 <= 0 || w64 > INT_MAX || h64 > INT_MAX) return;
        const int w = static_cast<int>(w64);
        const int h = static_cast<int>(h64);

        m_target.circleX = std::clamp(m_target.circleX, 0, w - 1);
        m_target.circleY = std::clamp(m_target.circleY, 0, h - 1);

        if (m_target.boxW < 0) { m_target.boxX += m_target.boxW; m_target.boxW = 0; }
        if (m_target.boxH < 0) { m_target.boxY += m_target.boxH; m_target.boxH = 0; }

        int64_t bx1 = m_target.boxX;
        int64_t by1 = m_target.boxY;
        int64_t bx2 = bx1 + static_cast<int64_t>(m_target.boxW);
        int64_t by2 = by1 + static_cast<int64_t>(m_target.boxH);

        bx1 = std::clamp<int64_t>(bx1, 0, w);
        by1 = std::clamp<int64_t>(by1, 0, h);
        bx2 = std::clamp<int64_t>(bx2, bx1, w);
        by2 = std::clamp<int64_t>(by2, by1, h);

        m_target.boxX = static_cast<int>(bx1);
        m_target.boxY = static_cast<int>(by1);
        m_target.boxW = static_cast<int>(bx2 - bx1);
        m_target.boxH = static_cast<int>(by2 - by1);
    }

    bool IsAlive() const {
        return m_state.load(std::memory_order_acquire) == State::Running &&
               !m_threadExited.load(std::memory_order_acquire);
    }

    State GetState() const { return m_state.load(); }

    ~Overlay() {
        if (m_thread.joinable() &&
            std::this_thread::get_id() == m_thread.get_id()) {
            try {
                OutputDebugStringA(
                    "[Overlay] FATAL: destructor called from render thread\n");
            } catch (...) {}
            std::terminate();
        }
        Stop();
    }

private:
    void SafeJoinLocked(const char* context) noexcept {
        if (!m_thread.joinable()) return;

        try {
            m_thread.join();
            return;
        } catch (const std::system_error& e) {
            try {
                Logger::Error("Overlay",
                    std::string(context) + ": join failed: " + e.what());
            } catch (...) {}
        } catch (...) {}

        if (m_threadExited.load(std::memory_order_acquire)) {
            try { m_thread.detach(); }
            catch (...) {
                Logger::Fatal("Overlay",
                    std::string(context) +
                    ": join and detach both failed — terminating");
                std::terminate();
            }
        } else {
            Logger::Fatal("Overlay",
                std::string(context) +
                ": join failed while thread alive — terminating to prevent UAF");
            std::terminate();
        }
    }

    static LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM w, LPARAM l) {
        Overlay* self = reinterpret_cast<Overlay*>(
            GetWindowLongPtrW(hwnd, GWLP_USERDATA));

        switch (msg) {
            case WM_NCCREATE: {
                auto* cs = reinterpret_cast<CREATESTRUCTW*>(l);
                SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                    reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
                return TRUE;
            }
            case WM_CLOSE:
                if (self) self->m_running.store(false, std::memory_order_release);
                DestroyWindow(hwnd);
                return 0;
            case WM_DESTROY:
                PostQuitMessage(0);
                return 0;
            case WM_NCDESTROY:
                if (self) {
                    HWND expected = hwnd;
                    self->m_hwnd.compare_exchange_strong(
                        expected, nullptr,
                        std::memory_order_acq_rel);
                }
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                break;
            default:
                break;
        }
        return DefWindowProcW(hwnd, msg, w, l);
    }

    struct ClassRegistrar {
        std::wstring name;
        HINSTANCE    inst = nullptr;
        bool         registered = false;
        ~ClassRegistrar() noexcept {
            if (registered && !name.empty() && inst) {
                try { UnregisterClassW(name.c_str(), inst); } catch (...) {}
            }
        }
    };

    struct GdiObjectSelector {
        HDC     dc  = nullptr;
        HGDIOBJ obj = nullptr;
        HGDIOBJ old = nullptr;
        bool    ok  = false;
        GdiObjectSelector(HDC hdc, HGDIOBJ o) : dc(hdc), obj(o) {
            if (!dc || !obj) return;
            old = SelectObject(dc, obj);
            ok  = (old != nullptr && old != HGDI_ERROR);
        }
        ~GdiObjectSelector() noexcept {
            if (dc && old && old != HGDI_ERROR) {
                try { SelectObject(dc, old); } catch (...) {}
            }
        }
    };

    struct GdiGuard {
        HWND     hwnd = nullptr;
        HDC      hScreenDC = nullptr;
        HDC      hMemDC = nullptr;
        HBITMAP  hBitmap = nullptr;
        HGDIOBJ  oldBitmap = nullptr;
        HPEN     hPenGreen = nullptr, hPenYellow = nullptr, hPenRed = nullptr;

        static bool IsValidHandle(HGDIOBJ h) noexcept {
            return h != nullptr && h != HGDI_ERROR;
        }
        ~GdiGuard() noexcept {
            try {
                if (hMemDC && IsValidHandle(oldBitmap)) SelectObject(hMemDC, oldBitmap);
                if (hPenGreen)  DeleteObject(hPenGreen);
                if (hPenYellow) DeleteObject(hPenYellow);
                if (hPenRed)    DeleteObject(hPenRed);
                if (hBitmap)    DeleteObject(hBitmap);
                if (hMemDC)     DeleteDC(hMemDC);
                if (hScreenDC)  ReleaseDC(nullptr, hScreenDC);
            } catch (...) {}
            if (hwnd && IsWindow(hwnd)) {
                try { if (!DestroyWindow(hwnd)) {} } catch (...) {}
            }
        }
    };

    struct DirtyRegion {
        int  x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        bool valid = false;
        void merge(int nx1, int ny1, int nx2, int ny2) noexcept {
            if (!valid) { x1 = nx1; y1 = ny1; x2 = nx2; y2 = ny2; valid = true; return; }
            x1 = (std::min)(x1, nx1); y1 = (std::min)(y1, ny1);
            x2 = (std::max)(x2, nx2); y2 = (std::max)(y2, ny2);
        }
    };

    static void ClearRegion(void* pBits, int w, int h,
                            const DirtyRegion& r) noexcept {
        if (!r.valid) return;
        int x1 = std::clamp(r.x1, 0, w);
        int y1 = std::clamp(r.y1, 0, h);
        int x2 = std::clamp(r.x2, x1, w);
        int y2 = std::clamp(r.y2, y1, h);
        if (x2 <= x1 || y2 <= y1) return;
        uint8_t* base = static_cast<uint8_t*>(pBits);
        const size_t rowBytes = static_cast<size_t>(w) * 4u;
        const size_t x1Bytes  = static_cast<size_t>(x1) * 4u;
        const size_t rowLen   = static_cast<size_t>(x2 - x1) * 4u;
        for (int y = y1; y < y2; ++y)
            memset(base + static_cast<size_t>(y) * rowBytes + x1Bytes, 0, rowLen);
    }

    void RenderThread() noexcept {
        try {
            ClassRegistrar classReg;
            classReg.name = m_className;
            classReg.inst = GetModuleHandleW(0);

            GdiGuard gdi;

            WNDCLASSEXW wc = { sizeof(WNDCLASSEXW) };
            wc.lpfnWndProc   = OverlayWndProc;
            wc.hInstance     = classReg.inst;
            wc.lpszClassName = classReg.name.c_str();
            if (!RegisterClassExW(&wc))
                throw std::runtime_error("RegisterClassExW failed");
            classReg.registered = true;

            RECT bounds;
            { std::lock_guard<std::mutex> lock(m_mtx); bounds = m_Bounds; }

            const int64_t w64 = static_cast<int64_t>(bounds.right)
                              - static_cast<int64_t>(bounds.left);
            const int64_t h64 = static_cast<int64_t>(bounds.bottom)
                              - static_cast<int64_t>(bounds.top);
            if (w64 <= 0 || h64 <= 0 || w64 > 16384 || h64 > 16384)
                throw std::runtime_error("Invalid dimensions");
            const int w = static_cast<int>(w64);
            const int h = static_cast<int>(h64);

            const uint64_t wu = static_cast<uint64_t>(w);
            const uint64_t hu = static_cast<uint64_t>(h);
            if (hu == 0 || wu > (SIZE_MAX / 4u) / hu)
                throw std::runtime_error("DIB size overflow");
            const size_t dibSize = static_cast<size_t>(wu * hu * 4u);

            gdi.hwnd = CreateWindowExW(
                WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                classReg.name.c_str(), L"", WS_POPUP,
                static_cast<int>(bounds.left), static_cast<int>(bounds.top),
                w, h, 0, 0, classReg.inst, this);
            if (!gdi.hwnd) throw std::runtime_error("CreateWindowExW failed");
            m_hwnd.store(gdi.hwnd, std::memory_order_release);

            gdi.hScreenDC = GetDC(nullptr);
            if (!gdi.hScreenDC) throw std::runtime_error("GetDC failed");
            gdi.hMemDC = CreateCompatibleDC(gdi.hScreenDC);
            if (!gdi.hMemDC) throw std::runtime_error("CreateCompatibleDC failed");

            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth       = w;
            bmi.bmiHeader.biHeight      = -h;
            bmi.bmiHeader.biPlanes      = 1;
            bmi.bmiHeader.biBitCount    = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            void* pBits = nullptr;
            gdi.hBitmap = CreateDIBSection(gdi.hScreenDC, &bmi,
                DIB_RGB_COLORS, &pBits, nullptr, 0);
            if (!gdi.hBitmap || !pBits)
                throw std::runtime_error("CreateDIBSection failed");

            gdi.oldBitmap = SelectObject(gdi.hMemDC, gdi.hBitmap);
            if (!GdiGuard::IsValidHandle(gdi.oldBitmap))
                throw std::runtime_error("SelectObject (DIB) failed");

            gdi.hPenGreen  = CreatePen(PS_SOLID, 2, RGB(0, 255, 0));
            gdi.hPenYellow = CreatePen(PS_SOLID, 2, RGB(255, 255, 0));
            gdi.hPenRed    = CreatePen(PS_SOLID, 2, RGB(255, 0, 0));
            if (!gdi.hPenGreen || !gdi.hPenYellow || !gdi.hPenRed)
                throw std::runtime_error("CreatePen failed");

            HBRUSH hNullBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
            MSG msg = {};
            auto nextFrame = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lock(m_initMtx);
                m_initSuccess.store(true);
                m_initDone.store(true);
                m_initCv.notify_all();
            }

            while (m_running.load(std::memory_order_acquire)) {
                while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    if (msg.message == WM_QUIT) m_running.store(false);
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                if (!m_running.load(std::memory_order_acquire)) break;

                OverlayTarget t;
                int rad;
                DirtyRegion prevDirty;
                {
                    std::lock_guard<std::mutex> lock(m_mtx);
                    t = m_target; rad = m_circleRadius; prevDirty = m_prevDirty;
                }

                if (prevDirty.valid) ClearRegion(pBits, w, h, prevDirty);
                else                 memset(pBits, 0, dibSize);

                DirtyRegion curDirty;
                HPEN pen = t.firing ? gdi.hPenRed
                          : (t.found  ? gdi.hPenYellow : gdi.hPenGreen);

                {
                    GdiObjectSelector penSel(gdi.hMemDC, pen);
                    GdiObjectSelector brushSel(gdi.hMemDC, hNullBrush);

                    if (penSel.ok && brushSel.ok) {
                        const int64_t cX = t.circleX, cY = t.circleY, cR = rad;
                        const int64_t cLeft = cX - cR, cTop = cY - cR;
                        const int64_t cRight = cX + cR, cBottom = cY + cR;

                        if (!Ellipse(gdi.hMemDC, (int)cLeft, (int)cTop,
                                     (int)cRight, (int)cBottom)) {}
                        curDirty.merge(
                            (std::max)(0, (int)(cLeft - 2)),
                            (std::max)(0, (int)(cTop  - 2)),
                            (std::min)(w, (int)(cRight  + 2)),
                            (std::min)(h, (int)(cBottom + 2)));

                        if (t.found && t.boxW > 0 && t.boxH > 0) {
                            const int64_t bx1 = t.boxX, by1 = t.boxY;
                            const int64_t bx2 = bx1 + static_cast<int64_t>(t.boxW);
                            const int64_t by2 = by1 + static_cast<int64_t>(t.boxH);
                            if (!Rectangle(gdi.hMemDC, (int)bx1, (int)by1,
                                           (int)bx2, (int)by2)) {}
                            curDirty.merge(
                                (std::max)(0, (int)(bx1 - 2)),
                                (std::max)(0, (int)(by1 - 2)),
                                (std::min)(w, (int)(bx2 + 2)),
                                (std::min)(h, (int)(by2 + 2)));
                        }
                    }
                }

                if (curDirty.valid) {
                    int cx1 = std::clamp(curDirty.x1, 0, w);
                    int cy1 = std::clamp(curDirty.y1, 0, h);
                    int cx2 = std::clamp(curDirty.x2, cx1, w);
                    int cy2 = std::clamp(curDirty.y2, cy1, h);
                    uint32_t* pPixels = reinterpret_cast<uint32_t*>(pBits);
                    for (int y = cy1; y < cy2; ++y) {
                        uint32_t* row = pPixels + static_cast<size_t>(y) * w;
                        for (int x = cx1; x < cx2; ++x)
                            if (row[x] != 0) row[x] |= 0xFF000000;
                    }
                }

                { std::lock_guard<std::mutex> lock(m_mtx); m_prevDirty = curDirty; }

                POINT ptSrc = { 0, 0 };
                POINT ptDst = { static_cast<LONG>(bounds.left),
                                static_cast<LONG>(bounds.top) };
                SIZE sz = { w, h };
                BLENDFUNCTION blend = { AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };

                if (!UpdateLayeredWindow(gdi.hwnd, gdi.hScreenDC, &ptDst, &sz,
                                         gdi.hMemDC, &ptSrc, 0, &blend, ULW_ALPHA)) {
                    DWORD err = GetLastError();
                    try {
                        Logger::Error("Overlay", "UpdateLayeredWindow failed: " +
                            std::to_string(err));
                    } catch (...) {}
                    m_running.store(false);
                    break;
                }

                nextFrame += std::chrono::milliseconds(16);
                const auto now = std::chrono::steady_clock::now();
                if (nextFrame < now - std::chrono::milliseconds(100))
                    nextFrame = now + std::chrono::milliseconds(16);
                {
                    std::unique_lock<std::mutex> waitLock(m_waitMtx);
                    m_stopCv.wait_until(waitLock, nextFrame,
                        [this] { return !m_running.load(std::memory_order_acquire); });
                }
            }
        }
        catch (const std::exception& e) {
            try { Logger::Error("Overlay", std::string("RenderThread: ") + e.what()); }
            catch (...) {}
        }
        catch (...) {
            try { Logger::Error("Overlay", "RenderThread: unknown"); }
            catch (...) {}
        }

        {
            std::lock_guard<std::mutex> lock(m_initMtx);
            if (!m_initDone.load()) {
                m_initSuccess.store(false);
                m_initDone.store(true);
                m_initCv.notify_all();
            }
        }

        {
            HWND expected = m_hwnd.load(std::memory_order_acquire);
            if (expected) m_hwnd.compare_exchange_strong(expected, nullptr);
        }

        {
            State cur = m_state.load(std::memory_order_acquire);
            if (cur == State::Running || cur == State::Starting) {
                State expected = cur;
                m_state.compare_exchange_strong(
                    expected, State::Stopped,
                    std::memory_order_acq_rel);
            }
        }

        m_threadExited.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> exitLock(m_exitMtx);
            m_exitCv.notify_all();
        }
    }

    RECT m_Bounds = { 0, 0, 0, 0 };

    std::atomic<State>      m_state{ State::Idle };
    std::atomic<bool>       m_running{ false };
    std::atomic<bool>       m_initSuccess{ false };
    std::atomic<bool>       m_initDone{ false };
    std::atomic<bool>       m_threadExited{ false };
    std::atomic<HWND>       m_hwnd{ nullptr };
    std::atomic<int>        m_circleRadius{ 150 };

    std::wstring            m_className;
    OverlayTarget           m_target;
    DirtyRegion             m_prevDirty;

    std::mutex              m_mtx;
    std::mutex              m_initMtx;
    std::mutex              m_lifecycleMtx;
    std::mutex              m_waitMtx;
    std::mutex              m_exitMtx;

    std::condition_variable m_initCv;
    std::condition_variable m_stopCv;
    std::condition_variable m_exitCv;

    std::thread             m_thread;
};
