#pragma once
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <limits>
#include <mutex>
#include <atomic>
#include <algorithm>
#include "Logger.h"

using Microsoft::WRL::ComPtr;

struct MonitorInfo {
    HMONITOR handle = nullptr;
    RECT     bounds = { 0, 0, 0, 0 };
    int      adapterIdx = 0;
    int      outputIdx = 0;
    bool     isPrimary = false;
    DXGI_MODE_ROTATION rotation = DXGI_MODE_ROTATION_IDENTITY;
};

class DXGICapture {
public:
    enum class Status {
        Success, Timeout, AccessLost, DeviceRemoved, SessionDisconnected, Error
    };

    DXGICapture() = default;
    ~DXGICapture() {
        std::lock_guard<std::mutex> lock(m_apiMtx);
        CleanupInternal();
    }
    DXGICapture(const DXGICapture&) = delete;
    DXGICapture& operator=(const DXGICapture&) = delete;

    // ═══════════════════════════════════════════════════════════
    //  EnumerateMonitors — static
    // ═══════════════════════════════════════════════════════════
    static std::vector<MonitorInfo> EnumerateMonitors() {
        std::vector<MonitorInfo> result;
        ComPtr<IDXGIFactory> factory = CreateFactory();
        if (!factory) return result;

        for (UINT a = 0; ; ++a) {
            ComPtr<IDXGIAdapter> adapter;
            HRESULT hr = factory->EnumAdapters(a, adapter.GetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(hr)) {
                Logger::Error("DXGI", "EnumAdapters failed at index " +
                    std::to_string(a), hr);
                break;
            }

            for (UINT o = 0; ; ++o) {
                ComPtr<IDXGIOutput> output;
                hr = adapter->EnumOutputs(o, output.GetAddressOf());
                if (hr == DXGI_ERROR_NOT_FOUND) break;
                if (FAILED(hr)) {
                    Logger::Error("DXGI", "EnumOutputs failed at index " +
                        std::to_string(o), hr);
                    break;
                }

                DXGI_OUTPUT_DESC desc{};
                hr = output->GetDesc(&desc);
                if (FAILED(hr)) {
                    Logger::Error("DXGI", "Output GetDesc failed (adapter " +
                        std::to_string(a) + ", output " + std::to_string(o) + ")", hr);
                    continue;
                }
                if (desc.Monitor == nullptr || !desc.AttachedToDesktop) continue;

                MonitorInfo mi{};
                mi.handle     = desc.Monitor;
                mi.bounds     = desc.DesktopCoordinates;
                mi.adapterIdx = static_cast<int>(a);
                mi.outputIdx  = static_cast<int>(o);
                mi.rotation   = desc.Rotation;

                MONITORINFO monInfo{ sizeof(MONITORINFO) };
                if (GetMonitorInfo(desc.Monitor, &monInfo))
                    mi.isPrimary = (monInfo.dwFlags & MONITORINFOF_PRIMARY) != 0;

                result.push_back(mi);
            }
        }
        return result;
    }

    static const MonitorInfo* FindMonitorAt(const std::vector<MonitorInfo>& monitors, POINT pt) {
        for (const auto& m : monitors) {
            if (pt.x >= m.bounds.left && pt.x < m.bounds.right &&
                pt.y >= m.bounds.top  && pt.y < m.bounds.bottom)
                return &m;
        }
        return nullptr;
    }

    // ═══════════════════════════════════════════════════════════
    //  Initialize — dynamic index validation
    // ═══════════════════════════════════════════════════════════
    bool Initialize(int adapterIdx, int outputIdx) {
        std::lock_guard<std::mutex> lock(m_apiMtx);

        if (adapterIdx < 0 || outputIdx < 0) {
            Logger::Error("DXGI", "Negative indices (adapter=" +
                std::to_string(adapterIdx) + ", output=" +
                std::to_string(outputIdx) + ")");
            return false;
        }

        if (!CleanupInternal()) {
            Logger::Error("DXGI",
                "Initialize aborted: previous resources not released cleanly");
            return false;
        }

        m_AdapterIdx = adapterIdx;
        m_OutputIdx  = outputIdx;

        ComPtr<IDXGIFactory> factory = CreateFactory();
        if (!factory) return false;

        // Count adapters
        UINT adapterCount = 0;
        for (;;) {
            ComPtr<IDXGIAdapter> tmp;
            HRESULT hr = factory->EnumAdapters(adapterCount, tmp.GetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(hr)) break;
            ++adapterCount;
            if (adapterCount > 256) break;
        }
        if (static_cast<UINT>(adapterIdx) >= adapterCount) {
            Logger::Error("DXGI", "Adapter index " + std::to_string(adapterIdx) +
                " out of range (system has " + std::to_string(adapterCount) +
                " adapter(s))");
            return false;
        }

        ComPtr<IDXGIAdapter> adapter;
        HRESULT hr = factory->EnumAdapters(static_cast<UINT>(adapterIdx),
                                            adapter.GetAddressOf());
        if (FAILED(hr)) {
            Logger::Error("DXGI", "EnumAdapters(" + std::to_string(adapterIdx) +
                ") failed", hr);
            return false;
        }

        // Count outputs
        UINT outputCount = 0;
        for (;;) {
            ComPtr<IDXGIOutput> tmp;
            hr = adapter->EnumOutputs(outputCount, tmp.GetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(hr)) break;
            ++outputCount;
            if (outputCount > 256) break;
        }
        if (static_cast<UINT>(outputIdx) >= outputCount) {
            Logger::Error("DXGI", "Output index " + std::to_string(outputIdx) +
                " out of range (adapter " + std::to_string(adapterIdx) +
                " has " + std::to_string(outputCount) + " output(s))");
            return false;
        }

        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(static_cast<UINT>(outputIdx),
                                   output.GetAddressOf());
        if (FAILED(hr)) {
            Logger::Error("DXGI", "EnumOutputs(" + std::to_string(outputIdx) +
                ") failed", hr);
            return false;
        }

        ComPtr<IDXGIOutput1> output1;
        hr = output.As(&output1);
        if (FAILED(hr)) {
            Logger::Error("DXGI", "IDXGIOutput1 QI failed", hr);
            return false;
        }

        const UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        const D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };

        hr = E_FAIL;
        D3D_FEATURE_LEVEL chosenFL = D3D_FEATURE_LEVEL_11_0;
        for (auto fl : levels) {
            hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                createFlags, &fl, 1, D3D11_SDK_VERSION,
                m_Device.GetAddressOf(), &chosenFL, m_Context.GetAddressOf());
            if (SUCCEEDED(hr)) break;
        }
        if (FAILED(hr)) {
            Logger::Error("DXGI", "D3D11CreateDevice failed for all feature levels", hr);
            CleanupInternal();
            return false;
        }
        Logger::Info("DXGI", "D3D11 device created (FL=0x" +
            std::to_string(static_cast<unsigned>(chosenFL)) + ")");

        hr = output1->DuplicateOutput(m_Device.Get(), m_Duplication.GetAddressOf());
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE)
                Logger::Error("DXGI", "DuplicateOutput: max reached", hr);
            else if (hr == E_ACCESSDENIED)
                Logger::Error("DXGI", "DuplicateOutput: access denied", hr);
            else if (hr == DXGI_ERROR_UNSUPPORTED)
                Logger::Error("DXGI", "DuplicateOutput: unsupported", hr);
            else
                Logger::Error("DXGI", "DuplicateOutput failed", hr);
            CleanupInternal();
            return false;
        }

        if (!UpdateMonitorMetadataLocked(output.Get())) {
            CleanupInternal();
            return false;
        }

        m_initialized.store(true, std::memory_order_release);
        Logger::Info("DXGI",
            "Initialized adapter=" + std::to_string(adapterIdx) +
            " output=" + std::to_string(outputIdx) +
            " bounds=(" + std::to_string(m_MonitorBounds.left) + "," +
            std::to_string(m_MonitorBounds.top) + ")-(" +
            std::to_string(m_MonitorBounds.right) + "," +
            std::to_string(m_MonitorBounds.bottom) + ")" +
            " hdr10=" + (m_IsHDR10 ? "yes" : "no") +
            " scRGB=" + (m_IsScRGB ? "yes" : "no"));
        return true;
    }

    // ═══════════════════════════════════════════════════════════
    //  RecreateDuplication — fast path (keeps device)
    // ═══════════════════════════════════════════════════════════
    bool RecreateDuplication() {
        std::lock_guard<std::mutex> lock(m_apiMtx);
        if (!m_Device || !m_Context) return false;

        m_StagingTex.Reset();
        m_Duplication.Reset();

        ComPtr<IDXGIFactory> factory = CreateFactory();
        if (!factory) return false;

        ComPtr<IDXGIAdapter> adapter;
        HRESULT hr = factory->EnumAdapters(static_cast<UINT>(m_AdapterIdx),
                                            adapter.GetAddressOf());
        if (FAILED(hr)) return false;

        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(static_cast<UINT>(m_OutputIdx), output.GetAddressOf());
        if (FAILED(hr)) return false;

        ComPtr<IDXGIOutput1> output1;
        hr = output.As(&output1);
        if (FAILED(hr)) return false;

        hr = output1->DuplicateOutput(m_Device.Get(), m_Duplication.GetAddressOf());
        if (FAILED(hr)) {
            Logger::Error("DXGI", "RecreateDuplication failed", hr);
            return false;
        }

        if (!UpdateMonitorMetadataLocked(output.Get())) {
            m_Duplication.Reset();
            return false;
        }

        Logger::Info("DXGI",
            "Duplication recreated (device preserved); bounds=" +
            std::to_string(m_MonitorBounds.right - m_MonitorBounds.left) + "x" +
            std::to_string(m_MonitorBounds.bottom - m_MonitorBounds.top));
        return true;
    }

    // ═══════════════════════════════════════════════════════════
    //  GrabFrame
    // ═══════════════════════════════════════════════════════════
    Status GrabFrame(cv::Mat& outFrame) {
        std::lock_guard<std::mutex> lock(m_apiMtx);

        outFrame.release();

        if (!m_initialized.load(std::memory_order_acquire)) return Status::Error;
        if (!m_Duplication) return Status::Error;
        if (IsDeviceRemovedLocked()) return Status::DeviceRemoved;

        DXGI_OUTDUPL_FRAME_INFO info{};
        ComPtr<IDXGIResource> res;
        HRESULT hr = m_Duplication->AcquireNextFrame(1, &info, res.GetAddressOf());

        if (hr == DXGI_ERROR_WAIT_TIMEOUT)            return Status::Timeout;
        if (hr == DXGI_ERROR_ACCESS_LOST)             return Status::AccessLost;
        if (hr == DXGI_ERROR_DEVICE_REMOVED ||
            hr == DXGI_ERROR_DEVICE_RESET)            return Status::DeviceRemoved;
        if (hr == DXGI_ERROR_SESSION_DISCONNECTED)    return Status::SessionDisconnected;
        if (FAILED(hr)) {
            Logger::Error("DXGI", "AcquireNextFrame failed", hr);
            return Status::Error;
        }

        FrameGuard guard(m_Duplication.Get());

        if (info.AccumulatedFrames == 0) return Status::Timeout;
        if (!res) {
            Logger::Error("DXGI", "Null resource with AccFrames=" +
                std::to_string(info.AccumulatedFrames));
            return Status::Error;
        }

        ComPtr<ID3D11Texture2D> acquired;
        hr = res.As(&acquired);
        if (FAILED(hr)) {
            Logger::Error("DXGI", "ID3D11Texture2D QI failed", hr);
            return Status::Error;
        }

        D3D11_TEXTURE2D_DESC desc{};
        acquired->GetDesc(&desc);

        if (desc.Width == 0 || desc.Height == 0 ||
            desc.Width > 16384 || desc.Height > 16384) {
            Logger::Error("DXGI", "Invalid dimensions: " +
                std::to_string(desc.Width) + "x" + std::to_string(desc.Height));
            return Status::Error;
        }
        if (desc.SampleDesc.Count > 1) {
            Logger::Error("DXGI", "Unexpected MSAA texture");
            return Status::Error;
        }
        if (!IsFormatSupported(desc.Format)) {
            Logger::Error("DXGI", "Unsupported format: " + FormatName(desc.Format));
            return Status::Error;
        }

        if (!m_StagingTex ||
            m_StagingWidth  != desc.Width ||
            m_StagingHeight != desc.Height ||
            m_StagingFormat != desc.Format) {

            m_StagingTex.Reset();
            m_StagingWidth  = 0;
            m_StagingHeight = 0;
            m_StagingFormat = DXGI_FORMAT_UNKNOWN;

            D3D11_TEXTURE2D_DESC sdesc = desc;
            sdesc.Usage          = D3D11_USAGE_STAGING;
            sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            sdesc.BindFlags      = 0;
            sdesc.MiscFlags      = 0;
            sdesc.ArraySize      = 1;
            sdesc.MipLevels      = 1;
            sdesc.SampleDesc.Count   = 1;
            sdesc.SampleDesc.Quality = 0;

            hr = m_Device->CreateTexture2D(&sdesc, nullptr, m_StagingTex.GetAddressOf());
            if (FAILED(hr)) {
                if (hr == DXGI_ERROR_DEVICE_REMOVED || IsDeviceRemovedLocked())
                    return Status::DeviceRemoved;
                Logger::Error("DXGI", "CreateTexture2D failed", hr);
                return Status::Error;
            }
            m_StagingWidth  = desc.Width;
            m_StagingHeight = desc.Height;
            m_StagingFormat = desc.Format;
        }

        m_Context->CopyResource(m_StagingTex.Get(), acquired.Get());
        if (IsDeviceRemovedLocked()) {
            Logger::Error("DXGI", "Device removed during CopyResource");
            return Status::DeviceRemoved;
        }

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = m_Context->Map(m_StagingTex.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET ||
                IsDeviceRemovedLocked())
                return Status::DeviceRemoved;
            Logger::Error("DXGI", "Map failed", hr);
            return Status::Error;
        }
        MapGuard mapGuard(m_Context.Get(), m_StagingTex.Get());

        if (!mapped.pData) {
            Logger::Error("DXGI", "Map returned null pData");
            return Status::Error;
        }

        const UINT bpp = BytesPerPixel(desc.Format);
        const UINT64 expectedRowPitch = static_cast<UINT64>(desc.Width) * bpp;
        if (static_cast<UINT64>(mapped.RowPitch) < expectedRowPitch) {
            Logger::Error("DXGI", "RowPitch too small");
            return Status::Error;
        }

        try {
            if (!ConvertToBGRLocked(mapped.pData, mapped.RowPitch,
                                    static_cast<int>(desc.Width),
                                    static_cast<int>(desc.Height),
                                    desc.Format, outFrame)) {
                return Status::Error;
            }
            m_LastFormat = desc.Format;
        } catch (const std::exception& e) {
            Logger::Error("DXGI", std::string("Conversion: ") + e.what());
            outFrame.release();
            return Status::Error;
        }
        return Status::Success;
    }

    void Cleanup() {
        std::lock_guard<std::mutex> lock(m_apiMtx);
        if (!CleanupInternal()) {
            Logger::Warn("DXGI", "Cleanup: some resources still referenced");
        }
    }

    RECT GetMonitorBounds() const { std::lock_guard<std::mutex> l(m_apiMtx); return m_MonitorBounds; }
    int  GetAdapterIdx()    const { std::lock_guard<std::mutex> l(m_apiMtx); return m_AdapterIdx; }
    int  GetOutputIdx()     const { std::lock_guard<std::mutex> l(m_apiMtx); return m_OutputIdx; }
    bool IsHDR()            const { std::lock_guard<std::mutex> l(m_apiMtx); return m_IsHDR10 || m_IsScRGB; }
    DXGI_MODE_ROTATION GetRotation() const { std::lock_guard<std::mutex> l(m_apiMtx); return m_Rotation; }

    void SetScRGBExposure(float exp) {
        std::lock_guard<std::mutex> lock(m_apiMtx);
        if (std::isfinite(exp) && exp > 0.01f && exp <= 16.0f)
            m_ScRGBExposure = exp;
    }

private:
    static ComPtr<IDXGIFactory> CreateFactory() {
        ComPtr<IDXGIFactory1> factory1;
        HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1),
            reinterpret_cast<void**>(factory1.GetAddressOf()));
        if (SUCCEEDED(hr) && factory1) {
            ComPtr<IDXGIFactory> factory;
            if (SUCCEEDED(factory1.As(&factory))) return factory;
        }
        factory1.Reset();

        ComPtr<IDXGIFactory> factory;
        hr = CreateDXGIFactory(__uuidof(IDXGIFactory),
            reinterpret_cast<void**>(factory.GetAddressOf()));
        if (SUCCEEDED(hr) && factory) {
            Logger::Warn("DXGI", "Falling back to IDXGIFactory (Win7+)");
            return factory;
        }
        Logger::Error("DXGI", "CreateDXGIFactory also failed", hr);
        return nullptr;
    }

    bool IsDeviceRemovedLocked() const {
        if (!m_Device) return false;
        return m_Device->GetDeviceRemovedReason() != S_OK;
    }

    bool CleanupInternal() {
        m_initialized.store(false, std::memory_order_release);

        m_StagingTex.Reset();
        m_Duplication.Reset();
        m_Context.Reset();
        m_Device.Reset();

        m_StagingWidth  = 0;
        m_StagingHeight = 0;
        m_StagingFormat = DXGI_FORMAT_UNKNOWN;
        m_MonitorBounds = { 0, 0, 0, 0 };
        m_LastFormat    = DXGI_FORMAT_UNKNOWN;
        m_IsHDR10       = false;
        m_IsScRGB       = false;
        m_Rotation      = DXGI_MODE_ROTATION_IDENTITY;
        m_MaxLuminance  = 80.0f;

        bool ok = true;
        if (m_Device)      { Logger::Error("DXGI", "Cleanup: m_Device still referenced"); ok = false; }
        if (m_Context)     { Logger::Error("DXGI", "Cleanup: m_Context still referenced"); ok = false; }
        if (m_Duplication) { Logger::Error("DXGI", "Cleanup: m_Duplication still referenced"); ok = false; }
        if (m_StagingTex)  { Logger::Error("DXGI", "Cleanup: m_StagingTex still referenced"); ok = false; }
        return ok;
    }

    bool UpdateMonitorMetadataLocked(IDXGIOutput* output) {
        DXGI_OUTPUT_DESC desc{};
        HRESULT hr = output->GetDesc(&desc);
        if (FAILED(hr)) {
            Logger::Error("DXGI", "GetDesc failed", hr);
            return false;
        }
        m_MonitorBounds = desc.DesktopCoordinates;
        m_Rotation      = desc.Rotation;

        m_IsHDR10  = false;
        m_IsScRGB  = false;
        m_MaxLuminance = 80.0f;

        ComPtr<IDXGIOutput6> output6;
        if (SUCCEEDED(output->QueryInterface(IID_PPV_ARGS(&output6))) && output6) {
            DXGI_OUTPUT_DESC1 desc1{};
            if (SUCCEEDED(output6->GetDesc1(&desc1))) {
                m_MaxLuminance = static_cast<float>(desc1.MaxLuminance);
                switch (desc1.ColorSpace) {
                    case DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020:
                        m_IsHDR10 = true;
                        break;
                    case DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709:
                        m_IsScRGB = true;
                        break;
                    default:
                        break;
                }
            }
        }
        return true;
    }

    static UINT BytesPerPixel(DXGI_FORMAT fmt) {
        switch (fmt) {
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R10G10B10A2_UNORM:  return 4;
            case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
            default:                              return 4;
        }
    }

    static bool IsFormatSupported(DXGI_FORMAT fmt) {
        switch (fmt) {
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R10G10B10A2_UNORM:
            case DXGI_FORMAT_R16G16B16A16_FLOAT: return true;
            default:                              return false;
        }
    }

    static const char* FormatName(DXGI_FORMAT fmt) {
        switch (fmt) {
            case DXGI_FORMAT_B8G8R8A8_UNORM:     return "BGRA8";
            case DXGI_FORMAT_R8G8B8A8_UNORM:     return "RGBA8";
            case DXGI_FORMAT_R10G10B10A2_UNORM:  return "HDR10";
            case DXGI_FORMAT_R16G16B16A16_FLOAT: return "scRGB";
            default:                              return "Unknown";
        }
    }

    bool ConvertToBGRLocked(const void* pData, UINT rowPitch,
                            int width, int height,
                            DXGI_FORMAT fmt, cv::Mat& outBGR) {
        cv::Mat unrotated;

        if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM || fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
            cv::Mat raw(height, width, CV_8UC4,
                        const_cast<void*>(pData), static_cast<size_t>(rowPitch));
            const int code = (fmt == DXGI_FORMAT_B8G8R8A8_UNORM)
                           ? cv::COLOR_BGRA2BGR : cv::COLOR_RGBA2BGR;
            cv::cvtColor(raw, unrotated, code);
        }
        else if (fmt == DXGI_FORMAT_R10G10B10A2_UNORM) {
            // ✅ HDR10: R[0-9] G[10-19] B[20-29] — take top 8 bits per channel
            const float*   pqLut   = GetPQtoLinearLUT();
            const uint8_t* srgbLut = GetSRGBLUT();
            const float    exposure = m_ScRGBExposure;

            unrotated.create(height, width, CV_8UC3);
            for (int y = 0; y < height; ++y) {
                const uint32_t* srcRow = reinterpret_cast<const uint32_t*>(
                    static_cast<const uint8_t*>(pData) + (size_t)y * rowPitch);
                uint8_t* dstRow = unrotated.ptr<uint8_t>(y);
                for (int x = 0; x < width; ++x) {
                    const uint32_t px = srcRow[x];
                    const uint32_t r10 = (px >>  0) & 0x3FF;
                    const uint32_t g10 = (px >> 10) & 0x3FF;
                    const uint32_t b10 = (px >> 20) & 0x3FF;

                    float rLin = pqLut[r10] * exposure;
                    float gLin = pqLut[g10] * exposure;
                    float bLin = pqLut[b10] * exposure;

                    dstRow[x * 3 + 0] = LinearToSRGB8(bLin, srgbLut);
                    dstRow[x * 3 + 1] = LinearToSRGB8(gLin, srgbLut);
                    dstRow[x * 3 + 2] = LinearToSRGB8(rLin, srgbLut);
                }
            }
        }
        else if (fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
            const uint8_t* srgbLut = GetSRGBLUT();
            const float    exposure = m_ScRGBExposure;

            unrotated.create(height, width, CV_8UC3);
            for (int y = 0; y < height; ++y) {
                const uint16_t* srcRow = reinterpret_cast<const uint16_t*>(
                    static_cast<const uint8_t*>(pData) + (size_t)y * rowPitch);
                uint8_t* dstRow = unrotated.ptr<uint8_t>(y);
                for (int x = 0; x < width; ++x) {
                    const float rF = HalfToFloat(srcRow[x * 4 + 0]) * exposure;
                    const float gF = HalfToFloat(srcRow[x * 4 + 1]) * exposure;
                    const float bF = HalfToFloat(srcRow[x * 4 + 2]) * exposure;
                    dstRow[x * 3 + 0] = LinearToSRGB8(bF, srgbLut);
                    dstRow[x * 3 + 1] = LinearToSRGB8(gF, srgbLut);
                    dstRow[x * 3 + 2] = LinearToSRGB8(rF, srgbLut);
                }
            }
        }
        else return false;

        switch (m_Rotation) {
            case DXGI_MODE_ROTATION_ROTATE90:
                cv::rotate(unrotated, outBGR, cv::ROTATE_90_CLOCKWISE);
                break;
            case DXGI_MODE_ROTATION_ROTATE180:
                cv::rotate(unrotated, outBGR, cv::ROTATE_180);
                break;
            case DXGI_MODE_ROTATION_ROTATE270:
                cv::rotate(unrotated, outBGR, cv::ROTATE_90_COUNTERCLOCKWISE);
                break;
            default:
                outBGR = unrotated;
                break;
        }
        return true;
    }

    static float PQtoLinear(uint32_t val10) {
        const float N = static_cast<float>(val10) / 1023.0f;
        constexpr float m1 = 2610.0f / 16384.0f;
        constexpr float m2 = 2523.0f / 4096.0f * 128.0f;
        constexpr float c1 = 3424.0f / 4096.0f;
        constexpr float c2 = 2413.0f / 4096.0f * 32.0f;
        constexpr float c3 = 2392.0f / 4096.0f * 32.0f;

        float Np = std::pow(N, 1.0f / m2);
        float num = std::max(Np - c1, 0.0f);
        float den = c2 - c3 * Np;
        if (den <= 0.0f) return 0.0f;
        float L_nits = std::pow(num / den, 1.0f / m1) * 10000.0f;
        return L_nits / 100.0f;
    }

    static const float* GetPQtoLinearLUT() {
        static float lut[1024];
        static std::once_flag flag;
        std::call_once(flag, []() {
            for (int i = 0; i < 1024; ++i)
                lut[i] = PQtoLinear(static_cast<uint32_t>(i));
        });
        return lut;
    }

    static const uint8_t* GetSRGBLUT() {
        static uint8_t lut[1024];
        static std::once_flag flag;
        std::call_once(flag, []() {
            for (int i = 0; i < 1024; ++i) {
                float v = static_cast<float>(i) / 1023.0f;
                float s = (v <= 0.0031308f)
                    ? (12.92f * v)
                    : (1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f);
                int iv = static_cast<int>(s * 255.0f + 0.5f);
                if (iv < 0)   iv = 0;
                if (iv > 255) iv = 255;
                lut[i] = static_cast<uint8_t>(iv);
            }
        });
        return lut;
    }

    static uint8_t LinearToSRGB8(float v, const uint8_t* lut) {
        if (!(v > 0.0f)) v = 0.0f;
        if (v > 1.0f)    v = 1.0f;
        int idx = static_cast<int>(v * 1023.0f + 0.5f);
        if (idx < 0)    idx = 0;
        if (idx > 1023) idx = 1023;
        return lut[idx];
    }

    static float HalfToFloat(uint16_t h) noexcept {
        const uint32_t sign = (h >> 15) & 0x1;
        const uint32_t exp  = (h >> 10) & 0x1F;
        const uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            float f = static_cast<float>(mant) / 1024.0f;
            f *= std::pow(2.0f, -14.0f);
            return sign ? -f : f;
        }
        if (exp == 0x1F) {
            if (mant == 0)
                return sign ? -std::numeric_limits<float>::infinity()
                            :  std::numeric_limits<float>::infinity();
            return std::numeric_limits<float>::quiet_NaN();
        }
        float f = 1.0f + static_cast<float>(mant) / 1024.0f;
        f *= std::pow(2.0f, static_cast<int>(exp) - 15);
        return sign ? -f : f;
    }

    class FrameGuard {
        IDXGIOutputDuplication* d_;
    public:
        explicit FrameGuard(IDXGIOutputDuplication* dup) : d_(dup) {}
        ~FrameGuard() {
            if (!d_) return;
            try {
                HRESULT hr = d_->ReleaseFrame();
                if (FAILED(hr) && hr != DXGI_ERROR_INVALID_CALL)
                    Logger::Error("DXGI", "ReleaseFrame failed", hr);
            } catch (...) {}
        }
        FrameGuard(const FrameGuard&) = delete;
        FrameGuard& operator=(const FrameGuard&) = delete;
    };

    class MapGuard {
        ID3D11DeviceContext* c_;
        ID3D11Resource*      r_;
    public:
        MapGuard(ID3D11DeviceContext* ctx, ID3D11Resource* res) : c_(ctx), r_(res) {}
        ~MapGuard() {
            if (c_ && r_) { try { c_->Unmap(r_, 0); } catch (...) {} }
        }
        MapGuard(const MapGuard&) = delete;
        MapGuard& operator=(const MapGuard&) = delete;
    };

    mutable std::mutex m_apiMtx;

    ComPtr<ID3D11Device>           m_Device;
    ComPtr<ID3D11DeviceContext>    m_Context;
    ComPtr<IDXGIOutputDuplication> m_Duplication;
    ComPtr<ID3D11Texture2D>        m_StagingTex;

    RECT        m_MonitorBounds = { 0, 0, 0, 0 };
    UINT        m_StagingWidth  = 0;
    UINT        m_StagingHeight = 0;
    DXGI_FORMAT m_StagingFormat = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT m_LastFormat    = DXGI_FORMAT_UNKNOWN;
    int         m_AdapterIdx    = 0;
    int         m_OutputIdx     = 0;

    DXGI_MODE_ROTATION m_Rotation     = DXGI_MODE_ROTATION_IDENTITY;
    bool               m_IsHDR10      = false;
    bool               m_IsScRGB      = false;
    float              m_MaxLuminance = 80.0f;
    float              m_ScRGBExposure = 1.0f;

    std::atomic<bool>  m_initialized{false};
};
