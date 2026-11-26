#pragma once
#include <string>
#include <string_view>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <set>
#include <vector>
#include <cctype>
#include <climits>
#include <algorithm>
#include <iomanip>
#include <charconv>
#include <shared_mutex>
#include <memory>
#include "Logger.h"

/// Thread-safety contract:
///   • Load/Reload/assignment → EXCLUSIVE lock on *this
///   • Snapshot() → SHARED lock, returns independent copy
///   • Direct field access (cfg.field) → safe only if no concurrent Load
///   • Missing keys after Load() → reset to defaults (full replacement)
struct Config {
    int         monitorMode      = 0;
    int         cropMargin       = 40;
    std::string gameProcess      = "";
    int         useGPU           = 1;
    std::string modelPath        = "best.onnx";
    int         inputSize        = 320;
    float       confThreshold    = 0.40f;
    float       nmsThreshold     = 0.45f;
    std::string modelType        = "yolov8";
    int         numClasses       = 1;
    int         circleRadius     = 150;
    float       aimSpeed         = 1.0f;
    int         headOffsetX      = 0;
    int         headOffsetY      = 0;
    float       predictionFactor = 1.0f;
    float       fireThresholdPx  = 8.0f;
    bool        strictMode       = false;

    static constexpr size_t kMaxLineLength = 4096;
    static constexpr size_t kMaxKeys       = 512;
    static constexpr size_t kMaxFileSize   = 1024 * 1024;

    enum class StrictOverride { FromFile, ForceOn, ForceOff };

    std::shared_ptr<std::shared_mutex> m_lock;

    Config() : m_lock(std::make_shared<std::shared_mutex>()) {}

    Config(const Config& o) {
        std::shared_lock<std::shared_mutex> lk(*o.m_lock);
        CopyFieldsFrom(o);
        m_lock = std::make_shared<std::shared_mutex>();
    }

    Config& operator=(const Config& o) {
        if (this == &o) return *this;
        if (m_lock == o.m_lock) return *this;
        Config tmp(o);
        std::unique_lock<std::shared_mutex> lk(*m_lock);
        CopyFieldsFrom(tmp);
        return *this;
    }

    Config(Config&&) = delete;
    Config& operator=(Config&&) = delete;

    Config Snapshot() const { return *this; }

    // ──────────────── Parsers ────────────────
    static std::string Trim(std::string_view s) {
        auto start = s.find_first_not_of(" \t\r\n");
        if (start == std::string_view::npos) return {};
        auto end = s.find_last_not_of(" \t\r\n");
        return std::string(s.substr(start, end - start + 1));
    }

    static bool EndsWithNoCase(const std::string& s, const std::string& suffix) {
        if (s.size() < suffix.size()) return false;
        const size_t offset = s.size() - suffix.size();
        for (size_t i = 0; i < suffix.size(); ++i)
            if (std::tolower(static_cast<unsigned char>(s[offset + i])) !=
                std::tolower(static_cast<unsigned char>(suffix[i]))) return false;
        return true;
    }

    static std::string ToLower(std::string s) {
        for (char& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    bool ParseInt(std::string_view s, int& out) const {
        std::string t = Trim(s);
        if (t.empty()) return false;
        if (t[0] == '+') { t.erase(0, 1); if (t.empty()) return false; }
        int v = 0;
        auto r = std::from_chars(t.data(), t.data() + t.size(), v);
        if (r.ec != std::errc{} || r.ptr != t.data() + t.size()) return false;
        out = v;
        return true;
    }

    bool ParseFloat(std::string_view s, float& out) const {
        std::string t = Trim(s);
        if (t.empty()) return false;
        if (t[0] == '+') { t.erase(0, 1); if (t.empty()) return false; }
        if (t.find(',') != std::string::npos) return false;
        float v = 0.0f;
        auto r = std::from_chars(t.data(), t.data() + t.size(), v);
        if (r.ec != std::errc{} || r.ptr != t.data() + t.size()) return false;
        if (!std::isfinite(v)) return false;
        out = v;
        return true;
    }

    bool ParseBool(std::string_view s, bool& out) const {
        std::string t = Trim(s);
        if (t.empty()) return false;
        std::string lower;
        lower.reserve(t.size());
        for (char c : t) lower.push_back(
            static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        if (lower == "true" || lower == "1" || lower == "yes" || lower == "on")  { out = true;  return true; }
        if (lower == "false" || lower == "0" || lower == "no" || lower == "off") { out = false; return true; }
        return false;
    }

    // ──────────────── Validation ────────────────
    struct ValidationIssue {
        bool        fatal;
        std::string message;
    };

    bool Validate(std::vector<ValidationIssue>& issues) const {
        auto warn  = [&](std::string m) { issues.push_back({false, std::move(m)}); };
        auto fatal = [&](std::string m) { issues.push_back({true,  std::move(m)}); };
        bool ok = true;

        if (monitorMode < 0 || monitorMode > 1) {
            fatal("monitor_mode must be 0 or 1");
            ok = false;
        }
        if (cropMargin < 0 || cropMargin > 1000) {
            fatal("crop_margin must be in [0, 1000]");
            ok = false;
        } else if (cropMargin < 20) {
            warn("crop_margin < 20 may reduce detection quality");
        }
        if (useGPU < 0 || useGPU > 1) {
            fatal("use_gpu must be 0 or 1");
            ok = false;
        }
        if (inputSize < 32 || inputSize > 4096 || inputSize % 32 != 0) {
            fatal("input_size must be in [32, 4096] and multiple of 32");
            ok = false;
        } else if (inputSize < 64) {
            warn("input_size < 64 may produce poor detection quality");
        }
        static const std::set<std::string> validTypes = {
            "yolov5", "yolov7", "yolov8", "yolov10", "yolov11"
        };
        if (modelType.empty()) {
            fatal("model_type is empty");
            ok = false;
        } else if (validTypes.find(modelType) == validTypes.end()) {
            fatal("Unsupported model_type '" + modelType + "'");
            ok = false;
        }
        if (numClasses <= 0 || numClasses > 1000) {
            fatal("num_classes must be in [1, 1000]");
            ok = false;
        } else if (modelType == "yolov10" && numClasses != 1) {
            warn("yolov10 ignores num_classes (tensor encodes class)");
        }
        if (!(confThreshold > 0.0f && confThreshold <= 1.0f)) {
            fatal("confidence must be in (0, 1]");
            ok = false;
        }
        if (!(nmsThreshold > 0.0f && nmsThreshold <= 1.0f)) {
            fatal("nms must be in (0, 1]");
            ok = false;
        }
        if (circleRadius < 10 || circleRadius > 1000) {
            fatal("circle_radius must be in [10, 1000]");
            ok = false;
        }
        if (!(aimSpeed > 0.01f && aimSpeed <= 10.0f)) {
            fatal("aim_speed must be in (0.01, 10]");
            ok = false;
        }
        if (headOffsetX < -500 || headOffsetX > 500 ||
            headOffsetY < -500 || headOffsetY > 500) {
            fatal("head_offset_x/y must be in [-500, 500]");
            ok = false;
        }
        if (!(predictionFactor >= 0.0f && predictionFactor <= 5.0f)) {
            fatal("prediction must be in [0, 5]");
            ok = false;
        }
        if (!(fireThresholdPx >= 0.0f && fireThresholdPx <= 200.0f)) {
            fatal("fire_threshold must be in [0, 200]");
            ok = false;
        }
        if (!gameProcess.empty()) {
            if (gameProcess.find('/') != std::string::npos ||
                gameProcess.find('\\') != std::string::npos) {
                fatal("game_process must be a filename, not a path");
                ok = false;
            } else {
                if (gameProcess.size() > 260) {
                    fatal("game_process name too long");
                    ok = false;
                }
                for (char c : gameProcess) {
                    unsigned char uc = static_cast<unsigned char>(c);
                    if (!(std::isalnum(uc) || c == '_' || c == '-' ||
                          c == '.' || c == ' ')) {
                        fatal("game_process contains invalid character");
                        ok = false;
                        break;
                    }
                }
                if (!EndsWithNoCase(gameProcess, ".exe"))
                    warn("game_process '" + gameProcess + "' doesn't end with .exe");
            }
        } else if (monitorMode == 0) {
            warn("monitor_mode=0 but game_process is empty");
        }
        if (modelPath.empty()) {
            fatal("model path is empty");
            ok = false;
        } else if (!EndsWithNoCase(modelPath, ".onnx")) {
            fatal("model file must have .onnx extension");
            ok = false;
        }
        return ok;
    }

    // ──────────────── Load / Reload ────────────────
    bool Load(const std::filesystem::path& configFilePath,
              StrictOverride strictOverride = StrictOverride::FromFile) {
        Config temp;   // fresh defaults
        if (!temp.LoadInternal(configFilePath, strictOverride)) return false;
        std::unique_lock<std::shared_mutex> lk(*m_lock);
        CopyFieldsFrom(temp);
        return true;
    }

    bool Reload(const std::filesystem::path& configFilePath,
                StrictOverride strictOverride = StrictOverride::FromFile) {
        return Load(configFilePath, strictOverride);
    }

private:
    void CopyFieldsFrom(const Config& o) {
        monitorMode      = o.monitorMode;
        cropMargin       = o.cropMargin;
        gameProcess      = o.gameProcess;
        useGPU           = o.useGPU;
        modelPath        = o.modelPath;
        inputSize        = o.inputSize;
        confThreshold    = o.confThreshold;
        nmsThreshold     = o.nmsThreshold;
        modelType        = o.modelType;
        numClasses       = o.numClasses;
        circleRadius     = o.circleRadius;
        aimSpeed         = o.aimSpeed;
        headOffsetX      = o.headOffsetX;
        headOffsetY      = o.headOffsetY;
        predictionFactor = o.predictionFactor;
        fireThresholdPx  = o.fireThresholdPx;
        strictMode       = o.strictMode;
    }

    static std::string PathToUTF8(const std::filesystem::path& p) noexcept {
        try {
#if defined(__cpp_char8_t) && __cpp_char8_t >= 201811L
            auto u8 = p.u8string();
            return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
#else
            return p.u8string();
#endif
        } catch (...) {
            try { return p.string(); }
            catch (...) { return "<unrepresentable-path>"; }
        }
    }

    static size_t FindInlineCommentStart(const std::string& line) noexcept {
        bool inEscape = false;
        for (size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            if (inEscape) { inEscape = false; continue; }
            if (c == '\\') { inEscape = true; continue; }
            if ((c == '#' || c == ';') && i > 0 &&
                (line[i-1] == ' ' || line[i-1] == '\t')) {
                return i;
            }
        }
        return std::string::npos;
    }

    static std::string Unescape(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        bool inEscape = false;
        for (char c : s) {
            if (inEscape) { out.push_back(c); inEscape = false; }
            else if (c == '\\') inEscape = true;
            else out.push_back(c);
        }
        if (inEscape) out.push_back('\\');
        return out;
    }

    bool LoadInternal(const std::filesystem::path& configFilePath,
                      StrictOverride strictOverride) {
        namespace fs = std::filesystem;

        try {
            std::error_code ec;
            auto sz = fs::file_size(configFilePath, ec);
            if (ec) {
                Logger::Fatal("Config", "Cannot stat config: " +
                    PathToUTF8(configFilePath) + " (" + ec.message() + ")");
                return false;
            }
            if (sz > kMaxFileSize) {
                Logger::Fatal("Config", "Config too large (" +
                    std::to_string(sz) + ")");
                return false;
            }
        } catch (const std::exception& e) {
            Logger::Fatal("Config", std::string("Stat: ") + e.what());
            return false;
        } catch (...) {
            Logger::Fatal("Config", "Stat unknown");
            return false;
        }

        std::ifstream file(configFilePath);
        if (!file.is_open()) {
            Logger::Fatal("Config", "Cannot open: " + PathToUTF8(configFilePath));
            return false;
        }

        std::vector<std::string> localUnknownKeys;
        std::vector<std::string> localWarnings;
        std::string line;
        std::set<std::string> seenKeys;
        int lineNum = 0;
        size_t keyCount = 0;
        size_t bytesRead = 0;
        bool firstLine = true;

        while (std::getline(file, line)) {
            ++lineNum;
            bytesRead += line.size() + 1;
            if (bytesRead > kMaxFileSize) {
                Logger::Fatal("Config", "File exceeded size limit at line " +
                    std::to_string(lineNum));
                return false;
            }

            if (firstLine) {
                firstLine = false;
                if (line.size() >= 3 &&
                    static_cast<unsigned char>(line[0]) == 0xEF &&
                    static_cast<unsigned char>(line[1]) == 0xBB &&
                    static_cast<unsigned char>(line[2]) == 0xBF) {
                    line.erase(0, 3);
                }
            }

            if (line.size() > kMaxLineLength) {
                Logger::Fatal("Config", "Line " + std::to_string(lineNum) +
                    ": exceeds max length");
                return false;
            }

            line = Trim(line);
            if (line.empty()) continue;
            if (line[0] == '#' || line[0] == ';') continue;

            if (line[0] == '[') {
                if (line.size() < 2 || line.back() != ']') {
                    localWarnings.push_back("Line " + std::to_string(lineNum) +
                        ": unterminated section (ignored)");
                    continue;
                }
                std::string sectionName = Trim(line.substr(1, line.size() - 2));
                if (sectionName.empty()) {
                    localWarnings.push_back("Line " + std::to_string(lineNum) +
                        ": empty section name (ignored)");
                }
                continue;
            }

            size_t commentPos = FindInlineCommentStart(line);
            if (commentPos != std::string::npos) {
                line = Trim(line.substr(0, commentPos));
            }
            if (line.empty()) continue;

            auto eq = line.find('=');
            if (eq == std::string::npos) {
                Logger::Fatal("Config", "Line " + std::to_string(lineNum) +
                    ": missing '='");
                return false;
            }
            std::string key = Trim(line.substr(0, eq));
            std::string val = Unescape(Trim(line.substr(eq + 1)));

            if (key.empty()) {
                Logger::Fatal("Config", "Line " + std::to_string(lineNum) +
                    ": empty key");
                return false;
            }
            if (seenKeys.count(key)) {
                Logger::Fatal("Config", "Line " + std::to_string(lineNum) +
                    ": duplicate key '" + key + "'");
                return false;
            }
            seenKeys.insert(key);
            if (++keyCount > kMaxKeys) {
                Logger::Fatal("Config", "Too many keys");
                return false;
            }

            bool ok = true;
            std::string failHint;

            if      (key == "monitor_mode")    ok = ParseInt(val, monitorMode);
            else if (key == "crop_margin")     ok = ParseInt(val, cropMargin);
            else if (key == "game_process")    gameProcess = val;
            else if (key == "use_gpu")         ok = ParseInt(val, useGPU);
            else if (key == "model")           modelPath = val;
            else if (key == "input_size")      ok = ParseInt(val, inputSize);
            else if (key == "confidence")      ok = ParseFloat(val, confThreshold);
            else if (key == "nms")             ok = ParseFloat(val, nmsThreshold);
            else if (key == "model_type")      modelType = ToLower(val);
            else if (key == "num_classes")     ok = ParseInt(val, numClasses);
            else if (key == "circle_radius")   ok = ParseInt(val, circleRadius);
            else if (key == "aim_speed")       ok = ParseFloat(val, aimSpeed);
            else if (key == "head_offset_x")   ok = ParseInt(val, headOffsetX);
            else if (key == "head_offset_y")   ok = ParseInt(val, headOffsetY);
            else if (key == "prediction")      ok = ParseFloat(val, predictionFactor);
            else if (key == "fire_threshold")  ok = ParseFloat(val, fireThresholdPx);
            else if (key == "strict")          ok = ParseBool(val, strictMode);
            else localUnknownKeys.push_back("Line " + std::to_string(lineNum) +
                ": unknown key '" + key + "'");

            if (!ok && val.find(',') != std::string::npos &&
                (key == "confidence" || key == "nms" || key == "aim_speed" ||
                 key == "prediction" || key == "fire_threshold")) {
                failHint = " — use '.' as decimal separator";
            }

            if (!ok) {
                Logger::Fatal("Config", "Line " + std::to_string(lineNum) +
                    ": failed to parse '" + key + "' (value: '" + val + "')" +
                    failHint);
                return false;
            }
        }

        if (file.bad()) {
            Logger::Fatal("Config", "I/O error reading: " +
                PathToUTF8(configFilePath));
            return false;
        }

        if (strictOverride == StrictOverride::ForceOn)       strictMode = true;
        else if (strictOverride == StrictOverride::ForceOff) strictMode = false;

        if (strictMode && !localUnknownKeys.empty()) {
            for (const auto& w : localUnknownKeys)
                Logger::Fatal("Config", w + " (strict mode)");
            return false;
        }
        for (const auto& w : localUnknownKeys)
            localWarnings.push_back(w + " (ignored)");

        // Path resolution — UTF-8 strict, no CP_ACP fallback
        try {
            if (modelPath.empty()) {
                Logger::Fatal("Config", "model path is empty");
                return false;
            }

            fs::path modelP;
            try {
                modelP = fs::u8path(modelPath);
            } catch (const std::exception& e) {
                Logger::Fatal("Config",
                    "Invalid UTF-8 in model path: " + std::string(e.what()));
                return false;
            }

            fs::path resolved;
            if (modelP.is_relative()) {
                fs::path configAbs = fs::absolute(configFilePath);
                fs::path configDir = configAbs.parent_path();
                if (configDir.empty()) configDir = fs::current_path();
                resolved = (configDir / modelP).lexically_normal();
            } else {
                resolved = modelP.lexically_normal();
            }

            std::error_code ec;
            if (!fs::exists(resolved, ec) || ec) {
                Logger::Fatal("Config", "Model missing: " + PathToUTF8(resolved));
                return false;
            }
            if (!fs::is_regular_file(resolved, ec) || ec) {
                Logger::Fatal("Config", "Model not regular: " + PathToUTF8(resolved));
                return false;
            }
            auto sz = fs::file_size(resolved, ec);
            if (ec) {
                Logger::Fatal("Config", "Cannot stat model: " + PathToUTF8(resolved));
                return false;
            }
            if (sz < 1024) {
                Logger::Fatal("Config", "Model too small (" +
                    std::to_string(sz) + " bytes); corrupt");
                return false;
            }

            try {
#if defined(__cpp_char8_t) && __cpp_char8_t >= 201811L
                auto u8 = resolved.u8string();
                modelPath.assign(reinterpret_cast<const char*>(u8.data()), u8.size());
#else
                modelPath = resolved.u8string();
#endif
            } catch (const std::exception& e) {
                Logger::Fatal("Config",
                    "Cannot convert model path: " + std::string(e.what()));
                return false;
            }
        } catch (const fs::filesystem_error& e) {
            Logger::Fatal("Config", "FS error: " + std::string(e.what()));
            return false;
        } catch (const std::exception& e) {
            Logger::Fatal("Config", "Path: " + std::string(e.what()));
            return false;
        } catch (...) {
            Logger::Fatal("Config", "Path unknown exception");
            return false;
        }

        std::vector<ValidationIssue> issues;
        bool validOk = Validate(issues);
        for (const auto& issue : issues) {
            if (issue.fatal) Logger::Fatal("Config", issue.message);
            else             Logger::Warn ("Config", issue.message);
        }
        for (const auto& w : localWarnings) Logger::Warn("Config", w);
        if (!validOk) return false;

        std::ostringstream summary;
        summary << "type=" << modelType
                << ", input=" << inputSize
                << ", classes=" << numClasses
                << ", conf=" << std::fixed << std::setprecision(2) << confThreshold
                << ", nms=" << nmsThreshold
                << ", radius=" << circleRadius
                << ", margin=" << cropMargin
                << ", speed=" << aimSpeed
                << ", pred=" << predictionFactor
                << ", fire=" << std::fixed << std::setprecision(1) << fireThresholdPx
                << ", gpu=" << useGPU
                << ", monitor=" << monitorMode
                << ", strict=" << (strictMode ? "on" : "off");
        Logger::Info("Config", "Loaded: " + summary.str());
        return true;
    }
};
