#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include "Logger.h"
#include "Config.h"

struct Detection { cv::Rect box; float confidence; int classId; };

struct LetterboxInfo {
    float scaleX    = 1.0f;
    float scaleY    = 1.0f;
    float padX      = 0.0f;
    float padY      = 0.0f;
    int   inputSize = 0;
};

struct TensorLayout {
    bool valid         = false;
    bool featureMajor  = true;
    int  features      = 0;
    int  candidates    = 0;
};

class YoloDetector {
public:
    enum class DetectStatus {
        Ok,
        Skipped,
        InvalidFrame,
        PreprocessFailed,
        InferenceFailed,
        ParseFailed,
    };

    DetectStatus GetLastStatus() const { return m_lastStatus.load(); }

    static TensorLayout AnalyzeTensorStatic(const cv::Mat& out,
                                            const std::string& modelType,
                                            int numClasses) {
        TensorLayout layout;
        if (out.empty() || out.dims != 3 || out.size[0] != 1) return layout;
        if (out.type() != CV_32F || !out.isContinuous()) return layout;

        int64_t totalElements =
            static_cast<int64_t>(out.size[1]) * static_cast<int64_t>(out.size[2]);
        constexpr int64_t kMaxElements = 100LL * 1024 * 1024;
        if (totalElements <= 0 || totalElements > kMaxElements) return layout;

        int expectedFeatures = 0;
        if      (modelType == "yolov10")                          expectedFeatures = 6;
        else if (modelType == "yolov5" || modelType == "yolov7")  expectedFeatures = 5 + numClasses;
        else if (modelType == "yolov8" || modelType == "yolov11") expectedFeatures = 4 + numClasses;
        else return layout;

        const int dim1 = out.size[1], dim2 = out.size[2];
        const bool dim1IsFeatures = (dim1 == expectedFeatures);
        const bool dim2IsFeatures = (dim2 == expectedFeatures);
        if (dim1IsFeatures == dim2IsFeatures) return layout;

        layout.featureMajor = dim1IsFeatures;
        layout.features     = expectedFeatures;
        layout.candidates   = dim1IsFeatures ? dim2 : dim1;

        if (layout.candidates <= 0 || layout.candidates > 200000) return layout;
        if (layout.features   <= 0 || layout.features   > 1024)   return layout;
        layout.valid = true;
        return layout;
    }

    bool Load(const Config& cfg) {
        if (cfg.inputSize <= 0 || cfg.inputSize > 4096 || cfg.inputSize % 32 != 0) {
            Logger::Fatal("YOLO", "Invalid inputSize=" +
                std::to_string(cfg.inputSize));
            return false;
        }
        if (cfg.numClasses <= 0 || cfg.numClasses > 10000) {
            Logger::Fatal("YOLO", "Invalid numClasses=" +
                std::to_string(cfg.numClasses));
            return false;
        }
        if (!(cfg.confThreshold > 0.0f && cfg.confThreshold <= 1.0f) ||
            !std::isfinite(cfg.confThreshold)) {
            Logger::Fatal("YOLO", "Invalid confThreshold=" +
                std::to_string(cfg.confThreshold));
            return false;
        }
        if (!(cfg.nmsThreshold > 0.0f && cfg.nmsThreshold <= 1.0f) ||
            !std::isfinite(cfg.nmsThreshold)) {
            Logger::Fatal("YOLO", "Invalid nmsThreshold=" +
                std::to_string(cfg.nmsThreshold));
            return false;
        }
        if (cfg.modelPath.empty()) {
            Logger::Fatal("YOLO", "Empty model path");
            return false;
        }

        cv::dnn::Net tempNet;
        try {
            tempNet = cv::dnn::readNetFromONNX(cfg.modelPath);
            if (tempNet.empty()) {
                Logger::Fatal("YOLO", "Empty network");
                return false;
            }
        } catch (const std::exception& e) {
            Logger::Fatal("YOLO", "ONNX Load Error: " + std::string(e.what()));
            return false;
        }

        const bool useOpenCL = (cfg.useGPU == 1 && cv::ocl::haveOpenCL());
        tempNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        tempNet.setPreferableTarget(useOpenCL ? cv::dnn::DNN_TARGET_OPENCL
                                              : cv::dnn::DNN_TARGET_CPU);

        std::vector<cv::String> names;
        try {
            names = tempNet.getUnconnectedOutLayersNames();
        } catch (const std::exception& e) {
            Logger::Fatal("YOLO", "getUnconnectedOutLayersNames: " +
                std::string(e.what()));
            return false;
        }
        if (names.empty()) {
            Logger::Fatal("YOLO", "Model has no output layers");
            return false;
        }
        for (const auto& n : names) {
            if (n.empty()) {
                Logger::Fatal("YOLO", "Empty output layer name");
                return false;
            }
        }

        bool openCLActive = useOpenCL;
        TensorLayout found;
        std::vector<cv::Mat> outs;
        bool validationOk = false;

        for (int attempt = 0; attempt < 2 && !validationOk; ++attempt) {
            try {
                cv::Mat dummy(cfg.inputSize, cfg.inputSize, CV_8UC3, cv::Scalar(0));
                cv::Mat blob;
                cv::dnn::blobFromImage(dummy, blob, 1.0 / 255.0,
                    cv::Size(cfg.inputSize, cfg.inputSize), cv::Scalar(), true, false);
                tempNet.setInput(blob);

                outs.clear();
                if (names.size() > 1) tempNet.forward(outs, names);
                else {
                    cv::Mat single = tempNet.forward();
                    if (!single.empty()) outs.push_back(single);
                }
                validationOk = true;
            } catch (const std::exception& e) {
                if (openCLActive && attempt == 0) {
                    Logger::Warn("YOLO", std::string("OpenCL validation failed: ") +
                        e.what() + " — retrying on CPU");
                    try {
                        tempNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                    } catch (const std::exception& e2) {
                        Logger::Fatal("YOLO", std::string("CPU fallback failed: ") +
                            e2.what());
                        return false;
                    }
                    openCLActive = false;
                    continue;
                }
                Logger::Fatal("YOLO", "Validation exception: " + std::string(e.what()));
                return false;
            }
        }

        if (!validationOk || outs.empty()) {
            Logger::Fatal("YOLO", "Model produced no outputs on test input");
            return false;
        }

        for (const auto& o : outs) {
            TensorLayout tl = AnalyzeTensorStatic(o, cfg.modelType, cfg.numClasses);
            if (tl.valid) { found = tl; break; }
        }

        if (!found.valid) {
            Logger::Fatal("YOLO",
                "Initial tensor validation failed — no output matches "
                "model_type='" + cfg.modelType + "', num_classes=" +
                std::to_string(cfg.numClasses));
            return false;
        }

        Logger::Info("YOLO",
            "Model validated: " + std::to_string(found.features) +
            " features, " + std::to_string(found.candidates) +
            " candidates, " + std::to_string(outs.size()) + " outputs" +
            (openCLActive ? " [OpenCL]" : " [CPU]"));

        {
            std::lock_guard<std::mutex> lock(m_detectMtx);
            m_Net        = std::move(tempNet);
            m_InputSize  = cfg.inputSize;
            m_ConfThresh = cfg.confThreshold;
            m_NmsThresh  = cfg.nmsThreshold;
            m_NumClasses = cfg.numClasses;
            m_ModelType  = ToLower(cfg.modelType);
            m_UseOpenCL.store(openCLActive);
            m_outputNames = names;
            m_LetterboxCanvas.release();
            m_Resized.release();
            m_Blob.release();
            m_modelLoaded.store(true, std::memory_order_release);
        }
        return true;
    }

    void Detect(const cv::Mat& frame, std::vector<Detection>& detections) {
        std::lock_guard<std::mutex> lock(m_detectMtx);
        detections.clear();
        m_LastBestConf.store(0.0f);

        if (!m_modelLoaded.load(std::memory_order_acquire) || m_Net.empty()) {
            m_lastStatus.store(DetectStatus::Skipped);
            return;
        }
        if (frame.empty()) {
            m_lastStatus.store(DetectStatus::Skipped);
            return;
        }
        if (frame.depth() != CV_8U ||
            (frame.channels() != 1 && frame.channels() != 3 && frame.channels() != 4)) {
            Logger::Error("YOLO", "Unsupported frame type (depth=" +
                std::to_string(frame.depth()) + ", channels=" +
                std::to_string(frame.channels()) + ")");
            m_lastStatus.store(DetectStatus::InvalidFrame);
            return;
        }

        for (int attempt = 0; attempt < 2; ++attempt) {
            LetterboxInfo lb;
            try {
                MakeLetterbox(frame, m_InputSize, lb);
                cv::dnn::blobFromImage(m_LetterboxCanvas, m_Blob, 1.0 / 255.0,
                    cv::Size(m_InputSize, m_InputSize), cv::Scalar(), true, false);
            } catch (const std::exception& e) {
                Logger::Error("YOLO", "Preprocess: " + std::string(e.what()));
                m_lastStatus.store(DetectStatus::PreprocessFailed);
                return;
            }

            std::vector<cv::Mat> outputs;
            try {
                if (m_outputNames.size() > 1) {
                    m_Net.setInput(m_Blob);
                    m_Net.forward(outputs, m_outputNames);
                } else {
                    m_Net.setInput(m_Blob);
                    cv::Mat single = m_Net.forward();
                    if (!single.empty()) outputs.push_back(single);
                }
            } catch (const std::exception& e) {
                if (m_UseOpenCL.load() && attempt == 0) {
                    Logger::Warn("YOLO", std::string("OpenCL inference failed: ") +
                        e.what() + " — attempting CPU fallback");
                    try {
                        m_Net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                    } catch (const std::exception& e2) {
                        Logger::Error("YOLO", std::string("CPU fallback failed: ") +
                            e2.what());
                        m_UseOpenCL.store(false);
                        m_lastStatus.store(DetectStatus::InferenceFailed);
                        return;
                    }
                    m_UseOpenCL.store(false);
                    continue;
                }
                Logger::Error("YOLO", "Inference failed: " + std::string(e.what()));
                m_lastStatus.store(DetectStatus::InferenceFailed);
                return;
            }

            if (outputs.empty()) {
                Logger::Error("YOLO", "Model produced no outputs");
                m_lastStatus.store(DetectStatus::InferenceFailed);
                return;
            }

            std::vector<cv::Rect> rawBoxes;
            std::vector<float>    rawConfs;
            std::vector<int>      rawClasses;
            int parsedOutputs = 0;

            try {
                for (const auto& out : outputs) {
                    if (out.empty() || out.type() != CV_32F || !out.isContinuous())
                        continue;
                    TensorLayout layout = AnalyzeTensorStatic(out, m_ModelType, m_NumClasses);
                    if (!layout.valid) continue;

                    if (m_ModelType == "yolov10") {
                        ParseEndToEnd(out, frame.size(), lb, layout,
                                      rawBoxes, rawConfs, rawClasses);
                    } else if (m_ModelType == "yolov5" || m_ModelType == "yolov7") {
                        ParseAnchorBased(out, frame.size(), lb, layout,
                                         rawBoxes, rawConfs, rawClasses);
                    } else {
                        ParseAnchorFree(out, frame.size(), lb, layout,
                                        rawBoxes, rawConfs, rawClasses);
                    }
                    ++parsedOutputs;
                }
            } catch (const std::exception& e) {
                Logger::Error("YOLO", "Parse exception: " + std::string(e.what()));
                m_lastStatus.store(DetectStatus::ParseFailed);
                return;
            }

            if (parsedOutputs == 0) {
                Logger::Error("YOLO", "No output tensor passed layout validation");
                m_lastStatus.store(DetectStatus::ParseFailed);
                return;
            }

            if (rawBoxes.size() != rawConfs.size() ||
                rawBoxes.size() != rawClasses.size()) {
                Logger::Error("YOLO", "Internal: raw arrays size mismatch");
                m_lastStatus.store(DetectStatus::ParseFailed);
                return;
            }

            if (rawBoxes.empty()) {
                m_lastStatus.store(DetectStatus::Ok);
                return;
            }

            if (m_ModelType == "yolov10") {
                for (size_t i = 0; i < rawBoxes.size(); ++i) {
                    detections.push_back({rawBoxes[i], rawConfs[i], rawClasses[i]});
                    if (rawConfs[i] > m_LastBestConf.load())
                        m_LastBestConf.store(rawConfs[i]);
                }
            } else {
                std::vector<int> keep;
                if (!NMSClassAware(rawBoxes, rawConfs, rawClasses,
                                   m_ConfThresh, m_NmsThresh, keep)) {
                    Logger::Error("YOLO", "NMS failed for all classes");
                    m_lastStatus.store(DetectStatus::ParseFailed);
                    return;
                }
                for (int i : keep) {
                    if (i < 0 || i >= static_cast<int>(rawBoxes.size())) continue;
                    detections.push_back({rawBoxes[i], rawConfs[i], rawClasses[i]});
                    if (rawConfs[i] > m_LastBestConf.load())
                        m_LastBestConf.store(rawConfs[i]);
                }
            }

            m_lastStatus.store(DetectStatus::Ok);
            return;
        }
    }

    float GetLastBestConf() const { return m_LastBestConf.load(); }

private:
    static std::string ToLower(std::string s) {
        for (char& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    static bool NMSClassAware(const std::vector<cv::Rect>& boxes,
                              const std::vector<float>& scores,
                              const std::vector<int>& classIds,
                              float confThresh, float nmsThresh,
                              std::vector<int>& keepOut) {
        keepOut.clear();
        if (boxes.empty()) return true;
        if (boxes.size() != scores.size() || boxes.size() != classIds.size()) {
            Logger::Error("YOLO", "NMSClassAware: input size mismatch");
            return false;
        }
        if (!std::isfinite(confThresh) || !std::isfinite(nmsThresh) ||
            confThresh <= 0.0f || confThresh > 1.0f ||
            nmsThresh <= 0.0f || nmsThresh > 1.0f) {
            Logger::Error("YOLO", "NMSClassAware: invalid thresholds");
            return false;
        }

        std::map<int, std::vector<int>> byClass;
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (scores[i] < confThresh) continue;
            byClass[classIds[i]].push_back(static_cast<int>(i));
        }

        for (auto& kv : byClass) {
            const int cls = kv.first;
            const auto& idxs = kv.second;

            std::vector<cv::Rect> cboxes;
            std::vector<float>    cscores;
            cboxes.reserve(idxs.size());
            cscores.reserve(idxs.size());
            for (int i : idxs) {
                cboxes.push_back(boxes[i]);
                cscores.push_back(scores[i]);
            }

            std::vector<int> ckeep;
            try {
                cv::dnn::NMSBoxes(cboxes, cscores, confThresh, nmsThresh, ckeep);
            } catch (const std::exception& e) {
                Logger::Warn("YOLO", "NMSBoxes (class " + std::to_string(cls) +
                    ") threw: " + e.what());
                continue;
            }
            for (int j : ckeep) {
                if (j >= 0 && j < static_cast<int>(idxs.size()))
                    keepOut.push_back(idxs[j]);
            }
        }
        return true;
    }

    static bool IsBoxValid(float x1, float y1, float x2, float y2) {
        return std::isfinite(x1) && std::isfinite(y1) &&
               std::isfinite(x2) && std::isfinite(y2) &&
               x2 > x1 && y2 > y1;
    }

    static void ClampBox(float& x1, float& y1, float& x2, float& y2, int w, int h) {
        x1 = std::clamp(x1, 0.0f, static_cast<float>(w));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(h));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(w));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(h));
    }

    static inline float ToFrameX(float v, const LetterboxInfo& lb) { return (v - lb.padX) / lb.scaleX; }
    static inline float ToFrameY(float v, const LetterboxInfo& lb) { return (v - lb.padY) / lb.scaleY; }

    void MakeLetterbox(const cv::Mat& img, int size, LetterboxInfo& lb) {
        if (img.empty()) throw std::invalid_argument("Empty image");
        if (size <= 0)   throw std::invalid_argument("Invalid size");

        cv::Mat converted;
        const cv::Mat* src = &img;
        if (img.channels() == 4) {
            cv::cvtColor(img, converted, cv::COLOR_BGRA2BGR);
            src = &converted;
        } else if (img.channels() == 1) {
            cv::cvtColor(img, converted, cv::COLOR_GRAY2BGR);
            src = &converted;
        }

        const int origW = src->cols;
        const int origH = src->rows;
        if (origW <= 0 || origH <= 0) throw std::invalid_argument("Invalid dimensions");

        const float scale = std::min(static_cast<float>(size) / origW,
                                     static_cast<float>(size) / origH);
        if (!std::isfinite(scale) || scale <= 0.0f)
            throw std::runtime_error("Invalid letterbox scale");

        const int newW = std::clamp(static_cast<int>(std::lround(origW * scale)), 1, size);
        const int newH = std::clamp(static_cast<int>(std::lround(origH * scale)), 1, size);

        const int padLeft = (size - newW) / 2;
        const int padTop  = (size - newH) / 2;

        if (m_Resized.cols != newW || m_Resized.rows != newH ||
            m_Resized.type() != CV_8UC3)
            m_Resized.create(newH, newW, CV_8UC3);
        cv::resize(*src, m_Resized, m_Resized.size(), 0, 0, cv::INTER_LINEAR);

        // Actual dimensions (source of truth)
        lb.scaleX = static_cast<float>(m_Resized.cols) / static_cast<float>(origW);
        lb.scaleY = static_cast<float>(m_Resized.rows) / static_cast<float>(origH);
        lb.padX   = static_cast<float>(padLeft);
        lb.padY   = static_cast<float>(padTop);
        lb.inputSize = size;

        if (m_LetterboxCanvas.cols != size || m_LetterboxCanvas.rows != size ||
            m_LetterboxCanvas.type() != CV_8UC3)
            m_LetterboxCanvas.create(size, size, CV_8UC3);
        m_LetterboxCanvas.setTo(cv::Scalar(114, 114, 114));

        m_Resized.copyTo(m_LetterboxCanvas(cv::Rect(padLeft, padTop,
                                                     m_Resized.cols,
                                                     m_Resized.rows)));
    }

    void ParseAnchorFree(const cv::Mat& out, cv::Size frameSize,
                         const LetterboxInfo& lb, const TensorLayout& layout,
                         std::vector<cv::Rect>& boxes, std::vector<float>& confs,
                         std::vector<int>& classes) {
        if (layout.features != 4 + m_NumClasses)
            throw std::runtime_error("ParseAnchorFree: unexpected feature count");

        const float* data = out.ptr<float>(0);
        const bool major = layout.featureMajor;
        const int  C     = layout.candidates;
        const int  F     = layout.features;

        for (int i = 0; i < C; ++i) {
            int bestClass = -1;
            float maxClass = 0.0f;
            for (int c = 0; c < m_NumClasses; ++c) {
                float v = major ? data[static_cast<size_t>(4 + c) * C + i]
                                : data[static_cast<size_t>(i) * F + (4 + c)];
                if (std::isfinite(v) && v > maxClass) { maxClass = v; bestClass = c; }
            }
            if (bestClass == -1 || maxClass < m_ConfThresh) continue;

            float cx = major ? data[static_cast<size_t>(0) * C + i] : data[static_cast<size_t>(i) * F + 0];
            float cy = major ? data[static_cast<size_t>(1) * C + i] : data[static_cast<size_t>(i) * F + 1];
            float w  = major ? data[static_cast<size_t>(2) * C + i] : data[static_cast<size_t>(i) * F + 2];
            float h  = major ? data[static_cast<size_t>(3) * C + i] : data[static_cast<size_t>(i) * F + 3];

            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(w)  || !std::isfinite(h) || w <= 0 || h <= 0) continue;

            float x1 = ToFrameX(cx - w * 0.5f, lb);
            float y1 = ToFrameY(cy - h * 0.5f, lb);
            float x2 = ToFrameX(cx + w * 0.5f, lb);
            float y2 = ToFrameY(cy + h * 0.5f, lb);
            ClampBox(x1, y1, x2, y2, frameSize.width, frameSize.height);
            if (!IsBoxValid(x1, y1, x2, y2)) continue;

            int bw = static_cast<int>(x2 - x1);
            int bh = static_cast<int>(y2 - y1);
            if (bw < 1 || bh < 1) continue;

            boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), bw, bh);
            confs.push_back(maxClass);
            classes.push_back(bestClass);
        }
    }

    void ParseAnchorBased(const cv::Mat& out, cv::Size frameSize,
                          const LetterboxInfo& lb, const TensorLayout& layout,
                          std::vector<cv::Rect>& boxes, std::vector<float>& confs,
                          std::vector<int>& classes) {
        if (layout.features != 5 + m_NumClasses)
            throw std::runtime_error("ParseAnchorBased: unexpected feature count");

        const float* data = out.ptr<float>(0);
        const bool major = layout.featureMajor;
        const int  C     = layout.candidates;
        const int  F     = layout.features;

        for (int i = 0; i < C; ++i) {
            float obj = major ? data[static_cast<size_t>(4) * C + i]
                              : data[static_cast<size_t>(i) * F + 4];
            if (!std::isfinite(obj) || obj < 0.01f) continue;

            int bestClass = -1;
            float maxClass = 0.0f;
            for (int c = 0; c < m_NumClasses; ++c) {
                float v = major ? data[static_cast<size_t>(5 + c) * C + i]
                                : data[static_cast<size_t>(i) * F + (5 + c)];
                if (std::isfinite(v) && v > maxClass) { maxClass = v; bestClass = c; }
            }
            if (bestClass == -1) continue;

            float conf = obj * maxClass;
            if (!std::isfinite(conf) || conf < m_ConfThresh) continue;

            float cx = major ? data[static_cast<size_t>(0) * C + i] : data[static_cast<size_t>(i) * F + 0];
            float cy = major ? data[static_cast<size_t>(1) * C + i] : data[static_cast<size_t>(i) * F + 1];
            float w  = major ? data[static_cast<size_t>(2) * C + i] : data[static_cast<size_t>(i) * F + 2];
            float h  = major ? data[static_cast<size_t>(3) * C + i] : data[static_cast<size_t>(i) * F + 3];

            if (!std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(w)  || !std::isfinite(h) || w <= 0 || h <= 0) continue;

            float x1 = ToFrameX(cx - w * 0.5f, lb);
            float y1 = ToFrameY(cy - h * 0.5f, lb);
            float x2 = ToFrameX(cx + w * 0.5f, lb);
            float y2 = ToFrameY(cy + h * 0.5f, lb);
            ClampBox(x1, y1, x2, y2, frameSize.width, frameSize.height);
            if (!IsBoxValid(x1, y1, x2, y2)) continue;

            int bw = static_cast<int>(x2 - x1);
            int bh = static_cast<int>(y2 - y1);
            if (bw < 1 || bh < 1) continue;

            boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), bw, bh);
            confs.push_back(conf);
            classes.push_back(bestClass);
        }
    }

    void ParseEndToEnd(const cv::Mat& out, cv::Size frameSize,
                       const LetterboxInfo& lb, const TensorLayout& layout,
                       std::vector<cv::Rect>& boxes, std::vector<float>& confs,
                       std::vector<int>& classes) {
        if (layout.features != 6)
            throw std::runtime_error("ParseEndToEnd: unexpected feature count");

        const int  N     = layout.candidates;
        const int  F     = layout.features;
        const bool major = layout.featureMajor;
        const float* data = out.ptr<float>(0);

        for (int i = 0; i < N; ++i) {
            float x1  = major ? data[(size_t)0 * N + i] : data[(size_t)i * F + 0];
            float y1  = major ? data[(size_t)1 * N + i] : data[(size_t)i * F + 1];
            float x2  = major ? data[(size_t)2 * N + i] : data[(size_t)i * F + 2];
            float y2  = major ? data[(size_t)3 * N + i] : data[(size_t)i * F + 3];
            float cf  = major ? data[(size_t)4 * N + i] : data[(size_t)i * F + 4];
            float cid = major ? data[(size_t)5 * N + i] : data[(size_t)i * F + 5];

            if (!std::isfinite(cf) || cf < m_ConfThresh) continue;
            if (!std::isfinite(x1) || !std::isfinite(y1) ||
                !std::isfinite(x2) || !std::isfinite(y2) ||
                !std::isfinite(cid)) continue;

            if (cid < 0.0f || cid >= static_cast<float>(m_NumClasses)) continue;
            int classId = static_cast<int>(std::lround(cid));
            if (classId < 0 || classId >= m_NumClasses) continue;

            float bx1 = ToFrameX(x1, lb);
            float by1 = ToFrameY(y1, lb);
            float bx2 = ToFrameX(x2, lb);
            float by2 = ToFrameY(y2, lb);
            ClampBox(bx1, by1, bx2, by2, frameSize.width, frameSize.height);
            if (!IsBoxValid(bx1, by1, bx2, by2)) continue;

            int bw = static_cast<int>(bx2 - bx1);
            int bh = static_cast<int>(by2 - by1);
            if (bw < 1 || bh < 1) continue;

            boxes.emplace_back(static_cast<int>(bx1), static_cast<int>(by1), bw, bh);
            confs.push_back(cf);
            classes.push_back(classId);
        }
    }

    cv::dnn::Net        m_Net;
    int                 m_InputSize  = 320;
    float               m_ConfThresh = 0.40f;
    float               m_NmsThresh  = 0.45f;
    int                 m_NumClasses = 1;
    std::string         m_ModelType  = "yolov8";

    mutable std::mutex  m_detectMtx;
    std::atomic<bool>   m_UseOpenCL{false};
    std::atomic<bool>   m_modelLoaded{false};
    std::atomic<float>  m_LastBestConf{0.0f};
    std::atomic<DetectStatus> m_lastStatus{DetectStatus::Skipped};

    std::vector<cv::String> m_outputNames;

    cv::Mat m_LetterboxCanvas;
    cv::Mat m_Resized;
    cv::Mat m_Blob;
};
