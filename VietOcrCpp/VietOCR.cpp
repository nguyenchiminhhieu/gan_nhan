#pragma once
#include "VietOCR.h"
#include <onnxruntime_cxx_api.h>
#include "PillowResize/PillowResize.hpp"
#include <regex>
#include <TGMTutil.h>
#include <TGMTfile.h>
#include "TGMTdebugger.h"
#include <numeric>    // For std::accumulate
#include <functional> // For std::multiplies


VietOCR* VietOCR::m_instance = nullptr;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

VietOCR::VietOCR()
{
    VietOCR::max_seq_length = 128;
    VietOCR::sos_token = 1;
    VietOCR::eos_token = 2;


    cnn_path_wstr = std::wstring(L"weight/cnn.onnx");
    encoder_path_wstr = std::wstring(L"weight/encoder.onnx");
    decoder_path_wstr = std::wstring(L"weight/decoder.onnx");

    vocab = Vocab();


    info[L"ID_number"] = L"";
    info[L"Name"] = L"";
    info[L"Date_of_birth"] = L"";
    info[L"Gender"] = L"";
    info[L"Nationality"] = L"";
    info[L"Place_of_origin"] = L"";
    info[L"Place_of_residence"] = L"";
    
    // Create Session
    /******* Create ORT environment *******/
    std::string instanceName{ "vietocr" };
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());

    //Ort::SessionOptions sessionOptions;
    //sessionOptions.SetIntraOpNumThreads(1);

    // Enable CUDA
    //sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});

    // Sets graph optimization level (Here, enable all possible optimizations)
    //sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Create cnn session by loading the onnx model
    cnnSession = Ort::Session(env, cnn_path_wstr.c_str(), Ort::SessionOptions{nullptr} /*sessionOptions*/);

    /******* Create allocator *******/
    // Allocator is used to get model information
    Ort::AllocatorWithDefaultOptions allocator;

    // Create encoder session by loading the onnx model
    encoderSession = Ort::Session(env, encoder_path_wstr.c_str(), Ort::SessionOptions{nullptr} /*sessionOptions*/);

    // Create decoder session by loading the onnx model
    decoderSession = Ort::Session(env, decoder_path_wstr.c_str(), Ort::SessionOptions{nullptr} /*sessionOptions*/);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

VietOCR::~VietOCR()
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

VietOCR* VietOCR::GetInstance()
{
    if (!m_instance)
        m_instance = new VietOCR();
    return m_instance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int VietOCR::resize(int w, int h, int expected_height, int image_min_width, int image_max_width)
{
    int new_w = (int)((float)expected_height * (float)(w) / (float)(h));
    int round_to = 10;
    double x = ((double)new_w / round_to);
    new_w = (int)(std::ceil(x) * round_to);
    new_w = std::max(new_w, image_min_width);
    new_w = std::min(new_w, image_max_width);
    return new_w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

///////////////////////////////////////////////////////////////////////////////////////////////////

bool VietOCR::check(std::vector<int64_t> translated_sentence, int eos_token)
{

    if (translated_sentence.back() == eos_token)
    {
        return false;
    }

    return true;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

std::wstring VietOCR::translate(cv::Mat img)
{
    /************************************************* CNN PART *************************************************/

    /******* Inputs *******/
    // Number of input nodes
    size_t numInputNodes = cnnSession.GetInputCount();

    // Name of input
    // 0 means the first input
    auto mInputName = cnnSession.GetInputNameAllocated(0, allocator);

    // Input type
    Ort::TypeInfo inputTypeInfo = cnnSession.GetInputTypeInfo(0);

    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    // Input shape
    std::vector<int64_t> mInputDims = inputTensorInfo.GetShape();

    /******* Outputs *******/
    // Number of output nodes
    size_t numOutputNodes = cnnSession.GetOutputCount();

    // Name of output
    // 0 means the first output
    auto mOutputName = cnnSession.GetOutputNameAllocated(0, allocator);

    // Output type
    Ort::TypeInfo outputTypeInfo = cnnSession.GetOutputTypeInfo(0);

    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    // Output shape
    std::vector<int64_t> mOutputDims = outputTensorInfo.GetShape();                                         

    int w = img.cols;
    int h = img.rows;

    // Get new width
    mInputDims.at(3) = VietOCR::resize(w, h, /*image_height*/ 32, /*image_min_width*/ 32, /*image_max_width*/ 512);

    if (mInputDims.at(3) % 4 == 3)
    {
        mOutputDims.at(0) = (mInputDims.at(3) - 3) / 2;
    }
    else if (mInputDims.at(3) % 4 == 2)
    {
        mOutputDims.at(0) = (mInputDims.at(3) - 2) / 2;
    }
    else if (mInputDims.at(3) % 4 == 1)
    {
        mOutputDims.at(0) = (mInputDims.at(3) - 1) / 2;
    }
    else
    {
        mOutputDims.at(0) = mInputDims.at(3) / 2;
    }
    mOutputDims.at(1) = mInputDims.at(0);       // Transpose output dimention 1 (output)
    mOutputDims.at(2) = 256;                    // Transpose output dimention 2 (hidden)

    // Resize image 
    cv::Mat resizedImage, preprocessedImage;
    cv::cvtColor(img, img, cv::ColorConversionCodes::COLOR_BGR2RGB);

    resizedImage = PillowResize::resize(img, cv::Size(mInputDims.at(3), mInputDims.at(2)), 1);

    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage, 1.0 / 255);

    // Create input tensor buffer and assign preprocessed image to the buffer
    size_t inputTensorSize = vectorProduct(mInputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());

    size_t outputTensorSize = vectorProduct(mOutputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{ mInputName.get() };
    std::vector<const char*> outputNames{ mOutputName.get() };
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo,
                                                            inputTensorValues.data(),
                                                            inputTensorSize,
                                                            mInputDims.data(),
                                                            mInputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                            outputTensorValues.data(),
                                                            outputTensorSize,
                                                            mOutputDims.data(),
                                                            mOutputDims.size()));

    cnnSession.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), numInputNodes /*Number of inputs*/,
                                               outputNames.data(), outputTensors.data(), numOutputNodes /*Number of outputs*/);

    /************************************************* ENCODER PART *************************************************/

    /******* Inputs *******/
    // Number of input nodes
    size_t numEncoderInputNodes = encoderSession.GetInputCount();

    // Name of input
    // 0 means the first input
    auto mEncoderInputName = encoderSession.GetInputNameAllocated(0, allocator);

    // Input type
    Ort::TypeInfo encoderInputTypeInfo = encoderSession.GetInputTypeInfo(0);

    auto encoderInputTensorInfo = encoderInputTypeInfo.GetTensorTypeAndShapeInfo();

    // Input shape
    std::vector<int64_t> mEncoderInputDims = encoderInputTensorInfo.GetShape();
    mEncoderInputDims.at(0) = mOutputDims.at(0);

    /******* Outputs *******/
    // Number of output nodes
    size_t numEncoderOutputNodes = encoderSession.GetOutputCount();

    // Name of output
    // 0 means the first output
    auto mEncoderOutputName = encoderSession.GetOutputNameAllocated(0, allocator);
    auto mEncoderOutputHiddenName = encoderSession.GetOutputNameAllocated(1, allocator);

    // Output type
    Ort::TypeInfo encoderOutputTypeInfo = encoderSession.GetOutputTypeInfo(0);
    Ort::TypeInfo encoderOutputHiddenTypeInfo = encoderSession.GetOutputTypeInfo(1);

    auto encoderOutputTensorInfo = encoderOutputTypeInfo.GetTensorTypeAndShapeInfo();
    auto encoderOutputHiddenTensorInfo = encoderOutputHiddenTypeInfo.GetTensorTypeAndShapeInfo();

    // Output shape
    std::vector<int64_t> mEncoderOutputDims = encoderOutputTensorInfo.GetShape();
    std::vector<int64_t> mEncoderOutputHiddenDims = encoderOutputHiddenTensorInfo.GetShape();
    mEncoderOutputDims.at(0) = mEncoderInputDims.at(0);                                                         


    size_t encoderInputTensorSize = vectorProduct(mEncoderInputDims);

    size_t encoderOutputTensorSize = vectorProduct(mEncoderOutputDims);
    std::vector<float> encoderOutputTensorValues(encoderOutputTensorSize);

    size_t encoderOutputHiddenTensorSize = vectorProduct(mEncoderOutputHiddenDims);
    std::vector<float> encoderOutputHiddenTensorValues(encoderOutputHiddenTensorSize);

    std::vector<const char*> encoderInputNames{ mEncoderInputName.get() };
    std::vector<const char*> encoderOutputNames{ mEncoderOutputName.get(), mEncoderOutputHiddenName.get() };

    std::vector<Ort::Value> encoderInputTensors;
    std::vector<Ort::Value> encoderOutputTensors;

    encoderInputTensors.push_back(Ort::Value::CreateTensor<float>(  memoryInfo,
                                                                    outputTensorValues.data(),
                                                                    encoderInputTensorSize,
                                                                    mEncoderInputDims.data(),
                                                                    mEncoderInputDims.size()));

    encoderOutputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo,
                                                                    encoderOutputTensorValues.data(),
                                                                    encoderOutputTensorSize,
                                                                    mEncoderOutputDims.data(),
                                                                    mEncoderOutputDims.size()));

    encoderOutputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo,
                                                                    encoderOutputHiddenTensorValues.data(),
                                                                    encoderOutputHiddenTensorSize,
                                                                    mEncoderOutputHiddenDims.data(),
                                                                    mEncoderOutputHiddenDims.size()));

    encoderSession.Run(Ort::RunOptions{ nullptr }, encoderInputNames.data(), encoderInputTensors.data(), numEncoderInputNodes /*Number of inputs*/,
                                                   encoderOutputNames.data(), encoderOutputTensors.data(), numEncoderOutputNodes /*Number of outputs*/);

    /************************************************* DECODER PART *************************************************/

    /******* Inputs *******/
    // Number of input nodes
    size_t numDecoderInputNodes = decoderSession.GetInputCount();

    // Name of input
    // 0 means the first input
    auto mDecoderInputTGTName = decoderSession.GetInputNameAllocated(0, allocator);
    auto mDecoderInputHiddenName = decoderSession.GetInputNameAllocated(1, allocator);
    auto mDecoderInputName = decoderSession.GetInputNameAllocated(2, allocator);

    // Input type
    Ort::TypeInfo decoderInputTGTTypeInfo = decoderSession.GetInputTypeInfo(0);
    Ort::TypeInfo decoderInputHiddenTypeInfo = decoderSession.GetInputTypeInfo(1);
    Ort::TypeInfo decoderInputTypeInfo = decoderSession.GetInputTypeInfo(2);

    auto decoderInputTensorInfo = decoderInputTypeInfo.GetTensorTypeAndShapeInfo();
    auto decoderInputHiddenTensorInfo = decoderInputHiddenTypeInfo.GetTensorTypeAndShapeInfo();
    auto decoderInputTGTTensorInfo = decoderInputTGTTypeInfo.GetTensorTypeAndShapeInfo();

    // Input shape
    std::vector<int64_t> mDecoderInputDims = decoderInputTensorInfo.GetShape();
    mDecoderInputDims.at(0) = mEncoderOutputDims.at(0);
    std::vector<int64_t> mDecoderInputHiddenDims = decoderInputHiddenTensorInfo.GetShape();
    std::vector<int64_t> mDecoderInputTGTDims = decoderInputTGTTensorInfo.GetShape();

    /******* Outputs *******/
    // Number of output nodes
    size_t numDecoderOutputNodes = decoderSession.GetOutputCount();

    // Name of output
    // 0 means the first output
    auto mDecoderOutputName = decoderSession.GetOutputNameAllocated(0, allocator);
    auto mDecoderOutputHiddenName = decoderSession.GetOutputNameAllocated(1, allocator);
    auto mDecoderOutputLastName = decoderSession.GetOutputNameAllocated(2, allocator);

    // Output type
    Ort::TypeInfo decoderOutputTypeInfo = decoderSession.GetOutputTypeInfo(0);
    Ort::TypeInfo decoderOutputHiddenTypeInfo = decoderSession.GetOutputTypeInfo(1);
    Ort::TypeInfo decoderOutputLastTypeInfo = decoderSession.GetOutputTypeInfo(2);

    auto decoderOutputTensorInfo = decoderOutputTypeInfo.GetTensorTypeAndShapeInfo();
    auto decoderOutputHiddenTensorInfo = decoderOutputHiddenTypeInfo.GetTensorTypeAndShapeInfo();
    auto decoderOutputLastTensorInfo = decoderOutputLastTypeInfo.GetTensorTypeAndShapeInfo();

    // Output shape
    std::vector<int64_t> mDecoderOutputDims = decoderOutputTensorInfo.GetShape();
    std::vector<int64_t> mDecoderOutputHiddenDims = decoderOutputHiddenTensorInfo.GetShape();
    std::vector<int64_t> mDecoderOutputLastDims = decoderOutputLastTensorInfo.GetShape();
    mDecoderOutputLastDims.at(0) = mDecoderInputDims.at(1);
    mDecoderOutputLastDims.at(1) = mDecoderInputDims.at(0);                             


    size_t decoderInputTensorSize = vectorProduct(mDecoderInputDims);

    size_t decoderInputHiddenTensorSize = vectorProduct(mDecoderInputHiddenDims);
    std::vector<float> decoderInputHiddenTensorValues(decoderInputHiddenTensorSize);
    std::vector<float> decoder_hidden = encoderOutputHiddenTensorValues;

    size_t decoderInputTGTTensorSize = vectorProduct(mDecoderInputTGTDims);
    std::vector<int64_t> decoderInputTGTTensorValues(decoderInputTGTTensorSize);

    size_t decoderOutputTensorSize = vectorProduct(mDecoderOutputDims);
    std::vector<float> decoderOutputTensorValues(decoderOutputTensorSize);
    std::vector<float> decoderOutputs;

    size_t decoderOutputHiddenTensorSize = vectorProduct(mDecoderOutputHiddenDims);
    std::vector<float> decoderOutputHiddenTensorValues(decoderOutputHiddenTensorSize);

    size_t decoderOutputLastTensorSize = vectorProduct(mDecoderOutputLastDims);
    std::vector<float> decoderOutputLastTensorValues(decoderOutputLastTensorSize);

    std::vector<const char*> decoderInputNames{ mDecoderInputTGTName.get(), mDecoderInputHiddenName.get(), mDecoderInputName.get() };
    std::vector<const char*> decoderOutputNames{ mDecoderOutputName.get(), mDecoderOutputHiddenName.get(), mDecoderOutputLastName.get() };

    std::vector<Ort::Value> decoderInputTensors;
    std::vector<Ort::Value> decoderOutputTensors;

    std::vector<int64_t> tgt_inp;
    int max_length = 0;
    std::vector<int64_t> translated_sentence = { {VietOCR::sos_token} };

    while ((max_length <= VietOCR::max_seq_length) && (VietOCR::check(translated_sentence, VietOCR::eos_token)))
    {
        tgt_inp.push_back(translated_sentence.back());
        decoderInputTGTTensorValues = tgt_inp;
        decoderInputHiddenTensorValues = decoder_hidden;

        decoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo,
                                                                        decoderInputTGTTensorValues.data(),
                                                                        decoderInputTGTTensorSize,
                                                                        mDecoderInputTGTDims.data(),
                                                                        mDecoderInputTGTDims.size()));

        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(  memoryInfo,
                                                                        decoderInputHiddenTensorValues.data(),
                                                                        decoderInputHiddenTensorSize,
                                                                        mDecoderInputHiddenDims.data(),
                                                                        mDecoderInputHiddenDims.size()));

        decoderInputTensors.push_back(Ort::Value::CreateTensor<float>(  memoryInfo,
                                                                        encoderOutputTensorValues.data(),
                                                                        decoderInputTensorSize,
                                                                        mDecoderInputDims.data(),
                                                                        mDecoderInputDims.size()));

        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo,
                                                                        decoderOutputTensorValues.data(),
                                                                        decoderOutputTensorSize,
                                                                        mDecoderOutputDims.data(),
                                                                        mDecoderOutputDims.size()));

        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo,
                                                                        decoderOutputHiddenTensorValues.data(),
                                                                        decoderOutputHiddenTensorSize,
                                                                        mDecoderOutputHiddenDims.data(),
                                                                        mDecoderOutputHiddenDims.size()));

        decoderOutputTensors.push_back(Ort::Value::CreateTensor<float>( memoryInfo,
                                                                        decoderOutputLastTensorValues.data(),
                                                                        decoderOutputLastTensorSize,
                                                                        mDecoderOutputLastDims.data(),
                                                                        mDecoderOutputLastDims.size()));

        decoderSession.Run(Ort::RunOptions{ nullptr }, decoderInputNames.data(), decoderInputTensors.data(), numDecoderInputNodes /*Number of inputs*/,
                                                       decoderOutputNames.data(), decoderOutputTensors.data(), numDecoderOutputNodes /*Number of outputs*/);

        decoderInputTensors.clear();
        decoderOutputTensors.clear();
        decoderInputTGTTensorValues.clear();
        tgt_inp.clear();

        decoderOutputs = decoderOutputTensorValues;
        decoder_hidden.clear();
        decoder_hidden = decoderOutputHiddenTensorValues;

        // Convert to Torch tensor
        /*torch::Tensor tensor_output = torch::from_blob(decoderOutputs.data(), { 1, mDecoderOutputDims.at(1) }, torch::kFloat);

        auto topoutput = torch::topk(tensor_output, 1);
        torch::Tensor indices = std::get<1>(topoutput);
        int64_t indice = *indices[0].data_ptr<int64_t>();
        translated_sentence.push_back({ indice });*/

        // Remove libtorch
        int maxIndex = std::distance(decoderOutputs.begin(), std::max_element(decoderOutputs.begin(), decoderOutputs.end()));
        translated_sentence.push_back(maxIndex);

        max_length += 1;

        decoderOutputs.clear();
    }
    
    return vocab.decode(translated_sentence);;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::wstring VietOCR::ReadMatCropped(cv::Mat img)
{
    return translate(img);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::wstring> regexSplit(std::wregex pattern, std::wstring string)
{
    std::vector<std::wstring> res = std::vector<std::wstring>();

    // Token iterator for splitting
    std::wsregex_token_iterator iter(string.begin(), string.end(), pattern, -1);
    std::wsregex_token_iterator end;
    while (iter != end) {
        res.push_back(*iter);
        ++iter;
    }

    return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::wstring joinWString(std::wstring s1, std::wstring s2)
{
    return TGMTutil::WTrim(regexSplit(std::wregex(LR"(:|;|residence|sidencs|ence|end)"), s1).back()) + L", " + TGMTutil::WTrim(s2);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void VietOCR::GetInfomation(std::vector<std::wstring> results)
{
    std::wregex regex_dob(LR"(\d{2}/\d{2})");
    std::wregex regex_residence(LR"([0-9]{2}/[0-9]{2}/|[0-9]{4,10}|Date|Demo|Dis|Dec|Dale|ting|fer|gical|ping|exp|ver|pate|cond|trị|đến|không|Không|Có|Pat|ter|ity)");

    for (int i = 0; i < results.size(); i++)
    {
        // Name
        if (std::regex_search(results[i], std::wregex(L"tên|name")))
        {
            if (!(std::regex_search(results[i + 1], std::wregex(LR"(\d+)"))))
            {
                info[L"Name"] = results[i+1];
            }
            else
            {
                info[L"Name"] = results[i + 2];
            }
            continue;
        }

        // Date of birth
        if (std::regex_search(results[i], std::wregex(L"sinh|birth|bith")))
        {
            if (std::regex_search(results[i], regex_dob))
            {
                info[L"Date_of_birth"] = results[i];
            }
            else if (std::regex_search(results[i - 1], regex_dob))
            {
                info[L"Date_of_birth"] = results[i - 1];
            }
            else if (std::regex_search(results[i + 1], regex_dob))
            {
                info[L"Date_of_birth"] = results[i + 1];
            }
            else
            {
                info[L"Date_of_birth"] = L"";
            }
            info[L"Date_of_birth"] = TGMTutil::SplitWString(info[L"Date_of_birth"], L' ').back();
            continue;
        }

        // Gender
        if (std::regex_search(results[i], std::wregex(L"Giới|Sex")))
        {
            if (std::regex_search(results[i], std::wregex(L"Nữ|nữ")))
            {
                info[L"Gender"] = L"Nữ";
            }
            else
            {
                info[L"Gender"] = L"Nam";
            }
            continue;
        }

        // Nationality
        if (std::regex_search(results[i], std::wregex(L"Quốc|tịch|Nat")))
        {
            std::wstring pattern = regexSplit(std::wregex(LR"(\\:|\\,|\\.|ty|tịch|[0-9])"), results[i]).back();
            pattern = TGMTutil::WTrim(pattern);
            if (not std::regex_search(pattern, std::wregex(L"ty|ing")) and (pattern.length() >= 3))
            {
                info[L"Nationality"] = pattern;
            }
            else if (not std::regex_search(results[i+1], std::wregex(regex_dob)))
            {
                info[L"Nationality"] = results[i + 1];
            }
            else
            {
                info[L"Nationality"] = results[i - 1];
            }

            for (std::wstring s : regexSplit(std::wregex(L"\s+"), info[L"Nationality"]))
            {
                if (s.length() < 3)
                {
                    info[L"Nationality"] = TGMTutil::WTrim(regexSplit(std::wregex(s), info[L"Nationality"]).back());
                }
            }
            if (std::regex_search(info[L"Nationality"], std::wregex(LR"(Nam)")))
            {
                info[L"Nationality"] = LR"(Việt Nam)";
            }
            continue;
        }

        // Place of origin
        if (std::regex_search(results[i], std::wregex(LR"(Quê|origin|ongin|ngin|orging)")))
        {
            if (not std::regex_search(results[i + 1], std::wregex(LR"([0 - 9] {4})")))
            {
                std::wstring origin_pat = TGMTutil::WTrim(regexSplit(std::wregex(LR"(:|;|of|ging|gin|ggong)"), results[i]).back());
                if (origin_pat.length() > 2)
                {
                    info[L"Place_of_origin"] = origin_pat + L", " + results[i + 1];
                }
                else
                {
                    info[L"Place_of_origin"] = results[i + 1];
                }
            }
            else
            {
                info[L"Place_of_origin"] = L"";
            }
            continue;
        }

        // Place of residence
        if (std::regex_search(results[i], std::wregex(LR"(Nơi|trú|residence)")))
        {
            std::wstring vals2;
            std::wstring vals3;
            if ((i + 2) > results.size() - 1)
            {
                vals2 = L"";
            }
            else if (results[i + 2].length() > 5)
            {
                vals2 = results[i + 2];
            }
            else
            {
                vals2 = results[-1];
            }

            if ((i + 3) > results.size() - 1)
            {
                vals3 = L"";
            }
            else
            {
                vals3 = results[i + 3];
            }

            if (TGMTutil::WTrim(regexSplit(std::wregex(LR"(:|;|residence|ence|end)"), results[i]).back()) != L"")
            {
                if ((vals2 != L"") and not std::regex_search(vals2, regex_residence))
                {
                    info[L"Place_of_residence"] = joinWString(results[i], vals2);
                }
                else if ((vals3 != L"") and not std::regex_search(vals3, regex_residence))
                {
                    info[L"Place_of_residence"] = joinWString(results[i], vals3);
                }
                else if (not std::regex_search(results[-1], regex_residence))
                {
                    info[L"Place_of_residence"] = joinWString(results[i], results[-1]);
                }
                else
                {
                    info[L"Place_of_residence"] = results[-1];
                }
            }
            else
            {
                if ((vals2 != L"") and not std::regex_search(vals2, regex_residence))
                {
                    info[L"Place_of_residence"] = vals2;
                }
                else
                {
                    info[L"Place_of_residence"] = results[-1];
                }
            }
            continue;
        }
        else if (i == results.size()-1)
        {
            if (info[L"Place_of_residence"] == L"")
            {
                if (not std::regex_search(results[-1], regex_residence))
                {
                    info[L"Place_of_residence"] = results[-1];
                }
                else if (not std::regex_search(results[-2], regex_residence))
                {
                    info[L"Place_of_residence"] = results[-2];
                }
                else
                {
                    info[L"Place_of_residence"] = L"";
                }
            }
            continue;
        }
        else
        {
            continue;
        }


    }
}