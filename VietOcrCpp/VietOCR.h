#pragma once
#include <iostream>

#include <onnxruntime_cxx_api.h>
#include "stdafx.h"
#include "Vocab.h"


#define GetVietOCR VietOCR::GetInstance


class VietOCR
{
    static VietOCR* m_instance;

    int max_seq_length;
    int sos_token;
    int eos_token;
    
    std::wstring cnn_path_wstr;
    std::wstring encoder_path_wstr;
    std::wstring decoder_path_wstr;

    Ort::Session cnnSession{ nullptr };
    Ort::Session encoderSession{ nullptr };
    Ort::Session decoderSession{ nullptr };
    Ort::AllocatorWithDefaultOptions allocator;
    
    Vocab vocab;

    std::wstring translate(cv::Mat img);
public:

    VietOCR();
    ~VietOCR();

    static VietOCR* GetInstance();

    std::map<std::wstring, std::wstring> info;

	static int resize(int w, int h, int expected_height, int image_min_width, int image_max_width);

    bool check(std::vector<int64_t> translated_sentence, int eos_token);

    std::wstring ReadMatCropped(cv::Mat img);
    void GetInfomation(std::vector<std::wstring> results);
};
