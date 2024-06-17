#if !defined(LIB_CS) && !defined(LIB_CPP)

#pragma once
#include <fstream>
#include "VietOCR.h"
#include "TGMTdebugger.h"
#include "TGMTfile.h"



void Test()
{
    std::cout << std::endl;

    cv::Mat image = cv::imread("text_4.jpg");

    VietOCR vi = VietOCR();
    //StartCountTime("read");
    std::wstring s = vi.ReadMatCropped(image);
    //StopAndPrintCountTime("read");
    PrintUnicode(s.c_str());
    std::cout << std::endl;

}


int main()
{
    SetConsoleOutputCP(CP_UTF8);

    Test();
    
    return 0;
}

#endif