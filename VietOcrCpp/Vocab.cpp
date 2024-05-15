#include "Vocab.h"


Vocab::Vocab()
{
    Vocab::pad = 0;
    Vocab::go = 1;
    Vocab::eos = 2;
    Vocab::mask_token = 3;
    
    Vocab::m_chars = L"aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";

    for (int i = 0;i < m_chars.size(); i++)
    {
        c2i.insert(std::pair<wchar_t, int>(m_chars[i], i+ 4));
        i2c.insert(std::pair<int, wchar_t>(i + 4, m_chars[i]));
    }

    //c2i = { c:i + 4 for i, c in enumerate(chars) }

    //i2c = { i + 4:c for i, c in enumerate(chars) }

    i2c[0] = (wchar_t) "<pad>";
    i2c[1] = (wchar_t) "<sos>";
    i2c[2] = (wchar_t) "<eos>";
    i2c[3] = (wchar_t) "*";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<int> Vocab::encode(std::wstring chars)
{
    std::vector<int> encodeOutputs;

    encodeOutputs.push_back(Vocab::go);

    for (int i = 0; i < chars.size(); i++)
    {
        encodeOutputs.push_back(Vocab::c2i[chars[i]]);
    }

    encodeOutputs.push_back(Vocab::eos);

    return encodeOutputs;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

std::wstring Vocab::decode(std::vector<int64_t> ids)
{
    std::vector<std::wstring> sentences;
    int first;
    int last;

    sentences.push_back(L"");
    
    if (std::find(ids.begin(), ids.end(), Vocab::go) != ids.end())
    {
        first = 1;
    }
    else 
    { 
        first = 0; 
    }
    
    auto it = std::find(ids.begin(), ids.end(), Vocab::eos);
    if (it != ids.end())
    {
        last = it - ids.begin();

        for (int i = first; i < last; i++)
        {
            sentences[0] += Vocab::i2c[ids[i]];
        }
    }
    else
    {
        last = ids.end() - ids.begin();

        for (int i = first; i < last+1; i++)
        {
            sentences[0] += Vocab::i2c[ids[i]];
        }
    }

    return sentences[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////

