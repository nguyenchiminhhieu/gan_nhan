#pragma once

#include <string>
#include <map>
#include <vector>


class Vocab
{	
	int pad;
	int go;
	int eos;
	int mask_token;
	std::wstring m_chars;
public:
	
	std::map<wchar_t, int> c2i;
	std::map<int, wchar_t> i2c;

	Vocab();

	std::vector<int> encode(std::wstring chars);
	std::wstring decode(std::vector<int64_t> ids);

};

