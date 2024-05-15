#pragma once
#include <string>
#include <map>

#define GetTGMTConfig TGMTConfig::GetInstance
#ifdef UNICODE
#define TGMT_STRING std::wstring
#define TGMTSTR LPCWSTR
#else
#define TGMT_STRING std::string
#define TGMTSTR LPCSTR
#endif

class TGMTConfig
{
	static TGMTConfig* instance;

	class INIReader
	{
	private:
		bool m_lastReadSuccess;
		
		std::map<std::string, std::string> m_values;
		static std::string MakeKey(std::string section, std::string name);
		static int ValueHandler(void* user, char* section, char* name, char* value);
	public:
		bool m_writeDefaultValueIfNotExist = false;

		INIReader();

		bool LoadSetting(std::string settingFile);

		std::string ReadString(std::string section, std::string name, std::string default_value);
		int ReadInt(std::string section, std::string name, int default_value);
		double ReadDouble(std::string section, std::string name, double default_value);
		//valid true values are "true", "yes", "on", "1",
		// and valid false values are "false", "no", "off", "0" (not case sensitive).
		bool ReadBool(std::string section, std::string name, bool default_value);
		bool LastReadSuccess();
	}m_reader;


	std::string m_iniFile;

	bool m_lastReadSuccess;
public:
	static TGMTConfig* GetInstance()
	{
		if (!instance)
			instance = new TGMTConfig();
		return instance;
	}

	TGMTConfig();
	~TGMTConfig();

	bool LoadConfig(std::string settingFile = "config.ini");

#if (defined(WIN32) || defined(WIN64)) && defined( UNICODE)
	std::wstring ReadWString(std::string section, std::string key, std::wstring defaultValue = L"");
#endif
	std::string ReadString(std::string section, std::string key, std::string defaultValue = "");
	int ReadInt(std::string section, std::string key, int defaultValue = 0);
	bool ReadBool(std::string section, std::string key, bool defaultValue = false);
	double ReadDouble(std::string section, std::string key, double defaultValue = 0.f);

	bool LastReadSuccess();

#if defined(WIN32) || defined(WIN64)
	void WriteString(std::string section, std::string key, std::string value);
#ifdef UNICODE
	void WriteWString(std::string section, std::string key, std::wstring value);
#endif
	void WriteInt(std::string section, std::string key, int value);
	void WriteDouble(std::string section, std::string key, double value);
	void WriteBool(std::string section, std::string key, bool value);
#endif


	void SetWriteDefaultValueIfNotExist(bool enable) { m_reader.m_writeDefaultValueIfNotExist = enable; };
};
