#pragma once
#include <string>
#if defined(WIN32) ||  defined(_WIN64)
#include <windows.h>
#endif

#include <functional>
void debug_out(int color, const char* fmt, ...);

void StartCountTime(const char* taskName);
int StopCountTime(const char* taskName);
int StopAndPrintCountTime(const char* taskName);
void TGMTSetConsoleTitle(const char* fmt,...);
void TGMTSetConsoleFont(const char* fmt, ...);
void WriteLog(char* fmt, ...);
void DeleteLog(int nDaysAgo);
std::string GetCurrentDateTime(bool removeSpecialCharacter = false);
std::string GetCurrentDate();
int GetOrderDayInYear(int month, int day);

#if defined(WIN32) ||  defined(_WIN64)
void ShowMessageBox(char* title, char* fmt, ...);
void ShowMessageBoxW(std::wstring title, std::wstring msg);
void ShowErrorBoxW(std::wstring title, std::wstring msg);
void ShowErrorBox(const char* title, const char* fmt, ...);

void PrintUnicode(const wchar_t s[]);

//create crash dump, using: 
//SetUnhandledExceptionFilter(CreateMinidump);
LONG WINAPI CreateMinidump(struct _EXCEPTION_POINTERS* apExceptionInfo);
#endif

#ifdef _DEBUG
#define DEBUG_OUT(...)						debug_out(0, __VA_ARGS__)
#define DEBUG_OUT_CON(...)					debug_out_con(__VA_ARGS__)
#define DEBUG_OUT_COLOR(c, ...)				debug_out(c, __VA_ARGS__)
#else
#define DEBUG_OUT(...)					
#define DEBUG_OUT_CON(...)				
#define DEBUG_OUT_COLOR(c, ...)			
#endif

#if defined(WIN32) ||  defined(WIN64) || defined (OS_LINUX)
#define TODO(...)							debug_out(0, __VA_ARGS__)
#define PrintMessage(...)					debug_out(0, __VA_ARGS__)
#define PrintError(...)						debug_out(1, __VA_ARGS__)
#define PrintMessageYellow(...)				debug_out(2, __VA_ARGS__)
#define PrintSuccess(...)					debug_out(3, __VA_ARGS__)
#define PrintMessageGreen(...)				debug_out(3, __VA_ARGS__)
#define PrintMessageBlue(...)				debug_out(4, __VA_ARGS__)
#elif defined (ANDROID) || defined (__APPLE__)
#define TODO(...)							
#define PrintMessage(...)					
#define PrintError(...)						
#define PrintMessageYellow(...)				
#define PrintSuccess(...)					
#define PrintMessageGreen(...)				
#define PrintMessageBlue(...)				
#endif


#if defined(LIB_CS) || defined(LIB_CPP) || defined(_MANAGED)
void OutputDebug(std::string str);
void ShowAssertManaged(const char* fmt, ...);
#if defined(WIN32) ||  defined(_WIN64)
#define ASSERT(value,x,...) if(!(value)) \
{\
	ShowAssertManaged("%s(%d):\n" #x, __FILE__, __LINE__,__VA_ARGS__ );\
}
#else
#define ASSERT(value,x,...)
#endif
#else
#if defined(WIN32) ||  defined(_WIN64)
#define ASSERT(value,x,...) if(!(value)) \
{\
	debug_out(1,  "%s(%d):\n" #x, __FILE__, __LINE__,__VA_ARGS__); \
	__debugbreak(); \
}

#define CHECK(value,x,...) \
if(value) \
{\
	debug_out(3, #x " success", __VA_ARGS__); \
}\
else\
{\
	debug_out(1, #x " failed", __VA_ARGS__); \
	WriteLog("TestCaseLog.txt", "%s(%d) " #x " failed\n", __FILE__, __LINE__,__VA_ARGS__);\
}

#elif ANDROID || OS_LINUX || __APPLE__
#define ASSERT(value,...)
#endif
#endif


#define START_COUNT_TIME StartCountTime
#define STOP_COUNT_TIME StopCountTime
#define STOP_AND_PRINT_COUNT_TIME StopAndPrintCountTime

#define SET_CONSOLE_TITLE(...) TGMTSetConsoleTitle(__VA_ARGS__);
#define SET_CONSOLE_FONT(...) TGMTSetConsoleFont(__VA_ARGS__);

#define GetTGMTdebugger TGMTdebugger::GetInstance

class TGMTdebugger
{
	static TGMTdebugger* m_instance;
public:
	static TGMTdebugger* GetInstance()
	{
		if (!m_instance)
			m_instance = new TGMTdebugger();
		return m_instance;
	}

	std::function<void(std::string)> OnTGMTdebuggerLog;
};



