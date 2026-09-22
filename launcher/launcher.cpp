// Portable launcher: use only the runtime beside this executable, without a shell.
#include <windows.h>
#include <wchar.h>
#include <stdio.h>

static int fail(const wchar_t* message, DWORD code = 0) {
    wchar_t detail[2048];
    if (code) {
        wchar_t reason[1024] = {};
        FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       nullptr, code, 0, reason, 1024, nullptr);
        swprintf(detail, 2048, L"%ls\n\nWindows error %lu: %ls", message, code, reason);
        message = detail;
    }
    MessageBoxW(nullptr, message, L"Apex Highlights", MB_OK | MB_ICONERROR);
    return 1;
}

static bool is_file(const wchar_t* path) {
    DWORD attributes = GetFileAttributesW(path);
    return attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY);
}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    wchar_t root[32768];
    DWORD length = GetModuleFileNameW(nullptr, root, 32768);
    if (!length || length >= 32768)
        return fail(L"Cannot determine the application directory.");
    wchar_t* separator = wcsrchr(root, L'\\');
    if (!separator) return fail(L"Invalid application directory.");
    *separator = L'\0';
    wchar_t python[32768], script[32768], command[65536];
    if (swprintf(python, 32768, L"%ls\\runtime\\pythonw.exe", root) < 0 ||
        swprintf(script, 32768, L"%ls\\launch.pyw", root) < 0)
        return fail(L"The application path is too long.");
    if (!is_file(python) || !is_file(script))
        return fail(L"Required application files are missing.\nExtract the complete portable ZIP, then run ApexHighlights.exe.\nIf security software removed a file, check its protection history.");
    // An explicit application path prevents PATH search and executable ambiguity.
    swprintf(command, 65536, L"\"%ls\" -I \"%ls\"", python, script);
    STARTUPINFOW startup = {};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {};
    if (!CreateProcessW(python, command, nullptr, nullptr, FALSE, 0,
                        nullptr, root, &startup, &process))
        return fail(L"Could not start Apex Highlights.", GetLastError());
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return 0;
}
