/**
 * @file PPOKit/tests/support/echo_program.cpp
 * @brief Program started by the process tests: prints its arguments one per line (stdout and
 *        stderr), then waits with --wait, or exits with the code after --exit.
 */
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#endif

int main(int argc, char** argv)
{
    int code = 0;
    bool wait = false;
    for (int index = 1; index < argc; ++index)
    {
        std::string const argument = argv[index];
        if (argument == "--exit" && index + 1 < argc)
        {
            code = std::atoi(argv[++index]);
            continue;
        }
        if (argument == "--hold-pipe")
        {
            // The direct child exits with a grandchild still owning stdout/stderr.
#if defined(_WIN32)
            STARTUPINFOW start{};
            start.cb = sizeof start;
            start.dwFlags = STARTF_USESTDHANDLES;
            start.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
            start.hStdError = GetStdHandle(STD_ERROR_HANDLE);
            start.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            wchar_t path[MAX_PATH]{};
            GetModuleFileNameW(nullptr, path, MAX_PATH);
            std::wstring command = std::wstring(L"\"") + path + L"\" --wait";
            PROCESS_INFORMATION child{};
            if (!CreateProcessW(path, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &start, &child))
                return 2;
            CloseHandle(child.hProcess);
            CloseHandle(child.hThread);
#else
            auto const child = ::fork();
            if (child < 0) return 2;
            if (child == 0)
            {
                ::sleep(60);
                ::_exit(0);
            }
#endif
            std::cout << "inherited" << std::endl;
            return 7;
        }
        if (argument == "--wait")
        {
            wait = true;
            continue;
        }
        std::cout << "arg: " << argument << std::endl;
    }
    std::cerr << "to stderr" << std::endl;
    if (wait)
    {
        std::cout << "waiting" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
    return code;
}
