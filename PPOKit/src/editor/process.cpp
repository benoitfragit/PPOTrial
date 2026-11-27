/**
 * @file PPOKit/src/editor/process.cpp
 * @brief Starting, watching and killing a program, on Windows and POSIX systems.
 */
#include "ppokit/editor/process.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <unistd.h>
#endif

namespace ppokit::editor {

namespace {

/** @brief Bytes read from the pipe at once. */
constexpr std::size_t ReadChunk = 4096;

bool NeedsQuotes(std::string const& argument)
{
    return argument.empty() || argument.find_first_of(" \t\"") != std::string::npos;
}

/** @brief Resolves a launch path before creating the child process. */
std::filesystem::path AbsoluteLaunchPath(std::filesystem::path const& path)
{
    if (path.empty())
    {
        return {};
    }
    std::error_code error;
    std::filesystem::path const absolute = std::filesystem::absolute(path, error);
    if (error)
    {
        throw std::runtime_error("cannot resolve launch path: " + path.u8string());
    }
    return absolute.lexically_normal();
}

/**
 * @brief `argument` quoted for a Windows command line (the rules CommandLineToArgvW reads
 *        back): backslashes before a quote are doubled, quotes are escaped.
 */
std::string QuoteArgument(std::string const& argument)
{
    if (!NeedsQuotes(argument))
    {
        return argument;
    }
    std::string quoted = "\"";
    std::size_t backslashes = 0;
    for (char const character : argument)
    {
        if (character == '\\')
        {
            ++backslashes;
            continue;
        }
        if (character == '"')
        {
            quoted.append(backslashes * 2 + 1, '\\');
        }
        else
        {
            quoted.append(backslashes, '\\');
        }
        backslashes = 0;
        quoted += character;
    }
    quoted.append(backslashes * 2, '\\');
    quoted += '"';
    return quoted;
}

} // namespace

/** @brief Platform handles, the reader thread and the lines it collected. */
struct Process::State final
{
    std::mutex mutex;
    std::deque<std::string> lines;
    std::string partial;
    std::thread reader;
    std::atomic<bool> finishing{false};
    std::optional<int> exitCode;
    bool started = false;
#if defined(_WIN32)
    HANDLE job = nullptr;
    HANDLE process = nullptr;
    HANDLE output = nullptr;
#else
    pid_t pid = -1;
    int output = -1;
#endif

    /** @brief Splits `bytes` into lines (CR before LF dropped) and keeps the last MaximumLines. */
    void Append(char const* bytes, std::size_t count)
    {
        std::lock_guard<std::mutex> const lock(mutex);
        for (std::size_t index = 0; index < count; ++index)
        {
            char const character = bytes[index];
            if (character == '\n')
            {
                if (!partial.empty() && partial.back() == '\r')
                {
                    partial.pop_back();
                }
                lines.push_back(std::move(partial));
                partial.clear();
                if (lines.size() > MaximumLines)
                {
                    lines.pop_front();
                }
                continue;
            }
            // Bound an unterminated console line as well as the number of queued lines.
            if (partial.size() < 64 * 1024)
            {
                partial += character;
            }
        }
    }

    void Flush()
    {
        std::lock_guard<std::mutex> const lock(mutex);
        if (!partial.empty())
        {
            lines.push_back(std::move(partial));
            partial.clear();
        }
    }

    void Read()
    {
        char buffer[ReadChunk];
        // Read only available bytes. After the direct child exits, preserve queued
        // output but never wait for an inherited writer or an endless output stream.
        std::size_t remainingDrain = 1024 * 1024;
        for (;;)
        {
            bool const ending = finishing.load();
#if defined(_WIN32)
            DWORD available = 0;
            if (!PeekNamedPipe(output, nullptr, 0, nullptr, &available, nullptr))
            {
                break;
            }
            if (available == 0)
            {
                if (ending) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            DWORD read = 0;
            if (!ReadFile(output, buffer, std::min<DWORD>(available, sizeof buffer), &read, nullptr) || read == 0)
            {
                break;
            }
#else
            pollfd descriptor{output, POLLIN, 0};
            int const ready = ::poll(&descriptor, 1, ending ? 0 : 20);
            if (ready < 0 && errno == EINTR && !ending) continue;
            if (ready <= 0)
            {
                if (ending || ready < 0) break;
                continue;
            }
            ssize_t const read = ::read(output, buffer, sizeof buffer);
            if (read < 0 && errno == EINTR && !ending) continue;
            if (read <= 0) break;
#endif
            auto const count = static_cast<std::size_t>(read);
            Append(buffer, count);
            if (ending)
            {
                if (remainingDrain <= count) break;
                remainingDrain -= count;
            }
        }
        Flush();
    }

    /** @brief Waits for the reader (the pipe closes when the program ends) and releases handles. */
    void Finish()
    {
        finishing.store(true);
#if defined(_WIN32)
        if (job)
        {
            CloseHandle(job); // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE also owns descendants.
            job = nullptr;
        }
#else
        if (pid > 0)
        {
            ::kill(-pid, SIGKILL);
        }
#endif
        if (reader.joinable())
        {
            reader.join();
        }
#if defined(_WIN32)
        if (output != nullptr)
        {
            CloseHandle(output);
            output = nullptr;
        }
        if (process != nullptr)
        {
            CloseHandle(process);
            process = nullptr;
        }
#else
        if (output >= 0)
        {
            ::close(output);
            output = -1;
        }
        if (pid > 0)
        {
            int status = 0;
            pid_t const child = pid;
            pid_t const reaped = ::waitpid(child, &status, WNOHANG);
            if (reaped == 0)
            {
                // An uninterruptible child must not freeze Studio shutdown or become a zombie later.
                std::thread([child] {
                    int childStatus = 0;
                    while (::waitpid(child, &childStatus, 0) < 0 && errno == EINTR) {}
                }).detach();
            }
        }
        pid = -1;
#endif
    }
};

Process::Process() : state_(std::make_unique<State>()) {}

Process::~Process()
{
    if (Running())
    {
        Kill();
    }
    state_->Finish();
}

#if defined(_WIN32)

namespace {

std::wstring Wide(std::string const& text)
{
    if (text.empty())
    {
        return {};
    }
    int const size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (size <= 0)
    {
        throw std::runtime_error("a process argument is not valid UTF-8");
    }
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
            wide.data(), size) != size)
    {
        throw std::runtime_error("cannot convert a process argument to UTF-16");
    }
    return wide;
}

bool NeedsQuotes(std::wstring const& argument)
{
    return argument.empty() || argument.find_first_of(L" \t\"") != std::wstring::npos;
}

/** @brief Quotes one UTF-16 argument according to CommandLineToArgvW rules. */
std::wstring QuoteWideArgument(std::wstring const& argument)
{
    if (!NeedsQuotes(argument))
    {
        return argument;
    }
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;
    for (wchar_t const character : argument)
    {
        if (character == L'\\')
        {
            ++backslashes;
            continue;
        }
        quoted.append(backslashes * (character == L'"' ? 2 : 1), L'\\');
        if (character == L'"')
        {
            quoted += L'\\';
        }
        backslashes = 0;
        quoted += character;
    }
    quoted.append(backslashes * 2, L'\\');
    quoted += L'"';
    return quoted;
}

/** @brief Builds the exact UTF-16 command line passed to CreateProcessW. */
std::wstring WideCommandLine(std::filesystem::path const& program, std::vector<std::string> const& arguments)
{
    std::wstring command = QuoteWideArgument(program.wstring());
    for (std::string const& argument : arguments)
    {
        command += L' ';
        command += QuoteWideArgument(Wide(argument));
    }
    return command;
}

} // namespace

void Process::Start(std::filesystem::path const& program, std::vector<std::string> const& arguments,
    std::filesystem::path const& folder)
{
    if (Running())
    {
        throw std::runtime_error("a program is already running: stop it first");
    }
    state_->Finish();
    state_ = std::make_unique<State>();
    std::filesystem::path const resolvedProgram = AbsoluteLaunchPath(program);
    std::filesystem::path const resolvedFolder = AbsoluteLaunchPath(folder);
    commandLine_ = QuoteCommandLine(resolvedProgram, arguments);

    // A handle list prevents one concurrent launch from inheriting another launch's writer.
    SIZE_T attributeSize = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeSize);
    std::vector<unsigned char> attributeStorage(attributeSize);
    auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributeStorage.data());
    std::wstring command = WideCommandLine(resolvedProgram, arguments);
    std::wstring const workingFolder = resolvedFolder.wstring();
    std::wstring const executable = resolvedProgram.wstring();
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE readEnd = nullptr;
    HANDLE writeEnd = nullptr;
    if (!CreatePipe(&readEnd, &writeEnd, &security, 0))
    {
        throw std::runtime_error("cannot create the output pipe of " + resolvedProgram.u8string());
    }
    HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    HANDLE inherited[] = {writeEnd, input};
    bool const initialized = InitializeProcThreadAttributeList(attributes, 1, 0, &attributeSize) != FALSE;
    if (!initialized || input == INVALID_HANDLE_VALUE || !job
        || !SetHandleInformation(readEnd, HANDLE_FLAG_INHERIT, 0)
        || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits)
        || !UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
            sizeof inherited, nullptr, nullptr))
    {
        if (initialized) DeleteProcThreadAttributeList(attributes);
        if (input != INVALID_HANDLE_VALUE) CloseHandle(input);
        if (job) CloseHandle(job);
        CloseHandle(readEnd);
        CloseHandle(writeEnd);
        throw std::runtime_error("cannot prepare the child process job and pipes");
    }
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof startup;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdOutput = writeEnd;
    startup.StartupInfo.hStdError = writeEnd;
    startup.StartupInfo.hStdInput = input;
    startup.lpAttributeList = attributes;
    PROCESS_INFORMATION information{};
    BOOL const created = CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr,
        workingFolder.empty() ? nullptr : workingFolder.c_str(), &startup.StartupInfo, &information);
    DWORD const createError = GetLastError();
    DeleteProcThreadAttributeList(attributes);
    CloseHandle(input);
    CloseHandle(writeEnd);
    if (!created)
    {
        CloseHandle(job);
        CloseHandle(readEnd);
        throw std::runtime_error("cannot start " + resolvedProgram.u8string() + " (Windows error "
                                 + std::to_string(createError) + ")");
    }
    if (!AssignProcessToJobObject(job, information.hProcess) || ResumeThread(information.hThread) == DWORD(-1))
    {
        TerminateProcess(information.hProcess, 1);
        CloseHandle(information.hThread);
        CloseHandle(information.hProcess);
        CloseHandle(job);
        CloseHandle(readEnd);
        throw std::runtime_error("cannot start the child in its job");
    }
    CloseHandle(information.hThread);
    state_->job = job;
    state_->process = information.hProcess;
    state_->output = readEnd;
    state_->started = true;
    try
    {
        state_->reader = std::thread([state = state_.get()] { state->Read(); });
    }
    catch (...)
    {
        Kill();
        state_->Finish();
        throw;
    }
}

bool Process::Running()
{
    if (!state_->started || state_->exitCode)
    {
        return false;
    }
    if (WaitForSingleObject(state_->process, 0) != WAIT_OBJECT_0)
    {
        return true;
    }
    DWORD code = 0;
    GetExitCodeProcess(state_->process, &code);
    state_->exitCode = static_cast<int>(code);
    state_->Finish();
    return false;
}

void Process::Kill()
{
    if (!Running())
    {
        return;
    }
    TerminateJobObject(state_->job, 1);
    // Never wait on the GUI thread. A following Running/Poll observes completion.
    Running();
}

#else

void Process::Start(std::filesystem::path const& program, std::vector<std::string> const& arguments,
    std::filesystem::path const& folder)
{
    if (Running())
    {
        throw std::runtime_error("a program is already running: stop it first");
    }
    state_->Finish();
    state_ = std::make_unique<State>();
    std::filesystem::path const resolvedProgram = AbsoluteLaunchPath(program);
    std::filesystem::path const resolvedFolder = AbsoluteLaunchPath(folder);
    commandLine_ = QuoteCommandLine(resolvedProgram, arguments);

    // Everything the child needs is prepared before fork: it only calls async-signal-safe functions.
    std::string const executable = resolvedProgram.string();
    std::string const workingFolder = resolvedFolder.string();
    std::vector<std::string> words{executable};
    words.insert(words.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    for (std::string& word : words)
    {
        argv.push_back(word.data());
    }
    argv.push_back(nullptr);
    int pipeEnds[2] = {-1, -1};
#if defined(__linux__)
    if (::pipe2(pipeEnds, O_CLOEXEC) != 0)
#else
    if (::pipe(pipeEnds) != 0)
#endif
    {
        throw std::runtime_error("cannot create the output pipe of " + resolvedProgram.u8string());
    }
#if defined(__linux__)
    pid_t const parent = ::getpid();
#endif
    pid_t const pid = ::fork();
    if (pid < 0)
    {
        ::close(pipeEnds[0]);
        ::close(pipeEnds[1]);
        throw std::runtime_error("cannot start " + executable);
    }
    if (pid == 0)
    {
#if defined(__linux__)
        // Cover both sides of the fork/prctl race if the owning Studio crashes.
        if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || ::getppid() != parent) ::_exit(126);
#endif
        if (::setpgid(0, 0) != 0 || ::dup2(pipeEnds[1], STDOUT_FILENO) < 0
            || ::dup2(pipeEnds[1], STDERR_FILENO) < 0)
        {
            ::_exit(126);
        }
        ::close(pipeEnds[0]);
        ::close(pipeEnds[1]);
        if (!workingFolder.empty() && ::chdir(workingFolder.c_str()) != 0)
        {
            ::_exit(126);
        }
        ::execv(executable.c_str(), argv.data());
        char const message[] = "cannot run the program\n";
        if (::write(STDERR_FILENO, message, sizeof message - 1) < 0)
        {
            ::_exit(127);
        }
        ::_exit(127);
    }
    ::setpgid(pid, pid); // The child also sets its group before exec.
    ::close(pipeEnds[1]);
    state_->pid = pid;
    state_->output = pipeEnds[0];
    state_->started = true;
    try
    {
        state_->reader = std::thread([state = state_.get()] { state->Read(); });
    }
    catch (...)
    {
        Kill();
        state_->Finish();
        throw;
    }
}

bool Process::Running()
{
    if (!state_->started || state_->exitCode)
    {
        return false;
    }
    int status = 0;
    pid_t result;
    do { result = ::waitpid(state_->pid, &status, WNOHANG); } while (result < 0 && errno == EINTR);
    if (result == 0)
    {
        return true;
    }
    if (result == state_->pid && WIFEXITED(status))
    {
        state_->exitCode = WEXITSTATUS(status);
    }
    else if (result == state_->pid && WIFSIGNALED(status))
    {
        state_->exitCode = 128 + WTERMSIG(status);
    }
    else
    {
        state_->exitCode = -1;
    }
    state_->Finish();
    return false;
}

void Process::Kill()
{
    if (!Running())
    {
        return;
    }
    ::kill(-state_->pid, SIGKILL);
    ::kill(state_->pid, SIGKILL);
    // Never wait on the GUI thread. A following Running/Poll reaps the direct child.
    Running();
}

#endif

std::optional<int> Process::ExitCode()
{
    Running();
    return state_->exitCode;
}

std::vector<std::string> Process::TakeLines()
{
    std::lock_guard<std::mutex> const lock(state_->mutex);
    std::vector<std::string> lines(std::make_move_iterator(state_->lines.begin()),
        std::make_move_iterator(state_->lines.end()));
    state_->lines.clear();
    return lines;
}

std::string QuoteCommandLine(std::filesystem::path const& program, std::vector<std::string> const& arguments)
{
    std::string line = QuoteArgument(program.u8string());
    for (std::string const& argument : arguments)
    {
        line += ' ';
        line += QuoteArgument(argument);
    }
    return line;
}

} // namespace ppokit::editor
