/**
 * @file PPOKit/include/ppokit/view/frames.hpp
 * @brief The UDP frames a Visualiseur sends after each step, and their reading.
 * @details Header-only, no dependency: a view DLL reads frames without linking anything.
 * A frame describes itself, so a view needs nothing else to read it:
 *
 * @verbatim
 * "PPKV"  version:u16  reserved:u16  sequence:u64  agents:u32  ports:u32
 * per port: nameLength:u16  name  size:u32  values: agents x size float32
 * @endverbatim
 *
 * Numbers are in the byte order of the machine (little-endian on every supported
 * platform). The sequence grows by one per frame sent: a view can see what it missed.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ppokit::view {

inline constexpr char FrameMagic[4] = {'P', 'P', 'K', 'V'};
inline constexpr std::uint16_t FrameVersion = 1;

/** @brief One port of a frame, as written: its name and its number of values per agent. */
struct FramePort final
{
    std::string name;
    std::uint32_t size = 0;
};

/** @brief Writes frames for a fixed list of ports; its buffer is allocated once. */
class FrameWriter final
{
public:
    /**
     * @param ports The ports, in the order Write receives them.
     * @param maximumAgents Largest number of agents of a frame.
     */
    FrameWriter(std::vector<FramePort> ports, std::uint32_t maximumAgents) : ports_(std::move(ports))
    {
        std::size_t size = HeaderSize;
        for (FramePort const& port : ports_)
        {
            size += sizeof(std::uint16_t) + port.name.size() + sizeof(std::uint32_t)
                + std::size_t{maximumAgents} * port.size * sizeof(float);
        }
        buffer_.resize(size);
        maximumAgents_ = maximumAgents;
    }

    /**
     * @brief Writes one frame.
     * @param values One pointer per port, each to agents x size values.
     * @return The frame's size in bytes (Data() holds it), 0 for too many agents.
     */
    std::size_t Write(std::uint64_t sequence, std::uint32_t agents, float const* const* values) noexcept
    {
        if (agents > maximumAgents_)
        {
            return 0;
        }
        unsigned char* cursor = buffer_.data();
        Put(cursor, FrameMagic, sizeof(FrameMagic));
        PutValue(cursor, FrameVersion);
        PutValue(cursor, std::uint16_t{0});
        PutValue(cursor, sequence);
        PutValue(cursor, agents);
        PutValue(cursor, static_cast<std::uint32_t>(ports_.size()));
        for (std::size_t index = 0; index < ports_.size(); ++index)
        {
            FramePort const& port = ports_[index];
            PutValue(cursor, static_cast<std::uint16_t>(port.name.size()));
            Put(cursor, port.name.data(), port.name.size());
            PutValue(cursor, port.size);
            Put(cursor, values[index], std::size_t{agents} * port.size * sizeof(float));
        }
        return static_cast<std::size_t>(cursor - buffer_.data());
    }

    unsigned char const* Data() const noexcept { return buffer_.data(); }

private:
    static constexpr std::size_t HeaderSize = 4 + 2 + 2 + 8 + 4 + 4;

    static void Put(unsigned char*& cursor, void const* data, std::size_t size) noexcept
    {
        std::memcpy(cursor, data, size);
        cursor += size;
    }
    template<typename Value>
    static void PutValue(unsigned char*& cursor, Value value) noexcept
    {
        Put(cursor, &value, sizeof(value));
    }

    std::vector<FramePort> ports_;
    std::uint32_t maximumAgents_ = 0;
    std::vector<unsigned char> buffer_;
};

/** @brief A frame read back: its ports and their values. */
struct Frame final
{
    struct Port final
    {
        std::string name;
        std::uint32_t size = 0;
        /** @brief agents x size values. */
        std::vector<float> values;
    };

    std::uint64_t sequence = 0;
    std::uint32_t agents = 0;
    std::vector<Port> ports;

    /** @brief The port named `name`, or null. */
    Port const* Find(std::string_view name) const noexcept
    {
        for (Port const& port : ports)
        {
            if (port.name == name)
            {
                return &port;
            }
        }
        return nullptr;
    }

    /** @brief Value `index` of agent `agent` on port `name`, or `fallback` when absent. */
    float Value(std::string_view name, std::uint32_t agent, std::uint32_t index, float fallback = 0.F) const noexcept
    {
        Port const* const port = Find(name);
        if (port == nullptr || agent >= agents || index >= port->size)
        {
            return fallback;
        }
        return port->values[std::size_t{agent} * port->size + index];
    }
};

/** @brief Reads bytes in order, refusing to read past the end. */
class ByteReader final
{
public:
    ByteReader(unsigned char const* data, std::size_t size) noexcept : data_(data), size_(size) {}

    /** @brief Copies `count` bytes into `out`. @return False when fewer remain. */
    bool Take(void* out, std::size_t count) noexcept
    {
        if (position_ + count > size_)
        {
            return false;
        }
        std::memcpy(out, data_ + position_, count);
        position_ += count;
        return true;
    }

    /** @brief Reads `count` bytes as text. @return False when fewer remain. */
    bool TakeText(std::string& out, std::size_t count)
    {
        if (position_ + count > size_)
        {
            return false;
        }
        out.assign(reinterpret_cast<char const*>(data_ + position_), count);
        position_ += count;
        return true;
    }

    std::size_t Remaining() const noexcept { return size_ - position_; }

private:
    unsigned char const* data_;
    std::size_t size_;
    std::size_t position_ = 0;
};

/** @brief Reads a frame; nothing when the bytes are not a whole frame of this version. */
inline std::optional<Frame> ReadFrame(unsigned char const* data, std::size_t size)
{
    ByteReader reader(data, size);
    char magic[4] = {};
    std::uint16_t version = 0;
    std::uint16_t reserved = 0;
    std::uint32_t portCount = 0;
    Frame frame;
    bool const header = reader.Take(magic, sizeof(magic)) && std::memcmp(magic, FrameMagic, sizeof(magic)) == 0
        && reader.Take(&version, sizeof(version)) && version == FrameVersion && reader.Take(&reserved, sizeof(reserved))
        && reader.Take(&frame.sequence, sizeof(frame.sequence)) && reader.Take(&frame.agents, sizeof(frame.agents))
        && reader.Take(&portCount, sizeof(portCount));
    if (!header)
    {
        return std::nullopt;
    }
    for (std::uint32_t index = 0; index < portCount; ++index)
    {
        std::uint16_t nameLength = 0;
        Frame::Port port;
        if (!reader.Take(&nameLength, sizeof(nameLength)) || !reader.TakeText(port.name, nameLength)
            || !reader.Take(&port.size, sizeof(port.size)))
        {
            return std::nullopt;
        }
        std::size_t const count = std::size_t{frame.agents} * port.size;
        if (count > reader.Remaining() / sizeof(float))
        {
            return std::nullopt; // Checked before allocating: a damaged size cannot ask for much.
        }
        port.values.resize(count);
        if (!reader.Take(port.values.data(), port.values.size() * sizeof(float)))
        {
            return std::nullopt;
        }
        frame.ports.push_back(std::move(port));
    }
    return frame;
}

} // namespace ppokit::view
