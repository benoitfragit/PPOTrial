/**
 * @file PPOKit/src/artifacts/sha256.cpp
 * @brief SHA-256 compression function and padding (FIPS 180-4, section 6.2).
 */
#include "ppokit/artifacts/sha256.hpp"

#include <fstream>
#include <stdexcept>

namespace ppokit::artifacts {
namespace {

/** @brief The round constants: first 32 bits of the fractional parts of the cube roots of the first 64 primes. */
constexpr std::array<std::uint32_t, 64> kRound{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

/** @brief The initial hash: first 32 bits of the fractional parts of the square roots of the first 8 primes. */
constexpr std::array<std::uint32_t, 8> kInitial{
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

constexpr std::uint32_t RotateRight(std::uint32_t value, unsigned bits) noexcept
{
    return (value >> bits) | (value << (32U - bits));
}

/** @brief Size of the chunks a file is read in. */
constexpr std::size_t kFileChunk = 1U << 16U;

} // namespace

Sha256::Sha256() noexcept : state_(kInitial) {}

void Sha256::Compress(std::uint8_t const* block) noexcept
{
    // Message schedule: 16 big-endian words, then 48 derived ones.
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index)
    {
        std::uint8_t const* const bytes = block + 4 * index;
        words[index] = (static_cast<std::uint32_t>(bytes[0]) << 24U) | (static_cast<std::uint32_t>(bytes[1]) << 16U)
            | (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
    }
    for (std::size_t index = 16; index < 64; ++index)
    {
        std::uint32_t const s0 =
            RotateRight(words[index - 15], 7) ^ RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3U);
        std::uint32_t const s1 =
            RotateRight(words[index - 2], 17) ^ RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::array<std::uint32_t, 8> work = state_;
    for (std::size_t round = 0; round < 64; ++round)
    {
        std::uint32_t const e = work[4];
        std::uint32_t const a = work[0];
        std::uint32_t const sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
        std::uint32_t const choose = (e & work[5]) ^ (~e & work[6]);
        std::uint32_t const first = work[7] + sum1 + choose + kRound[round] + words[round];
        std::uint32_t const sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
        std::uint32_t const majority = (a & work[1]) ^ (a & work[2]) ^ (work[1] & work[2]);
        std::uint32_t const second = sum0 + majority;
        work[7] = work[6];
        work[6] = work[5];
        work[5] = work[4];
        work[4] = work[3] + first;
        work[3] = work[2];
        work[2] = work[1];
        work[1] = work[0];
        work[0] = first + second;
    }
    for (std::size_t index = 0; index < 8; ++index)
    {
        state_[index] += work[index];
    }
}

void Sha256::Update(void const* data, std::size_t size) noexcept
{
    auto const* bytes = static_cast<std::uint8_t const*>(data);
    messageBytes_ += size;
    for (std::size_t index = 0; index < size; ++index)
    {
        block_[blockSize_++] = bytes[index];
        if (blockSize_ == block_.size())
        {
            Compress(block_.data());
            blockSize_ = 0;
        }
    }
}

std::string Sha256::HexDigest()
{
    // Padding: one 1 bit, zeros, then the message length in bits on the last 8 bytes.
    std::uint64_t const messageBits = messageBytes_ * 8U;
    block_[blockSize_++] = 0x80U;
    if (blockSize_ > 56)
    {
        while (blockSize_ < block_.size())
        {
            block_[blockSize_++] = 0;
        }
        Compress(block_.data());
        blockSize_ = 0;
    }
    while (blockSize_ < 56)
    {
        block_[blockSize_++] = 0;
    }
    for (std::size_t index = 0; index < 8; ++index)
    {
        block_[56 + index] = static_cast<std::uint8_t>(messageBits >> (56U - 8U * index));
    }
    Compress(block_.data());
    blockSize_ = 0;

    constexpr char kHex[] = "0123456789abcdef";
    std::string digest;
    digest.reserve(64);
    for (std::uint32_t const word : state_)
    {
        for (int shift = 28; shift >= 0; shift -= 4)
        {
            digest.push_back(kHex[(word >> static_cast<unsigned>(shift)) & 0xFU]);
        }
    }
    return digest;
}

std::string Sha256Hex(std::string_view bytes)
{
    Sha256 hasher;
    hasher.Update(bytes.data(), bytes.size());
    return hasher.HexDigest();
}

std::string Sha256OfFile(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("cannot open '" + path.string() + "' to hash it");
    }
    Sha256 hasher;
    std::string chunk(kFileChunk, '\0');
    while (stream)
    {
        stream.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        hasher.Update(chunk.data(), static_cast<std::size_t>(stream.gcount()));
    }
    if (!stream.eof())
    {
        throw std::runtime_error("cannot read '" + path.string() + "' to hash it");
    }
    return hasher.HexDigest();
}

} // namespace ppokit::artifacts
