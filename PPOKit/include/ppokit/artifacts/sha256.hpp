/**
 * @file PPOKit/include/ppokit/artifacts/sha256.hpp
 * @brief SHA-256 digests (FIPS 180-4) of bytes and of files, in lowercase hexadecimal.
 * @details The identity of an artifact by its content: two files with the same digest hold
 * the same bytes. The CI tools record the same digests (Python hashlib.sha256).
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace ppokit::artifacts {

/** @brief Incremental SHA-256: Update with the bytes in order, then HexDigest once. */
class Sha256 final
{
public:
    Sha256() noexcept;

    /** @brief Adds `size` bytes to the message. */
    void Update(void const* data, std::size_t size) noexcept;

    /**
     * @brief Ends the message and returns its digest (64 lowercase hexadecimal characters).
     * @details The hasher is spent afterwards: a new message needs a new Sha256.
     */
    std::string HexDigest();

private:
    void Compress(std::uint8_t const* block) noexcept;

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> block_{};
    std::size_t blockSize_ = 0;
    std::uint64_t messageBytes_ = 0;
};

/** @brief The digest of `bytes`. */
std::string Sha256Hex(std::string_view bytes);

/**
 * @brief The digest of the file at `path`, read in chunks.
 * @throws std::runtime_error when the file cannot be opened or read.
 */
std::string Sha256OfFile(std::filesystem::path const& path);

} // namespace ppokit::artifacts
