#pragma once

#include <array>
#include <cstdint>
#include <istream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ppo::examples::flight::detail {
/** @brief Read either libstdc++ or MSVC's mt19937 stream while preserving its sequence.
 * @param input Checkpoint stream positioned at the beginning of the RNG line.
 * @return Restored engine with the same future integer outputs.
 */
inline std::mt19937 ReadCheckpointRng(std::istream& input)
{
    constexpr std::size_t count=std::mt19937::state_size;
    constexpr std::uint32_t upper=0x80000000U,lower=0x7fffffffU,twist=0x9908b0dfU;
    std::string line;
    if(!std::getline(input>>std::ws,line)) throw std::runtime_error("FlightControl3D: missing RNG state");
    std::istringstream source(line);
    std::array<std::uint32_t,count> state{};
    for(auto& value:state)
    {
        std::uint64_t word{};
        if(!(source>>word) || word>UINT32_MAX) throw std::runtime_error("FlightControl3D: invalid RNG state");
        value=static_cast<std::uint32_t>(word);
    }
    source>>std::ws;
    bool const hasPosition=source.peek()!=std::char_traits<char>::eof();
    std::size_t position=count;
    if(hasPosition && (!(source>>position) || position>count))
        throw std::runtime_error("FlightControl3D: invalid RNG position");
    source>>std::ws;
    if(!source.eof()) throw std::runtime_error("FlightControl3D: trailing RNG state");

    // MSVC serializes the 624 preceding recurrence words, while libstdc++
    // serializes a block and the index of its next word. Reverse the in-place
    // twist to recover the preceding words when the block is partly consumed.
    if(hasPosition && position<count)
    {
        std::array<std::uint32_t,count> previous{};
        for(std::size_t i=count;i-->0;)
        {
            auto const reference=i<227 ? previous[i+397] : state[i-227];
            auto const mixed=state[i]^reference;
            auto const bit=mixed>>31U;
            auto const joined=((mixed^(bit ? twist : 0U))<<1U)|bit;
            previous[i]=(previous[i]&lower)|(joined&upper);
            if(i+1<count) previous[i+1]=(previous[i+1]&upper)|(joined&lower);
            else if((joined&lower)!=(state[0]&lower))
                throw std::runtime_error("FlightControl3D: inconsistent RNG state");
        }
        std::array<std::uint32_t,count> window{};
        std::size_t out=0;
        for(std::size_t i=position;i<count;++i) window[out++]=previous[i];
        for(std::size_t i=0;i<position;++i) window[out++]=state[i];
        state=window;
    }

    std::ostringstream probe;
    probe<<std::mt19937{};
    std::istringstream probeWords(probe.str());
    std::uint64_t word{};std::size_t nativeCount=0;
    while(probeWords>>word) ++nativeCount;
    if(nativeCount!=count && nativeCount!=count+1)
        throw std::runtime_error("FlightControl3D: unsupported native RNG format");
    std::ostringstream converted;
    for(auto const value:state) converted<<value<<' ';
    if(nativeCount==count+1) converted<<count;
    std::istringstream native(converted.str());
    std::mt19937 result;
    if(!(native>>result)) throw std::runtime_error("FlightControl3D: invalid RNG state");
    return result;
}
} // namespace ppo::examples::flight::detail
