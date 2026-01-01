#pragma once
#include <toml++/toml.h>
#include <filesystem>

namespace ppo::serialization {

template<typename Derived>
struct ArchiveBase {
                   
    template<typename T>
    void operator()(const char* name, T& value) {
        static_cast<Derived*>(this)->field(name, value);
    }

    template<typename T>
    void section(const char* name, T& obj) {
        auto sub = static_cast<Derived*>(this)->section_impl(name);
        if (sub) {
            obj.serialize(sub);
        }
    }

    Derived begin_section(const char* name) {
        return static_cast<Derived*>(this)->section_impl(name);
    }
};

struct TomlReadArchive : ArchiveBase<TomlReadArchive> {
    const toml::table* t{ nullptr };

    TomlReadArchive() = default;
    explicit TomlReadArchive(const toml::table& tbl) : t(&tbl) {}

    explicit operator bool() const { return t != nullptr; }

    // champ simple
    template<typename T>
    void field(const char* name, T& v) {
        if (!t) return;
        if (auto x = (*t)[name].value<T>())
            v = *x;
    }

    // std::array
    template<typename T, std::size_t N>
    void field(const char* name, std::array<T, N>& arr) {
        if (!t) return;
        if (auto a = (*t)[name].as_array()) {
            const std::size_t n = std::min(N, a->size());
            for (std::size_t i = 0; i < n; ++i)
                if (auto v = (*a)[i].value<T>())
                    arr[i] = *v;
        }
    }

    // section
    TomlReadArchive section_impl(const char* name) const {
        if (!t) return {};
        if (auto sub = (*t)[name].as_table())
            return TomlReadArchive{ *sub };
        return {}; // section absente
    }
};

struct TomlWriteArchive : ArchiveBase<TomlWriteArchive> {
    toml::table* t{ nullptr };

    explicit TomlWriteArchive(toml::table& tbl) : t(&tbl) {}

    explicit operator bool() const { return t != nullptr; }

    // champ simple
    template<typename T>
    void field(const char* name, const T& v) {
        t->insert_or_assign(name, v);
    }

    // std::array
    template<typename T, std::size_t N>
    void field(const char* name, const std::array<T, N>& arr) {
        toml::array a;
        for (const auto& v : arr)
            a.push_back(v);
        t->insert_or_assign(name, std::move(a));
    }

    // section
    TomlWriteArchive section_impl(const char* name) {
        auto sub =
            t->insert_or_assign(name, toml::table{})
            .first->second.as_table();
        return TomlWriteArchive{ *sub };
    }
};

template<typename T>
void LoadFromToml(const std::string& file, T& obj) {
    auto tbl = toml::parse_file(file);
    TomlReadArchive ar{ tbl };
    obj.serialize(ar);
}

template<typename T>
void SaveToToml(const std::string& file, const T& obj) {
    std::filesystem::path dir{ std::filesystem::path(file).parent_path() };
    if (!std::filesystem::exists(dir))
        std::filesystem::create_directories(dir);

    toml::table tbl;
    TomlWriteArchive ar{ tbl };
    const_cast<T&>(obj).serialize(ar);
    std::ofstream(file) << tbl;
}

}