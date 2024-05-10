#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>

class LazyFileWriter {
private:
    std::filesystem::path m_filepath;
    std::fstream m_file_stream;
    void openFileIfNeeded() {
        if (!m_file_stream.is_open()) {
            m_file_stream.open(m_filepath, std::ios::out | std::ios::app);
            if (!m_file_stream.is_open()) {
                throw std::runtime_error("Failed to open the lazy file writer");
            }
        }
    }

public:
    LazyFileWriter(const std::filesystem::path& filepath) : m_filepath{filepath}{}

    ~LazyFileWriter() {
        if (m_file_stream.is_open()) {
            m_file_stream.close();
        }
    }

    template<typename TypeToWrite>
    void write(TypeToWrite&& data) {
        openFileIfNeeded();
        m_file_stream << data << std::endl;
    }
};