#include "sndfile.hh"
#include "cxxopts.hpp"

#include "rnnoise.h"
#include <cstdint>
#include <filesystem>
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <array>
#include <memory>
#include <utility>
#include <optional>

#include "lazy_file_writer.hpp"

template <auto DeleterFunction>
using CustomDeleter = std::integral_constant<decltype(DeleterFunction), DeleterFunction>;

template <typename ManagedType, auto Functor>
using PointerWrapper = std::unique_ptr<ManagedType, CustomDeleter<Functor>>;


inline constexpr std::size_t AUDIO_BUFFER_LENGTH = 480;
inline constexpr std::size_t NUM_CHANNELS = 1;
inline constexpr std::size_t SAMPLERATE = 48000;

inline constexpr float RNNOISE_PCM16_MULTIPLY_FACTOR = 32768.0f;

using RNNoiseDenoiseStatePtr = PointerWrapper<DenoiseState,rnnoise_destroy>;
using RnnModelPtr = PointerWrapper<RNNModel,rnnoise_model_free>;
using TSamplesBufferArray = std::array<float,AUDIO_BUFFER_LENGTH>;

RnnModelPtr rnn_model_ptr;
RNNoiseDenoiseStatePtr rnnoise_denoise_state_ptr;

static void initialize_rnnoise_library(){
    rnnoise_denoise_state_ptr.reset(rnnoise_create(nullptr));
}

static void normalize_to_rnnoise_expected_level(TSamplesBufferArray& samples_buffer){
    for(auto& sample : samples_buffer){
            sample *= RNNOISE_PCM16_MULTIPLY_FACTOR;
    }
}

static void denormalize_from_rnnoise_expected_level(TSamplesBufferArray& samples_buffer){
    for(auto& sample : samples_buffer){
            sample /= RNNOISE_PCM16_MULTIPLY_FACTOR;
    }
}

static void dump_vad_prob(LazyFileWriter& lazy_probe_dumper,float vad_probe_value){
    lazy_probe_dumper.write(vad_probe_value);
}
static void process_audio_recording(
    LazyFileWriter& lazy_vad_probe_writer,
    const std::filesystem::path& input_file,
    const std::filesystem::path& output_file
){
    SndfileHandle input_audio_file_handle{SndfileHandle(input_file.c_str())};

    spdlog::info("Opened input audio file:{}", input_file.c_str());
    spdlog::info("Number of channels:{}", input_audio_file_handle.channels());
    spdlog::info("Samplerate:{}", input_audio_file_handle.samplerate());

    SndfileHandle output_audio_file_handle{SndfileHandle{
        output_file.c_str(),
        SFM_WRITE,
        SF_FORMAT_WAV | SF_FORMAT_PCM_16,
        NUM_CHANNELS,
        SAMPLERATE
        }
    };

    
    static TSamplesBufferArray samples_buffer{};

    spdlog::info("Processing audio...");
    while (input_audio_file_handle.read (samples_buffer.data(), samples_buffer.size()) != 0) {
        normalize_to_rnnoise_expected_level(samples_buffer);
        float vad_prob = rnnoise_process_frame(rnnoise_denoise_state_ptr.get(), samples_buffer.data(), samples_buffer.data());
        dump_vad_prob(lazy_vad_probe_writer,vad_prob);
        denormalize_from_rnnoise_expected_level(samples_buffer);
        output_audio_file_handle.write(samples_buffer.data(),samples_buffer.size());
    }
    spdlog::info("Processing done. WAVE file can be found at: {}", output_file.c_str());
}

int main(int argc, char** argv){
    cxxopts::Options options("rnnoise_libsoundfile denoiser", "Simple runner of rnnoise over WAVe files with 48K samplerate");
    options.add_options()
    ("input", "Input file to process",cxxopts::value<std::filesystem::path>())
    ("output", "Output file", cxxopts::value<std::filesystem::path>())
    ("vad_probe", "Path to store output VAD prob data", cxxopts::value<std::filesystem::path>()->default_value(std::filesystem::current_path()/"vad_prob.txt"))
    ("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        fmt::print(options.help());
        exit(0);
    }


    using TOptionalPathHolder = std::optional<std::filesystem::path>;
    TOptionalPathHolder input_file_path_opt = result["input"].as<std::filesystem::path>();
    TOptionalPathHolder output_file_path_opt = result["output"].as<std::filesystem::path>();
    TOptionalPathHolder output_vad_probe = result["vad_probe"].as<std::filesystem::path>();

    try{
        input_file_path_opt = result["input"].as<std::filesystem::path>();
        output_file_path_opt = result["output"].as<std::filesystem::path>();
        output_vad_probe = result["vad_probe"].as<std::filesystem::path>();
    }
    catch(...){
        std::cerr << "Failed to obtain one of the required CMD args. Check help message below and verify passed options:" << std::endl;
        fmt::print(options.help());
        exit(-1);
    }

    LazyFileWriter vad_file_probe(output_vad_probe.value());
    initialize_rnnoise_library();
    process_audio_recording(
        vad_file_probe,
        input_file_path_opt.value(),
        output_file_path_opt.value()
    );
    return 0;
}