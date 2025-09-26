#include "utils.h"
#include <chrono>

float get_current_time_in_mills() {
    const auto mills = static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    return mills;
}
